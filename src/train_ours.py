"""Train our own model, Manglish-ByT5: one ByT5-base that does two tasks, chosen by a prefix.

  clean: <noisy Manglish>  ->  clean Manglish            (all three sources, same rows as train_plus_all)
  ml: <Manglish>           ->  the sentence in Malayalam script  (Dakshina pairs + lexicon words, see ours_data.py)

What the comparison taught us and how it is used here:
  * ByT5 was the best model with more data, so it is the backbone, and all three sources are used.
  * Its weak spot was heavier and unseen noise, so every epoch draws fresh noise from a mix of all robustness
    noise types (NOISE_MIX) instead of one fixed corruption. The comparison models never saw vowel dropping or
    phonetic respelling; the report states that when it puts them side by side.
  * Its loss was still falling after 3 epochs, so it trains for 4 with a wider LoRA (attention + feed-forward).
The adapter is merged into the base weights at the end, so the result is one ordinary model folder:
outputs/ours_byt5/model (loadable with AutoModelForSeq2SeqLM, no peft needed).

English (--en-file ours_en_train.jsonl, see en_data.py), trained on top of that model (--base-path):
  en: <Manglish, some words in English>  ->  English
  ml: <Manglish, some words in English>  ->  Malayalam script with those words translated ("tomorrow" -> നാളെ)
with the original clean: and ml: rows replayed so the model keeps them:
  python train_ours.py --base-path <WORK>/outputs/ours_byt5/model --en-file ours_en_train.jsonl --epochs 1       --tag ours_byt5_en --auto-batch --grad-ckpt --resume
"""
import argparse
import json
import math
import random
import shutil
import sys
import time

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from config import PROC_DIR, OUT_DIR, SEED, EFFECTIVE_BATCH, WARMUP_RATIO, LORA_DROPOUT
import memory
from memory import pick_micro_batch, vram_budget_bytes, next_smaller, free_cuda, is_oom
from models import load_base, load_tokenizer, pick_dtype, count_params
from noise import corrupt, CONDITIONS
from train import make_probe, read_jsonl

TASK_PREFIX = {"clean": "clean: ", "ml": "ml: ", "en": "en: "}
# share of clean-task examples per noise type; "none" teaches the model to leave clean text alone
NOISE_MIX = {"base": 0.45, "heavy": 0.15, "extreme": 0.05, "vowel_drop": 0.10, "phonetic": 0.10,
             "phonetic_plus_base": 0.10, "none": 0.05}
# script-task inputs: typed cleanly, with the usual typos, or spelled another way
ML_INPUT_MIX = {"none": 0.50, "base": 0.35, "phonetic": 0.15}
# share of swappable Malayalam words written in English: en inputs are often all Manglish, mixed ml inputs never are
EN_SWAP_MIX = {0.0: 0.4, 0.2: 0.3, 0.5: 0.3}
MLX_SWAP_MIX = {0.3: 0.5, 0.6: 0.5}
# script-only training (English goes to a separate translator): plain Manglish too, which is most of what people type
SCRIPT_SWAP_MIX = {0.0: 0.5, 0.3: 0.25, 0.6: 0.25}
MAX_SRC = 192             # bytes; elongated noisy input can pass the 128 of the comparison
MAX_TGT = 384             # Malayalam is 3 bytes per character in UTF-8 (longest test target is 357)
EPOCHS = 4
LR = 3e-4
LORA_R, LORA_ALPHA = 32, 64
LORA_TARGETS = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
TAG = "ours_byt5"
PROBE_CANDIDATES = (8, 4, 2, 1)


def noisy_version(text, kind, rng):
    if kind == "none":
        return text
    if kind == "base":
        return corrupt(text, rng)
    return CONDITIONS[kind](text, rng) or text


def pick(mix, rng):
    r, acc = rng.random(), 0.0
    for k, w in mix.items():
        acc += w
        if r < acc:
            return k
    return next(reversed(mix))


def build_examples(clean_rows, ml_rows, epochs, seed=SEED, first_epoch=0, en_rows=(), mlx_rows=(), mlx_mix=MLX_SWAP_MIX):
    """Fresh noise for every epoch, deterministic in (seed, epoch, row): a resumed run sees the same data.
    en_rows/mlx_rows are en_data.py rows; each epoch draws a new spelling per word and a new set of English swaps."""
    from en_data import render
    out = []
    for ep in range(first_epoch, first_epoch + epochs):
        for i, s in enumerate(clean_rows):
            rng = random.Random(f"{seed}-clean-{ep}-{i}")
            out.append({"src": TASK_PREFIX["clean"] + noisy_version(s, pick(NOISE_MIX, rng), rng), "tgt": s})
        for i, r in enumerate(ml_rows):
            rng = random.Random(f"{seed}-ml-{ep}-{i}")
            out.append({"src": TASK_PREFIX["ml"] + noisy_version(r["roman"], pick(ML_INPUT_MIX, rng), rng),
                        "tgt": r["native"]})
        for i, r in enumerate(en_rows):
            rng = random.Random(f"{seed}-en-{ep}-{i}")
            text = render(r["toks"], rng, pick(EN_SWAP_MIX, rng))
            out.append({"src": TASK_PREFIX["en"] + noisy_version(text, pick(ML_INPUT_MIX, rng), rng), "tgt": r["en"]})
        for i, r in enumerate(mlx_rows):
            rng = random.Random(f"{seed}-mlx-{ep}-{i}")
            text = render(r["toks"], rng, pick(mlx_mix, rng))
            out.append({"src": TASK_PREFIX["ml"] + noisy_version(text, pick(ML_INPUT_MIX, rng), rng), "tgt": r["ml"]})
    return out


def sample_rows(rows, n, seed, tag):
    """n rows (all when n is 0 or larger), chosen reproducibly."""
    if not n or n >= len(rows):
        return list(rows)
    return random.Random(f"{seed}-{tag}").sample(list(rows), n)


def build_model(base_path=None, grad_ckpt=False):
    model = load_base("byt5", "base", base_path)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                     target_modules=LORA_TARGETS)
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-path", help="local google/byt5-base directory instead of the Hub id")
    ap.add_argument("--train-file", default="train_plus_all.jsonl")
    ap.add_argument("--ml-file", default="ours_ml_train.jsonl")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--max-train", type=int, default=0, help="cap rows per task (smoke tests)")
    ap.add_argument("--en-file", help="en_data.py output: adds the en: task and code-mixed ml: examples")
    ap.add_argument("--en-max", type=int, default=0, help="English pairs per epoch (0 = all)")
    ap.add_argument("--mlx-max", type=int, default=40000, help="code-mixed ml: examples per epoch, from pairs with swaps")
    ap.add_argument("--replay-max", type=int, default=0, help="cap on the clean: and ml: rows per epoch (0 = all)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--auto-batch", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default=TAG)
    a = ap.parse_args(argv)

    for f in (a.train_file, a.ml_file, *([a.en_file] if a.en_file else [])):
        if not (PROC_DIR / f).exists():
            sys.exit(f"{PROC_DIR / f} missing. Run extra_data.py (train_plus_all) and ours_data.py first.")
    set_seed(a.seed)
    clean_rows = [r["clean"] for r in read_jsonl(PROC_DIR / a.train_file)]
    ml_rows = read_jsonl(PROC_DIR / a.ml_file)
    en_rows, mlx_rows = [], []
    if a.en_file:
        all_en = read_jsonl(PROC_DIR / a.en_file)
        en_rows = sample_rows(all_en, a.en_max, a.seed, "en")
        mlx_rows = sample_rows([r for r in all_en if any(sw for _, sw in r["toks"])], a.mlx_max, a.seed, "mlx")
        clean_rows = sample_rows(clean_rows, a.replay_max, a.seed, "clean")
        ml_rows = sample_rows(ml_rows, a.replay_max, a.seed, "ml")
    if a.max_train:
        clean_rows, ml_rows = clean_rows[: a.max_train], ml_rows[: a.max_train]
        en_rows, mlx_rows = en_rows[: a.max_train], mlx_rows[: a.max_train]
    examples = build_examples(clean_rows, ml_rows, a.epochs, a.seed, en_rows=en_rows, mlx_rows=mlx_rows)

    tok = load_tokenizer("byt5", "base", a.base_path)
    ds = Dataset.from_list(examples)

    def enc(b):
        x = tok(b["src"], max_length=MAX_SRC, truncation=True)
        x["labels"] = tok(text_target=b["tgt"], max_length=MAX_TGT, truncation=True)["input_ids"]
        return x

    ds = ds.map(enc, batched=True, remove_columns=ds.column_names)
    _, prec = pick_dtype("byt5")
    collator_for = lambda m: DataCollatorForSeq2Seq(tok, model=m, label_pad_token_id=-100)

    model = build_model(a.base_path, a.grad_ckpt)
    trainable, total = count_params(model)
    bs = 2
    if a.auto_batch and torch.cuda.is_available():
        model.to("cuda")
        budget = vram_budget_bytes()
        # 576-byte worst cases: 16 never fits in 6 GB, and probing it can trip the driver itself
        bs = pick_micro_batch(make_probe(model, ds, collator_for(model), prec), EFFECTIVE_BATCH, budget,
                              candidates=PROBE_CANDIDATES)
        peaks = {k: round(v / 1e9, 2) for k, v in memory.LAST_PROBE.items()}
        print(f"[auto-batch] micro-batch {bs} x {EFFECTIVE_BATCH // bs}; budget {budget / 1e9:.2f} GB, peaks {peaks}",
              flush=True)
        set_seed(a.seed)
    accum = EFFECTIVE_BATCH // bs

    # The epochs are already laid out in `examples`, so the Trainer makes a single pass over them.
    total_steps = a.max_steps if a.max_steps > 0 else math.ceil(len(ds) / EFFECTIVE_BATCH)
    warmup_steps = max(1, round(WARMUP_RATIO * total_steps))
    out = OUT_DIR / a.tag

    def make_trainer(m, micro, acc):
        args = Seq2SeqTrainingArguments(
            output_dir=str(out / "ckpt"), per_device_train_batch_size=micro, gradient_accumulation_steps=acc,
            learning_rate=a.lr, num_train_epochs=1, max_steps=a.max_steps,
            logging_steps=min(25, max(1, total_steps // 3)), save_strategy="steps" if a.resume else "no",
            save_steps=a.save_steps, save_total_limit=1, report_to="none", seed=a.seed,
            lr_scheduler_type="linear", warmup_steps=warmup_steps,
            train_sampling_strategy="group_by_length",  # words and long script sentences mixed: cuts padding
            **prec)
        t = Seq2SeqTrainer(model=m, args=args, train_dataset=ds, data_collator=collator_for(m))
        t.model_accepts_loss_kwargs = False  # see train.py: keeps the logged loss a true per-token mean
        return t

    retries, resumed_from = [], None
    t0 = time.time()
    while True:
        trainer = make_trainer(model, bs, accum)
        ckpt_dir = out / "ckpt"
        resumed_from = get_last_checkpoint(str(ckpt_dir)) if a.resume and ckpt_dir.is_dir() else None
        if resumed_from:
            print(f"[resume] continuing from {resumed_from}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            res = trainer.train(resume_from_checkpoint=resumed_from)
            break
        except RuntimeError as e:
            if not is_oom(e):
                raise
            del trainer
            nxt = next_smaller(bs)
            if nxt is None or len(retries) >= 4:
                sys.exit(f"CUDA out of memory even at micro-batch {bs}. Try --grad-ckpt or close other GPU programs.")
            retries.append({"from": bs, "to": nxt})
            print(f"[oom] micro-batch {bs} ran out of memory; retrying at {nxt} x {EFFECTIVE_BATCH // nxt}", flush=True)
            bs, accum = nxt, EFFECTIVE_BATCH // nxt
            del model
            free_cuda()
            set_seed(a.seed)
            model = build_model(a.base_path, a.grad_ckpt)
    mins = (time.time() - t0) / 60

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    if not losses or any(not math.isfinite(x) for x in losses):
        sys.exit(f"Training loss is NaN/inf or never logged ({losses[:5]}).")
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
    model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "model", safe_serialization=True)
    tok.save_pretrained(out / "model")
    if a.resume:
        shutil.rmtree(out / "ckpt", ignore_errors=True)
    tasks = list(TASK_PREFIX) if a.en_file else ["clean", "ml"]
    info = {"model": "byt5", "size": "base", "tasks": tasks, "task_prefix": {t: TASK_PREFIX[t] for t in tasks},
            "base": a.base_path or "google/byt5-base", "en_rows": len(en_rows), "mlx_rows": len(mlx_rows),
            "en_swap_mix": EN_SWAP_MIX if a.en_file else None, "mlx_swap_mix": MLX_SWAP_MIX if a.en_file else None,
            "noise_mix": NOISE_MIX, "ml_input_mix": ML_INPUT_MIX, "epochs": a.epochs, "train_loss": res.training_loss,
            "first_logged_loss": losses[0], "last_logged_loss": losses[-1], "minutes": round(mins, 2),
            "steps": trainer.state.global_step, "trainable_params": trainable, "total_params": total,
            "trainable_pct": round(100 * trainable / total, 3), "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "lora_targets": LORA_TARGETS, "precision": next(iter(prec), "fp32"), "peak_gpu_gb": peak,
            "clean_rows": len(clean_rows), "ml_rows": len(ml_rows), "examples": len(ds), "micro_batch": bs,
            "grad_accum": accum, "oom_retries": retries, "resumed_from": resumed_from, "learning_rate": a.lr,
            "seed": a.seed, "max_src": MAX_SRC, "max_tgt": MAX_TGT, "loss_history": losses}
    (out / "train_info.json").write_text(json.dumps(info, indent=2))
    print("DONE", {k: v for k, v in info.items() if k != "loss_history"})
    return info


if __name__ == "__main__":
    main()
