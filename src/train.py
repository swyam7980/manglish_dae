"""Train one backbone with LoRA on noisy -> clean Manglish. Saves only the adapter.

Memory fail-safes (all keep the effective batch at 16): --auto-batch picks the largest micro-batch that fits the
free VRAM, an out-of-memory error halves it and retries, and --resume checkpoints so a long run survives a crash.
"""
import argparse
import contextlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from config import PROC_DIR, OUT_DIR, SEED, MAX_LEN, EPOCHS, BATCH_SIZE, GRAD_ACCUM, EFFECTIVE_BATCH, LR, WARMUP_RATIO
import memory
from memory import pick_micro_batch, vram_budget_bytes, next_smaller, free_cuda
from models import build_lora_model, load_tokenizer, input_prefix, pick_dtype, run_tag, count_params

MAX_OOM_RETRIES = 4


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def resolve_train_file(name):
    p = Path(name)
    return p if p.is_absolute() or p.exists() else PROC_DIR / name


def autocast_for(prec):
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    if prec.get("fp16"):
        return torch.autocast("cuda", dtype=torch.float16)
    if prec.get("bf16"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def make_probe(model, ds, collator, prec):
    """try_batch(bs): one forward+backward on the bs longest training examples, the worst case for memory."""
    order = sorted(range(len(ds)), key=lambda i: -(len(ds[i]["input_ids"]) + len(ds[i]["labels"])))
    dev = next(model.parameters()).device

    def try_batch(bs):
        batch = collator([ds[i] for i in order[:bs]])
        batch = {k: v.to(dev) for k, v in batch.items()}
        model.train()
        with autocast_for(prec):
            loss = model(**batch).loss
        loss.backward()
        model.zero_grad(set_to_none=True)
    return try_batch


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mbart", "mt5", "byt5"])
    ap.add_argument("--size", default="base", help="mt5/byt5 size: small|base (ignored for mbart)")
    ap.add_argument("--base-path", help="local model directory instead of the Hub id")
    ap.add_argument("--train-file", default="train.jsonl", help="file in data/processed (or a path) with clean/noisy rows")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-train", type=int, default=0, help="cap train rows (smoke tests)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (saves VRAM, slower)")
    ap.add_argument("--auto-batch", action="store_true", help="probe VRAM and use the largest micro-batch that fits")
    ap.add_argument("--resume", action="store_true", help="checkpoint every --save-steps and resume from the latest one")
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=LR, help="learning rate override (report any change as a deviation)")
    ap.add_argument("--seed", type=int, default=SEED, help="training seed only; the data split and noise stay fixed")
    ap.add_argument("--tag-suffix", default="", help="appended to the output folder name (smoke tests)")
    a = ap.parse_args(argv)

    train_path = resolve_train_file(a.train_file)
    if not train_path.exists():
        sys.exit(f"{train_path} missing. Run data_prep.py (and extra_data.py for extra data) first.")
    set_seed(a.seed)
    tok = load_tokenizer(a.model, a.size, a.base_path)
    pre = input_prefix(a.model)
    rows = read_jsonl(train_path)
    if a.max_train:
        rows = rows[: a.max_train]
    ds = Dataset.from_list([{"noisy": r["noisy"], "clean": r["clean"]} for r in rows])

    def enc(b):
        x = tok([pre + s for s in b["noisy"]], max_length=MAX_LEN, truncation=True)
        x["labels"] = tok(text_target=b["clean"], max_length=MAX_LEN, truncation=True)["input_ids"]
        return x

    ds = ds.map(enc, batched=True, remove_columns=ds.column_names)
    _, prec = pick_dtype(a.model)
    collator_for = lambda m: DataCollatorForSeq2Seq(tok, model=m, label_pad_token_id=-100)

    model = build_lora_model(a.model, a.size, a.base_path, a.grad_ckpt)
    trainable, total = count_params(model)
    bs, accum = BATCH_SIZE, GRAD_ACCUM
    if a.auto_batch and torch.cuda.is_available():
        model.to("cuda")  # the probe must run where training will, or it measures nothing
        budget = vram_budget_bytes()
        bs = pick_micro_batch(make_probe(model, ds, collator_for(model), prec), EFFECTIVE_BATCH, budget)
        accum = EFFECTIVE_BATCH // bs
        peaks = {k: round(v / 1e9, 2) for k, v in memory.LAST_PROBE.items()}
        print(f"[auto-batch] micro-batch {bs} x accumulation {accum} (effective {EFFECTIVE_BATCH}); "
              f"budget {budget / 1e9:.2f} GB, probe peaks GB {peaks}", flush=True)
        set_seed(a.seed)

    steps_per_epoch = math.ceil(len(ds) / EFFECTIVE_BATCH)
    total_steps = a.max_steps if a.max_steps > 0 else steps_per_epoch * a.epochs
    warmup_steps = max(1, round(WARMUP_RATIO * total_steps))  # explicit steps: works on every transformers version
    out = OUT_DIR / (run_tag(a.model, a.size) + a.tag_suffix)

    def make_trainer(m, micro, acc):
        args = Seq2SeqTrainingArguments(
            output_dir=str(out / "ckpt"), per_device_train_batch_size=micro,
            gradient_accumulation_steps=acc, learning_rate=a.lr, num_train_epochs=a.epochs,
            max_steps=a.max_steps, logging_steps=min(10, max(1, total_steps // 3)),
            save_strategy="steps" if a.resume else "no", save_steps=a.save_steps, save_total_limit=1,
            report_to="none", seed=a.seed, lr_scheduler_type="linear", warmup_steps=warmup_steps, **prec)
        t = Seq2SeqTrainer(model=m, args=args, train_dataset=ds, data_collator=collator_for(m))
        # With PEFT-wrapped models the Trainer skips dividing the loss by GRAD_ACCUM, so logged losses come out
        # GRAD_ACCUM times too large. Forcing it to divide makes the logged loss a true per-token mean.
        t.model_accepts_loss_kwargs = False
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
        except torch.cuda.OutOfMemoryError:
            del trainer
            nxt = next_smaller(bs)
            if nxt is None or len(retries) >= MAX_OOM_RETRIES:
                sys.exit(f"CUDA out of memory even at micro-batch {bs}. Try --grad-ckpt, a smaller --size for mt5/byt5 "
                         "(small), close other GPU programs, or run on Kaggle/Colab.")
            retries.append({"from": bs, "to": nxt})
            print(f"[oom] micro-batch {bs} ran out of memory; retrying at {nxt} x {EFFECTIVE_BATCH // nxt}", flush=True)
            bs, accum = nxt, EFFECTIVE_BATCH // nxt
            del model
            free_cuda()
            set_seed(a.seed)
            model = build_lora_model(a.model, a.size, a.base_path, a.grad_ckpt)
    mins = (time.time() - t0) / 60

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    if not losses or any(not math.isfinite(x) for x in losses):
        sys.exit(f"Training loss is NaN/inf or never logged ({losses[:5]}). For T5-family this is usually fp16 overflow; "
                 "check the precision flags in models.pick_dtype.")
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    if a.resume:
        shutil.rmtree(out / "ckpt", ignore_errors=True)
    info = {"model": a.model, "size": a.size if a.model != "mbart" else "large", "train_loss": res.training_loss,
            "first_logged_loss": losses[0], "last_logged_loss": losses[-1],
            "minutes": round(mins, 2), "steps": trainer.state.global_step,
            "trainable_params": trainable, "total_params": total,
            "trainable_pct": round(100 * trainable / total, 3),
            "precision": next(iter(prec), "fp32"),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
            "train_file": train_path.name, "train_rows": len(ds), "micro_batch": bs, "grad_accum": accum,
            "oom_retries": retries, "resumed_from": resumed_from, "learning_rate": a.lr, "seed": a.seed,
            "loss_history": losses}
    (out / "train_info.json").write_text(json.dumps(info, indent=2))
    print("DONE", {k: v for k, v in info.items() if k != "loss_history"})
    return info


if __name__ == "__main__":
    main()
