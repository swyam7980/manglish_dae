"""Train one backbone with LoRA on noisy -> clean Manglish. Saves only the adapter."""
import argparse
import json
import math
import sys
import time

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed

from config import PROC_DIR, OUT_DIR, SEED, MAX_LEN, EPOCHS, BATCH_SIZE, GRAD_ACCUM, LR, WARMUP_RATIO
from models import build_lora_model, load_tokenizer, input_prefix, pick_dtype, run_tag, count_params


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mbart", "mt5", "byt5"])
    ap.add_argument("--size", default="base", help="mt5/byt5 size: small|base (ignored for mbart)")
    ap.add_argument("--base-path", help="local model directory instead of the Hub id")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-train", type=int, default=0, help="cap train rows (smoke tests)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (saves VRAM, slower)")
    ap.add_argument("--lr", type=float, default=LR, help="learning rate override (report any change as a deviation)")
    ap.add_argument("--seed", type=int, default=SEED, help="training seed only; the data split and noise stay fixed")
    ap.add_argument("--tag-suffix", default="", help="appended to the output folder name (smoke tests)")
    a = ap.parse_args(argv)

    if not (PROC_DIR / "train.jsonl").exists():
        sys.exit(f"{PROC_DIR / 'train.jsonl'} missing. Run data_prep.py first.")
    set_seed(a.seed)
    tok = load_tokenizer(a.model, a.size, a.base_path)
    pre = input_prefix(a.model)
    rows = read_jsonl(PROC_DIR / "train.jsonl")
    if a.max_train:
        rows = rows[: a.max_train]
    ds = Dataset.from_list(rows)

    def enc(b):
        x = tok([pre + s for s in b["noisy"]], max_length=MAX_LEN, truncation=True)
        x["labels"] = tok(text_target=b["clean"], max_length=MAX_LEN, truncation=True)["input_ids"]
        return x

    ds = ds.map(enc, batched=True, remove_columns=ds.column_names)
    model = build_lora_model(a.model, a.size, a.base_path, a.grad_ckpt)
    trainable, total = count_params(model)
    _, prec = pick_dtype(a.model)
    steps_per_epoch = math.ceil(len(ds) / (BATCH_SIZE * GRAD_ACCUM))
    total_steps = a.max_steps if a.max_steps > 0 else steps_per_epoch * a.epochs
    warmup_steps = max(1, round(WARMUP_RATIO * total_steps))  # explicit steps: works on every transformers version
    out = OUT_DIR / (run_tag(a.model, a.size) + a.tag_suffix)
    args = Seq2SeqTrainingArguments(
        output_dir=str(out / "ckpt"), per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM, learning_rate=a.lr, num_train_epochs=a.epochs,
        max_steps=a.max_steps, logging_steps=min(10, max(1, total_steps // 3)), save_strategy="no", report_to="none",
        seed=a.seed, lr_scheduler_type="linear", warmup_steps=warmup_steps, **prec)
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds,
                             data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100))
    # With PEFT-wrapped models the Trainer skips dividing the loss by GRAD_ACCUM, so logged losses come out
    # GRAD_ACCUM times too large. Forcing it to divide makes the logged loss a true per-token mean.
    trainer.model_accepts_loss_kwargs = False
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        res = trainer.train()
    except torch.cuda.OutOfMemoryError:
        sys.exit("CUDA out of memory. Try --grad-ckpt, a smaller --size for mt5/byt5 (small), or run on Kaggle/Colab.")
    mins = (time.time() - t0) / 60

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    if not losses or any(not math.isfinite(x) for x in losses):
        sys.exit(f"Training loss is NaN/inf or never logged ({losses[:5]}). For T5-family this is usually fp16 overflow; "
                 "check the precision flags in models.pick_dtype.")
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    info = {"model": a.model, "size": a.size if a.model != "mbart" else "large", "train_loss": res.training_loss,
            "first_logged_loss": losses[0], "last_logged_loss": losses[-1],
            "minutes": round(mins, 2), "steps": trainer.state.global_step,
            "trainable_params": trainable, "total_params": total,
            "trainable_pct": round(100 * trainable / total, 3),
            "precision": next(iter(prec), "fp32"),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
            "loss_history": losses}
    (out / "train_info.json").write_text(json.dumps(info, indent=2))
    print("DONE", {k: v for k, v in info.items() if k != "loss_history"})
    return info


if __name__ == "__main__":
    main()
