"""Train one backbone with LoRA on noisy -> clean Manglish. Saves only the adapter."""
import argparse
import json
import time

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed

from config import PROC_DIR, OUT_DIR, SEED, MAX_LEN, EPOCHS, BATCH_SIZE, GRAD_ACCUM, LR
from models import build_lora_model, load_tokenizer, input_prefix, pick_dtype


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mbart", "mt5", "byt5"])
    ap.add_argument("--size", default="base", help="mt5/byt5 size: small|base (ignored for mbart)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-train", type=int, default=0, help="cap train rows (smoke tests)")
    ap.add_argument("--max-steps", type=int, default=-1)
    a = ap.parse_args()

    set_seed(SEED)
    tok = load_tokenizer(a.model, a.size)
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
    model = build_lora_model(a.model, a.size)
    _, prec = pick_dtype(a.model)
    out = OUT_DIR / f"{a.model}_{a.size if a.model != 'mbart' else 'large'}"
    args = Seq2SeqTrainingArguments(
        output_dir=str(out / "ckpt"), per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, num_train_epochs=a.epochs,
        max_steps=a.max_steps, logging_steps=25, save_strategy="no", report_to="none",
        seed=SEED, lr_scheduler_type="linear", warmup_ratio=0.03, **prec)
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds,
                             data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100))
    t0 = time.time()
    res = trainer.train()
    mins = (time.time() - t0) / 60
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    info = {"model": a.model, "size": a.size, "train_loss": res.training_loss, "minutes": round(mins, 2),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0}
    (out / "train_info.json").write_text(json.dumps(info, indent=2))
    print("DONE", info)


if __name__ == "__main__":
    main()
