"""Reconstruct the validation set with a trained adapter and score it. Also scores the no-model baseline."""
import argparse
import json

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM

from config import PROC_DIR, OUT_DIR, RES_DIR, MAX_LEN
from metrics import score
from models import model_name, load_tokenizer, input_prefix, gen_kwargs, pick_dtype


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def baseline():
    val = read_jsonl(PROC_DIR / "val.jsonl")
    refs = [r["clean"] for r in val]
    res = score([r["noisy"] for r in val], refs)
    RES_DIR.mkdir(exist_ok=True, parents=True)
    (RES_DIR / "metrics_identity.json").write_text(json.dumps(res, indent=2))
    print("identity (copy noisy input):", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mbart", "mt5", "byt5"])
    ap.add_argument("--size", default="base")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--baseline", action="store_true", help="score the copy-the-noisy-input baseline and exit")
    a = ap.parse_args()
    if a.baseline:
        return baseline()
    if not a.model:
        ap.error("--model is required unless --baseline is given")

    val = read_jsonl(PROC_DIR / "val.jsonl")
    if a.max_val:
        val = val[: a.max_val]
    tag = f"{a.model}_{a.size if a.model != 'mbart' else 'large'}"
    ad = OUT_DIR / tag / "adapter"
    tok = load_tokenizer(a.model, a.size)
    dtype, _ = pick_dtype(a.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForSeq2SeqLM.from_pretrained(model_name(a.model, a.size), torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, ad).to(dev).eval()
    pre, gk = input_prefix(a.model), gen_kwargs(a.model, tok)
    preds = []
    for i in range(0, len(val), a.batch):
        chunk = [pre + r["noisy"] for r in val[i:i + a.batch]]
        x = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(dev)
        with torch.no_grad():
            y = model.generate(**x, max_length=MAX_LEN, num_beams=a.beams, **gk)
        preds += tok.batch_decode(y, skip_special_tokens=True)
    refs = [r["clean"] for r in val]
    res = score(preds, refs)
    RES_DIR.mkdir(exist_ok=True, parents=True)
    pd.DataFrame({"noisy": [r["noisy"] for r in val], "clean": refs, "pred": preds}).to_csv(
        RES_DIR / f"preds_{tag}.csv", index=False, encoding="utf-8")
    (RES_DIR / f"metrics_{tag}.json").write_text(json.dumps(res, indent=2))
    print(tag, res)


if __name__ == "__main__":
    main()
