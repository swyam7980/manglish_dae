"""Reconstruct the validation set from a saved adapter and score it. Also scores the no-model baseline."""
import argparse
import json
import sys

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoTokenizer

from config import PROC_DIR, OUT_DIR, RES_DIR, MAX_LEN
from metrics import score
from models import load_base, input_prefix, gen_kwargs, run_tag


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_val(max_val=0):
    p = PROC_DIR / "val.jsonl"
    if not p.exists():
        sys.exit(f"{p} missing. Run data_prep.py first.")
    val = read_jsonl(p)
    return val[:max_val] if max_val else val


def baseline(max_val=0):
    val = load_val(max_val)
    res = score([r["noisy"] for r in val], [r["clean"] for r in val])
    RES_DIR.mkdir(exist_ok=True, parents=True)
    (RES_DIR / "metrics_identity.json").write_text(json.dumps(res, indent=2))
    print("identity (copy noisy input):", res)
    return res


def evaluate(model_key, size="base", beams=1, batch=16, max_val=0, base_path=None, tag_suffix=""):
    tag = run_tag(model_key, size) + tag_suffix
    ad = OUT_DIR / tag / "adapter"
    if not (ad / "adapter_config.json").exists():
        sys.exit(f"No adapter at {ad}. Train first: python train.py --model {model_key} --size {size}")
    val = load_val(max_val)
    # tokenizer and weights come from disk (the saved adapter), not from the training process
    tok = AutoTokenizer.from_pretrained(ad)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = PeftModel.from_pretrained(load_base(model_key, size, base_path), ad).to(dev).eval()
    pre, gk = input_prefix(model_key), gen_kwargs(model_key, tok)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    preds = []
    for i in range(0, len(val), batch):
        chunk = [pre + r["noisy"] for r in val[i:i + batch]]
        x = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(dev)
        with torch.no_grad():
            y = model.generate(**x, max_length=MAX_LEN, num_beams=beams, do_sample=False, **gk)
        preds += [p.strip() for p in tok.batch_decode(y, skip_special_tokens=True)]
    refs = [r["clean"] for r in val]
    res = score(preds, refs)
    res["empty_predictions"] = sum(not p for p in preds)
    RES_DIR.mkdir(exist_ok=True, parents=True)
    pd.DataFrame({"noisy": [r["noisy"] for r in val], "clean": refs, "pred": preds}).to_csv(
        RES_DIR / f"preds_{tag}.csv", index=False, encoding="utf-8")
    (RES_DIR / f"metrics_{tag}.json").write_text(json.dumps(res, indent=2))
    print(tag, res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mbart", "mt5", "byt5"])
    ap.add_argument("--size", default="base")
    ap.add_argument("--base-path")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--baseline", action="store_true", help="score the copy-the-noisy-input baseline and exit")
    a = ap.parse_args()
    if a.baseline:
        return baseline(a.max_val)
    if not a.model:
        ap.error("--model is required unless --baseline is given")
    evaluate(a.model, a.size, a.beams, a.batch, a.max_val, a.base_path, a.tag_suffix)


if __name__ == "__main__":
    main()
