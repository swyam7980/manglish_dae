"""Verify nothing is truncated at MAX_LEN for any backbone's tokenizer (important for ByT5)."""
import argparse
import json

from config import PROC_DIR, MAX_LEN
from models import load_tokenizer, input_prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base", help="mt5/byt5 size tier")
    ap.add_argument("--models", nargs="+", default=["mbart", "mt5", "byt5"])
    a = ap.parse_args()
    rows = []
    for split in ("train", "val"):
        rows += [json.loads(line) for line in open(PROC_DIR / f"{split}.jsonl", encoding="utf-8")]
    worst = 0
    for key in a.models:
        tok = load_tokenizer(key, a.size)
        pre = input_prefix(key)
        for field in ("noisy", "clean"):
            lens = [len(tok(pre + r[field])["input_ids"]) for r in rows]
            over = sum(n > MAX_LEN for n in lens)
            worst = max(worst, 100 * over / len(lens))
            print(f"{key:6s} {field:5s} max={max(lens):4d} mean={sum(lens) / len(lens):6.1f} "
                  f"over{MAX_LEN}={over} ({100 * over / len(lens):.2f}%)")
    if worst > 0.5:
        print("WARNING: >0.5% of rows would be truncated. Lower MAX_CHARS in config.py and re-run data_prep.py.")


if __name__ == "__main__":
    main()
