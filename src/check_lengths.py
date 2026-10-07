"""Verify nothing is truncated at MAX_LEN for any backbone's tokenizer (important for ByT5)."""
import json

from config import PROC_DIR, MAX_LEN
from models import load_tokenizer, input_prefix

rows = [json.loads(line) for line in open(PROC_DIR / "train.jsonl", encoding="utf-8")]
for key, size in (("mbart", "large"), ("mt5", "base"), ("byt5", "base")):
    tok = load_tokenizer(key, size)
    pre = input_prefix(key)
    lens = [len(tok(pre + r["noisy"])["input_ids"]) for r in rows[:3000]]
    over = sum(n > MAX_LEN for n in lens)
    print(f"{key:6s} max={max(lens):4d} mean={sum(lens) / len(lens):6.1f} over{MAX_LEN}={over} ({100 * over / len(lens):.2f}%)")
