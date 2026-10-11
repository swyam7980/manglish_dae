"""Reconstruct the validation set from a saved adapter and score it. Also scores the no-model baseline."""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch

from config import PROC_DIR, RES_DIR
from generation import load_adapter, generate, release
from metrics import score
from models import run_tag


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_val(max_val=0):
    p = PROC_DIR / "val.jsonl"
    if not p.exists():
        sys.exit(f"{p} missing. Run data_prep.py first.")
    val = read_jsonl(p)
    return val[:max_val] if max_val else val


def load_eval(name="val.jsonl", max_eval=0, seed=0):
    """Rows of an evaluation file in data/processed. max_eval draws a fixed random subsample (same rows every call),
    keeping the file's mix of sources; use it for the slower studies, never for the headline numbers."""
    p = Path(name) if Path(name).is_absolute() else PROC_DIR / name
    if not p.exists():
        sys.exit(f"{p} missing. Run data_prep.py (val.jsonl) or extra_data.py (test_<tag>.jsonl) first.")
    rows = read_jsonl(p)
    if max_eval and max_eval < len(rows):
        keep = sorted(random.Random(seed).sample(range(len(rows)), max_eval))
        rows = [rows[i] for i in keep]
    return rows


def baseline(max_val=0):
    val = load_val(max_val)
    res = score([r["noisy"] for r in val], [r["clean"] for r in val])
    RES_DIR.mkdir(exist_ok=True, parents=True)
    (RES_DIR / "metrics_identity.json").write_text(json.dumps(res, indent=2))
    print("identity (copy noisy input):", res)
    return res


def evaluate(model_key, size="base", beams=1, batch=16, max_val=0, base_path=None, tag_suffix=""):
    tag = run_tag(model_key, size) + tag_suffix
    val = load_val(max_val)
    loaded = load_adapter(model_key, size, tag_suffix, base_path)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    preds = generate(loaded, [r["noisy"] for r in val], beams, batch)
    refs = [r["clean"] for r in val]
    res = score(preds, refs)
    res["empty_predictions"] = sum(not p for p in preds)
    RES_DIR.mkdir(exist_ok=True, parents=True)
    pd.DataFrame({"noisy": [r["noisy"] for r in val], "clean": refs, "pred": preds}).to_csv(
        RES_DIR / f"preds_{tag}.csv", index=False, encoding="utf-8")
    (RES_DIR / f"metrics_{tag}.json").write_text(json.dumps(res, indent=2))
    print(tag, res)
    release(loaded)
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
