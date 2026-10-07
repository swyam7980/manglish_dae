"""Phase E: tiny end-to-end check per backbone (train a few steps, save, reload, reconstruct, verify)."""
import argparse
import json
import sys

import train
from config import OUT_DIR, RES_DIR
from evaluate import evaluate
from models import run_tag

SUFFIX = "_smoke"


def check(key, size, base_path, rows, steps, val_rows):
    info = train.main(["--model", key, "--size", size, "--max-train", str(rows), "--max-steps", str(steps),
                       "--tag-suffix", SUFFIX] + (["--base-path", base_path] if base_path else []))
    res = evaluate(key, size, max_val=val_rows, base_path=base_path, tag_suffix=SUFFIX)
    problems = []
    if info["trainable_pct"] >= 5:
        problems.append(f"trainable params {info['trainable_pct']}% (expected well under 5%)")
    if not info["last_logged_loss"] < 0.9 * info["first_logged_loss"]:
        problems.append(f"loss fell by less than 10% ({info['first_logged_loss']:.3f} -> {info['last_logged_loss']:.3f})")
    if res["CER"] > 100:
        problems.append(f"smoke CER {res['CER']}% is above 100% (output is garbage or never stops; a longer run may fix "
                        "the latter, run diagnose.py to check)")
    if res["empty_predictions"] == val_rows:
        problems.append("every prediction is empty")
    return problems, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mbart", "mt5", "byt5"])
    ap.add_argument("--size", default="base")
    ap.add_argument("--base-path", help="only valid with a single --models entry")
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--val-rows", type=int, default=50)
    a = ap.parse_args()
    failed = False
    for key in a.models:
        print(f"\n=== smoke test: {key}")
        problems, info = check(key, a.size, a.base_path, a.rows, a.steps, a.val_rows)
        print(f"{key}: trainable={info['trainable_pct']}% loss {info['first_logged_loss']:.3f}->{info['last_logged_loss']:.3f} "
              f"peak_gpu_gb={info['peak_gpu_gb']}")
        for p in problems:
            print("  PROBLEM:", p)
        failed |= bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
