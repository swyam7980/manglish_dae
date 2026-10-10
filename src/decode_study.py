"""Decoding study: does beam search or a length guard fix outputs that drop words (seen mostly with ByT5)?

For every trained seed of each system, scores greedy and 4-beam decoding, each with and without the length guard,
plus a rule-based baseline that only collapses letter runs. Writes results/decoding/.
"""
import argparse
import json

import pandas as pd

from config import RES_DIR
from evaluate import load_val
from generation import (load_adapter, generate, release, seed_suffixes, parse_systems, length_guard, collapse_runs,
                        GUARD_MIN_RATIO)
from metrics import score

DEFAULT_SYSTEMS = ["mbart", "byt5", "mt5:_lr1e-3"]
METRICS = ["CER", "WER", "BLEU", "chrF"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS, help="key or key:variant, e.g. mt5:_lr1e-3")
    ap.add_argument("--size", default="base")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--min-ratio", type=float, default=GUARD_MIN_RATIO)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--out", default="decoding")
    a = ap.parse_args(argv)

    val = load_val(a.max_val)
    noisy, refs = [r["noisy"] for r in val], [r["clean"] for r in val]
    out_dir = RES_DIR / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"system": "rule_collapse_runs", "seed": 0, "config": "rule", "guard_replaced": 0,
             **score([collapse_runs(n) for n in noisy], refs)}]
    for key, variant in parse_systems(a.systems):
        for seed, suffix in seed_suffixes(key, a.size, variant):
            loaded = load_adapter(key, a.size, suffix)
            for beams in (1, a.beams):
                preds = generate(loaded, noisy, beams, a.batch)
                guarded, replaced = length_guard(preds, noisy, a.min_ratio)
                for name, p, rep in ((f"beam{beams}", preds, 0), (f"beam{beams}+guard", guarded, replaced)):
                    rows.append({"system": f"{key}{variant}", "seed": seed, "config": name, "guard_replaced": rep,
                                 **score(p, refs)})
                    print(rows[-1], flush=True)
                if beams == 1:
                    pd.DataFrame({"noisy": noisy, "clean": refs, "pred": preds, "guarded": guarded}).to_csv(
                        out_dir / f"preds_{key}{suffix}_greedy.csv", index=False, encoding="utf-8")
            release(loaded)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "decoding_detail.csv", index=False)
    g = df.groupby(["system", "config"])
    summary = g[METRICS].agg(["mean", "std"]).round(2)
    summary["seeds"] = g.size()
    summary["guard_replaced_mean"] = g["guard_replaced"].mean().round(1)
    summary.to_csv(out_dir / "decoding_summary.csv")
    (out_dir / "decoding_summary.json").write_text(json.dumps(
        [{"system": s, "config": c, "seeds": int(len(d)), "guard_replaced": float(d.guard_replaced.mean()),
          **{m: {"mean": float(d[m].mean()), "sd": float(d[m].std(ddof=1)) if len(d) > 1 else None} for m in METRICS}}
         for (s, c), d in g], indent=1))
    print(summary.to_string())
    return df


if __name__ == "__main__":
    main()
