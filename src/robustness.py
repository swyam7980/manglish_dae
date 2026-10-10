"""Robustness test: score every trained seed on noise it was not trained on, without retraining.

Conditions (noise.CONDITIONS): the stored validation noise ('base'), heavier repetition/duplication/deletion, vowel
dropping and phonetic spelling variants. Each condition corrupts the same 325 clean validation sentences, so the
copy-the-input baseline and a rule-based cleaner are scored on identical inputs. Writes results/robustness/.
"""
import argparse
import json

import pandas as pd

from config import RES_DIR
from evaluate import load_val
from generation import load_adapter, generate, release, seed_suffixes, parse_systems, collapse_runs
from metrics import score
from noise import CONDITIONS, apply_condition

ROB_SEED = 11
DEFAULT_SYSTEMS = ["mbart", "byt5", "mt5:_lr1e-3"]
METRICS = ["CER", "WER", "BLEU", "chrF"]


def condition_inputs(val, name):
    clean = [r["clean"] for r in val]
    return [r["noisy"] for r in val] if name == "base" else apply_condition(clean, name, ROB_SEED)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS)
    ap.add_argument("--size", default="base")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--out", default="robustness")
    a = ap.parse_args(argv)

    val = load_val(a.max_val)
    refs = [r["clean"] for r in val]
    inputs = {c: condition_inputs(val, c) for c in CONDITIONS}
    out_dir = RES_DIR / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c, noisy in inputs.items():
        rows.append({"system": "copy_input", "seed": 0, "condition": c, **score(noisy, refs)})
        rows.append({"system": "rule_collapse_runs", "seed": 0, "condition": c, **score([collapse_runs(n) for n in noisy], refs)})
    for key, variant in parse_systems(a.systems):
        for seed, suffix in seed_suffixes(key, a.size, variant):
            loaded = load_adapter(key, a.size, suffix)
            for c, noisy in inputs.items():
                rows.append({"system": f"{key}{variant}", "seed": seed, "condition": c,
                             **score(generate(loaded, noisy, 1, a.batch), refs)})
                print(rows[-1], flush=True)
            release(loaded)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "robustness_detail.csv", index=False)
    g = df.groupby(["system", "condition"])
    summary = g[METRICS].agg(["mean", "std"]).round(2)
    summary["seeds"] = g.size()
    summary.to_csv(out_dir / "robustness_summary.csv")
    (out_dir / "robustness.json").write_text(json.dumps(
        [{"system": s, "condition": c, "seeds": int(len(d)),
          **{m: {"mean": float(d[m].mean()), "sd": float(d[m].std(ddof=1)) if len(d) > 1 else None} for m in METRICS}}
         for (s, c), d in g], indent=1))
    pivot = df.groupby(["system", "condition"])["CER"].mean().unstack("condition").round(2)
    print(pivot.to_string())
    return df


if __name__ == "__main__":
    main()
