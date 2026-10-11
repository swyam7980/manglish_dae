"""Score every trained seed of every system on the combined multi-source test set, overall and per source.

The test file (data/processed/test_<tag>.jsonl, built by extra_data.py) holds held-out sentences from each dataset;
its rows carry a `source` field. Models trained only on the Manglish class are scored on the other sources too, which
measures how far they transfer. Writes results/<out>/ with per-seed detail, a mean/SD summary and seed-42 predictions.
"""
import argparse
import json

import pandas as pd

from config import RES_DIR
from evaluate import load_eval
from generation import load_adapter, generate, release, seed_suffixes, parse_systems, collapse_runs
from metrics import score

METRICS = ["CER", "WER", "BLEU", "chrF"]
DEFAULT_SYSTEMS = ["mbart", "byt5", "mt5:_lr1e-3"]


def score_by_source(preds, refs, sources):
    """Metrics over all rows ('ALL') and over each source separately. Corpus-level, so long sentences weigh more."""
    out = {"ALL": score(preds, refs)}
    for src in sorted(set(sources)):
        idx = [i for i, s in enumerate(sources) if s == src]
        out[src] = score([preds[i] for i in idx], [refs[i] for i in idx])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", required=True, help="test_<tag>.jsonl in data/processed")
    ap.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS, help="key or key:variant, e.g. mt5:_lr1e-3")
    ap.add_argument("--size", default="base")
    ap.add_argument("--base-path", help="local model directory instead of the Hub id (offline tests)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--out", default="combined")
    a = ap.parse_args(argv)

    rows = load_eval(a.eval_file, a.max_eval)
    noisy, refs = [r["noisy"] for r in rows], [r["clean"] for r in rows]
    sources = [r.get("source", "unknown") for r in rows]
    out_dir = RES_DIR / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    detail = []

    def add(system, seed, preds):
        for src, m in score_by_source(preds, refs, sources).items():
            detail.append({"system": system, "seed": seed, "source": src, "n": len(refs) if src == "ALL" else sources.count(src), **m})

    add("copy_input", 0, noisy)
    add("rule_collapse_runs", 0, [collapse_runs(n) for n in noisy])
    for key, variant in parse_systems(a.systems):
        for seed, suffix in seed_suffixes(key, a.size, variant):
            loaded = load_adapter(key, a.size, suffix, a.base_path)
            preds = generate(loaded, noisy, 1, a.batch)
            release(loaded)
            add(f"{key}{variant}", seed, preds)
            print(f"{key}{suffix}", {d["source"]: d["CER"] for d in detail[-len(set(sources)) - 1:]}, flush=True)
            if seed == 42:
                pd.DataFrame({"source": sources, "noisy": noisy, "clean": refs, "pred": preds}).to_csv(
                    out_dir / f"preds_{key}{suffix}.csv", index=False, encoding="utf-8")
    df = pd.DataFrame(detail)
    df.to_csv(out_dir / "combined_detail.csv", index=False)
    g = df.groupby(["system", "source"])
    summary = [{"system": s, "source": src, "seeds": int(len(d)), "n": int(d.n.iloc[0]),
                **{m: {"mean": float(d[m].mean()), "sd": float(d[m].std(ddof=1)) if len(d) > 1 else None} for m in METRICS}}
               for (s, src), d in g]
    (out_dir / "combined_summary.json").write_text(json.dumps(summary, indent=1))
    print(df.groupby(["system", "source"])["CER"].mean().unstack("source").round(2).to_string())
    return df


if __name__ == "__main__":
    main()
