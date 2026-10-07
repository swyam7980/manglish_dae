"""Error analysis on saved predictions: worst examples, over-correction, unresolved repeats.
Writes results/analysis_<tag>.md for each preds_*.csv."""
import re
from itertools import groupby

import pandas as pd

from config import RES_DIR
from metrics import sentence_cer

N_WORST = 15
N_RANDOM = 30


def has_long_run(word, k=3):
    return any(len(list(g)) >= k for _, g in groupby(word))


def analyse(df):
    """Word-level patterns when noisy/clean/pred have equal word counts (the common case)."""
    over = tot_ok = unresolved = tot_rep = 0
    for noisy, clean, pred in zip(df.noisy, df.clean, df.pred.fillna("")):
        nw, cw, pw = noisy.split(), clean.split(), pred.split()
        if not (len(nw) == len(cw) == len(pw)):
            continue
        for n, c, p in zip(nw, cw, pw):
            if n == c:                      # word was not corrupted
                tot_ok += 1
                over += p != c              # ...but the model changed it: over-correction
            elif has_long_run(n) and not has_long_run(c):
                tot_rep += 1
                unresolved += has_long_run(p)  # elongation left in place
    word_counts_differ = sum(len(a.split()) != len(c.split()) for a, c in zip(df.pred.fillna(""), df.clean))
    return {
        "over_correction_rate_pct": round(100 * over / max(tot_ok, 1), 2),
        "unresolved_elongation_pct": round(100 * unresolved / max(tot_rep, 1), 2),
        "sentences_with_split_or_merged_words": int(word_counts_differ),
        "empty_predictions": int((df.pred.fillna("") == "").sum()),
    }


def main():
    files = sorted(RES_DIR.glob("preds_*.csv"))
    if not files:
        raise SystemExit("No preds_*.csv found. Run evaluate.py first.")
    for p in files:
        tag = p.stem.replace("preds_", "")
        df = pd.read_csv(p, keep_default_na=False)
        df["cer"] = [sentence_cer(a, b) for a, b in zip(df.pred, df.clean)]
        lines = [f"# Error analysis: {tag}", "", "## Patterns", ""]
        lines += [f"- {k}: {v}" for k, v in analyse(df).items()]
        lines += ["", f"## {N_WORST} worst sentences by CER", ""]
        for _, r in df.sort_values("cer", ascending=False).head(N_WORST).iterrows():
            lines += [f"- CER {r.cer:.1f}", f"  - noisy: `{r.noisy}`", f"  - clean: `{r.clean}`", f"  - pred : `{r.pred}`"]
        lines += ["", f"## {N_RANDOM} random sentences (seed 0) for hand inspection", ""]
        for _, r in df.sample(min(N_RANDOM, len(df)), random_state=0).iterrows():
            lines += [f"- noisy: `{r.noisy}`", f"  - clean: `{r.clean}`", f"  - pred : `{r.pred}`"]
        out = RES_DIR / f"analysis_{tag}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main()
