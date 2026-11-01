"""Test our models against the comparison models on exactly the inputs those were scored on.

  1. combined  test_all.jsonl (2,059 sentences, three sources), clean task. Paired bootstrap on CER against the stored
               seed-42 predictions of every combined-data model (results/combined_all/preds_*.csv) and the rule baseline.
  2. robustness the 600-sentence subsample and ROB_SEED noise of robustness.py, every noise condition. Our models
               trained on all of these noise types; the comparison models saw only the base noise.
  3. script    ours_ml_test.jsonl (658 held-out Dakshina sentences): Manglish -> Malayalam script from the clean and
               from the noisy Manglish, against a rule-based transliterator (indic_transliteration, ITRANS-style rules).
Writes results/ours/: preds_*.csv, ours_summary.json, ours_summary.md.
"""
import argparse
import json

import numpy as np
import pandas as pd

from config import RES_DIR
from evaluate import load_eval
from extra_data import read_jsonl
from config import PROC_DIR
from generation import collapse_runs, length_guard
from metrics import score
from noise import CONDITIONS
from robustness import condition_inputs
from seeds import bootstrap_diff, COMBINED_BOOT
from translate import Translator

LABELS = {"ours_byt5": "Manglish-ByT5 (ours)", "ours_tiny": "Manglish-Tiny (ours)"}
KINDS = {"ours_byt5": "byt5", "ours_tiny": "tiny"}
COMPARE_WITH = ["byt5_plusall", "mbart_plusall", "mt5_lr1e-3_plusall", "byt5", "mbart"]
ROB_SAMPLE = 600


def rule_transliterate(texts):
    """Rule-based Manglish -> Malayalam baseline. Casual Manglish is not a formal scheme, so this is a floor."""
    from indic_transliteration import sanscript
    return [sanscript.transliterate(t, sanscript.ITRANS, sanscript.MALAYALAM) for t in texts]


def by_source(df):
    out = {"ALL": score(df.pred.tolist(), df.clean.tolist())}
    for src, d in df.groupby("source"):
        out[src] = score(d.pred.tolist(), d.clean.tolist())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=list(LABELS))
    ap.add_argument("--parts", nargs="+", default=["combined", "robustness", "script"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-eval", type=int, default=0, help="cap sentences per part (smoke tests)")
    ap.add_argument("--out", default="ours")
    a = ap.parse_args(argv)
    out = RES_DIR / a.out
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "ours_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    rng = np.random.default_rng(0)

    test = load_eval("test_all.jsonl", a.max_eval)
    rob = load_eval("test_all.jsonl", min(ROB_SAMPLE, a.max_eval) if a.max_eval else ROB_SAMPLE)
    rob_inputs = {c: condition_inputs(rob, c) for c in CONDITIONS}
    ml = read_jsonl(PROC_DIR / "ours_ml_test.jsonl")[: a.max_eval or None]

    if "script" in a.parts:
        refs = [r["native"] for r in ml]
        sc = summary.setdefault("script", {})
        for inp in ("clean", "noisy"):
            sc.setdefault("rule_transliteration", {})[inp] = score(rule_transliterate([r[inp] for r in ml]), refs)

    for name in a.systems:
        tr = Translator(KINDS[name], guard=False)
        if "combined" in a.parts:
            raw = tr.translate([r["noisy"] for r in test], "clean", a.batch)
            guarded, replaced = length_guard(raw, [r["noisy"] for r in test])
            df = pd.DataFrame({"source": [r["source"] for r in test], "noisy": [r["noisy"] for r in test],
                               "clean": [r["clean"] for r in test], "pred": guarded, "pred_raw": raw})
            df.to_csv(out / f"preds_{name}.csv", index=False)
            entry = {"label": LABELS[name], "n": len(df), "guard_replaced": replaced, "scores": by_source(df),
                     "raw_scores": score(raw, df.clean.tolist()), "vs": []}
            for other in COMPARE_WITH + ["rule_collapse_runs"]:
                if other == "rule_collapse_runs":
                    opred = [collapse_runs(n) for n in df.noisy]
                else:
                    p = RES_DIR / "combined_all" / f"preds_{other}.csv"
                    if not p.exists() or a.max_eval:
                        continue
                    o = pd.read_csv(p).fillna("")
                    assert o.clean.tolist() == df.clean.tolist(), f"{p} rows are not aligned with test_all.jsonl"
                    opred = o.pred.tolist()
                mean, lo, hi = bootstrap_diff(df.pred.tolist(), opred, df.clean.tolist(), rng, COMBINED_BOOT)
                entry["vs"].append({"other": other, "mean": mean, "lo": lo, "hi": hi})
                print(f"[{name}] CER difference vs {other}: {mean:+.2f} ({lo:+.2f} to {hi:+.2f})", flush=True)
            summary.setdefault("combined", {})[name] = entry
            print(name, "combined", entry["scores"]["ALL"], "guard replaced", replaced, flush=True)
        if "robustness" in a.parts:
            refs = [r["clean"] for r in rob]
            res = {}
            for c, noisy in rob_inputs.items():
                preds, _ = length_guard(tr.translate(noisy, "clean", a.batch), noisy)
                res[c] = score(preds, refs)
                print(name, c, res[c], flush=True)
            summary.setdefault("robustness", {})[name] = res
        if "script" in a.parts:
            refs = [r["native"] for r in ml]
            res = {}
            for inp in ("clean", "noisy"):
                preds = tr.translate([r[inp] for r in ml], "ml", a.batch)
                res[inp] = score(preds, refs)
                pd.DataFrame({"roman": [r[inp] for r in ml], "native": refs, "pred": preds}).to_csv(
                    out / f"preds_ml_{inp}_{name}.csv", index=False)
                print(name, "script from", inp, res[inp], flush=True)
            summary["script"][name] = res
        del tr

    summary_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, out / "ours_summary.md")
    return summary


def write_markdown(s, path):
    lines = ["# Our models", ""]
    if "combined" in s:
        lines += ["## Combined test, clean task (CER %, lower is better)", "", "| system | ALL | " +
                  " | ".join(k for k in next(iter(s["combined"].values()))["scores"] if k != "ALL") + " |", ""]
        lines[-1] = "|" + "---|" * (lines[-2].count("|") - 1)
        for name, e in s["combined"].items():
            lines.append(f"| {e['label']} | " + " | ".join(f"{v['CER']:.2f}" for v in e["scores"].values()) + " |")
        lines += ["", "Paired bootstrap, CER(ours) - CER(other), negative favours ours:", ""]
        for name, e in s["combined"].items():
            for v in e["vs"]:
                lines.append(f"- {name} - {v['other']}: {v['mean']:+.2f} (95% CI {v['lo']:+.2f} to {v['hi']:+.2f})")
    if "robustness" in s:
        conds = list(next(iter(s["robustness"].values())))
        lines += ["", "## Robustness (CER %)", "", "| system | " + " | ".join(conds) + " |", "|" + "---|" * (len(conds) + 1)]
        for name, r in s["robustness"].items():
            lines.append(f"| {name} | " + " | ".join(f"{r[c]['CER']:.2f}" for c in conds) + " |")
    if "script" in s:
        lines += ["", "## Manglish -> Malayalam script (CER / chrF)", "", "| system | from clean | from noisy |", "|---|---|---|"]
        for name, r in s["script"].items():
            lines.append(f"| {name} | {r['clean']['CER']:.2f} / {r['clean']['chrF']:.1f} | "
                         f"{r['noisy']['CER']:.2f} / {r['noisy']['chrF']:.1f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
