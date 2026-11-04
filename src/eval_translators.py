"""Which translator should the second stage be? Score each on IN22 from the real Malayalam (its ceiling) and from the
Malayalam script our first-stage models wrote from Manglish (saved by eval_en.py, so those models need not rerun).
All systems use the same decoding (beam 5), and results merge across runs into results/ours_en/translators.{json,md}.

  PYTHONPATH=.deps_mt python src/eval_translators.py --systems it2_dist opus nllb [--max-eval 50]
"""
import argparse
import json

import numpy as np
import pandas as pd
import sacrebleu
import torch

from config import RES_DIR
from eval_en import boot, en_scores, sentence_chrf
from translators import LICENCES, SYSTEMS, Translator

SETS = ("in22_conv", "in22_gen")
INPUTS = {"gold": "ml", "tiny_typed": "script_ours_tiny_en__roman", "tiny_noisy": "script_ours_tiny_en__noisy",
          "tiny_mixed": "script_ours_tiny_en__mixed", "byt5_typed": "script_ours_byt5_en__roman",
          "byt5_noisy": "script_ours_byt5_en__noisy"}
HEAD = {"gold": "real Malayalam", "tiny_typed": "Tiny, typed", "tiny_noisy": "Tiny, noisy", "tiny_mixed": "Tiny, mixed",
        "byt5_typed": "ByT5, typed", "byt5_noisy": "ByT5, noisy"}


def write_markdown(s, path):
    lines = ["# Second-stage translators (BLEU / chrF, beam 5)", ""]
    for st in SETS:
        names = [n for n in SYSTEMS if n in s.get(st, {})]
        if not names:
            continue
        lines += [f"## {st}", "", "| system | licence | " + " | ".join(HEAD.values()) + " |", "|" + "---|" * (len(HEAD) + 2)]
        for n in names:
            v = s[st][n]
            lines.append(f"| {n} | {LICENCES[n]} | " + " | ".join(
                f"{v[i]['BLEU']:.1f} / {v[i]['chrF']:.1f}" if i in v else "-" for i in INPUTS) + " |")
        lines.append("")
    if s.get("vs"):
        lines += ["Paired bootstrap on sentence chrF (first minus second, positive favours the first):", ""]
        lines += [f"- {v['set']}/{v['input']}: {v['a']} - {v['b']}: {v['mean']:+.2f} (95% CI {v['lo']:+.2f} to {v['hi']:+.2f})"
                  for v in s["vs"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", choices=list(SYSTEMS), required=True)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-eval", type=int, default=0, help="cap sentences per set (smoke tests; nothing is saved)")
    ap.add_argument("--device", choices=["cuda", "cpu"])
    a = ap.parse_args(argv)
    out = RES_DIR / "ours_en"
    path = out / "translators.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    frames = {s: pd.read_csv(out / f"preds_en_{s}.csv") for s in SETS}
    if a.max_eval:
        frames = {s: f.head(a.max_eval) for s, f in frames.items()}
    for name in a.systems:
        tr = Translator(name, device=a.device, beams=a.beams)
        for st, f in frames.items():
            refs = f["en"].astype(str).tolist()
            res = {}
            for key, col in INPUTS.items():
                p = tr.translate(f[col].fillna("").astype(str).tolist(), a.batch)
                res[key] = en_scores(p, refs)
                print(f"[{name}] {st}/{key}: {res[key]}", flush=True)
                f[f"mt_{name}__{key}"] = p
                if a.max_eval and key == "gold":
                    for i in range(min(4, len(p))):
                        print("   ", refs[i], "=>", p[i], flush=True)
            if not a.max_eval:
                summary.setdefault(st, {})[name] = res
                saved = out / f"preds_mt_{st}.csv"
                old = pd.read_csv(saved) if saved.exists() else f[["ml", "en"]]
                for key in INPUTS:
                    old[f"{name}__{key}"] = f[f"mt_{name}__{key}"].values
                old.to_csv(saved, index=False)
                path.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
        del tr
        torch.cuda.empty_cache()
    if a.max_eval:
        return summary
    rng = np.random.default_rng(0)
    summary["vs"] = []
    for st in SETS:
        saved = out / f"preds_mt_{st}.csv"
        if not saved.exists():
            continue
        f = pd.read_csv(saved)
        refs = f["en"].astype(str).tolist()
        names = [n for n in SYSTEMS if f"{n}__gold" in f]
        chrf = {(n, k): sentence_chrf(f[f"{n}__{k}"].fillna("").astype(str).tolist(), refs) for n in names for k in ("gold", "tiny_typed")}
        best = max(names, key=lambda n: summary[st][n]["tiny_typed"]["BLEU"])
        for n in names:
            if n != best:
                for k in ("gold", "tiny_typed"):
                    m, lo, hi = boot(chrf[(best, k)], chrf[(n, k)], rng)
                    summary["vs"].append({"set": st, "input": k, "a": best, "b": n, "mean": m, "lo": lo, "hi": hi})
    path.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, out / "translators.md")
    return summary


if __name__ == "__main__":
    main()
