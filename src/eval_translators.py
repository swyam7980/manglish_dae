"""Which translator should the second stage be? Score each on IN22 from the real Malayalam (its ceiling) and from the
Malayalam script our first-stage models wrote from Manglish (saved by eval_en.py, so those models need not rerun).
All systems use the same decoding (beam 5), and results merge across runs into results/ours_en/translators.{json,md}.

  PYTHONPATH=.deps_mt python src/eval_translators.py --systems it2_dist opus nllb [--max-eval 50]
  python src/eval_translators.py --systems it2_1b --first ours_tiny_script    # a new first-stage model, same translator
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

import deps_mt  # noqa: F401  (must come before transformers)
from config import OUT_DIR, RES_DIR
from eval_en import boot, en_scores, sentence_chrf
from translators import BEST, LICENCES, SYSTEMS, Translator

SETS = ("in22_conv", "in22_gen")
INPUTS = {"gold": "ml", "tiny_typed": "script_ours_tiny_en__roman", "tiny_noisy": "script_ours_tiny_en__noisy",
          "tiny_mixed": "script_ours_tiny_en__mixed", "byt5_typed": "script_ours_byt5_en__roman",
          "byt5_noisy": "script_ours_byt5_en__noisy"}
HEAD = {"gold": "real Malayalam", "tiny_typed": "Tiny, typed", "tiny_noisy": "Tiny, noisy", "tiny_mixed": "Tiny, mixed",
        "byt5_typed": "ByT5, typed", "byt5_noisy": "ByT5, noisy"}
FIRST_INPUTS = (("typed", "in_roman"), ("noisy", "in_noisy"), ("mixed", "in_mixed"))


def short(tag):
    return tag.replace("ours_", "")


def write_markdown(s, path):
    lines = ["# Second-stage translators (BLEU / chrF, beam 5)", ""]
    for st in SETS:
        names = [n for n in SYSTEMS if n in s.get(st, {})]
        if not names:
            continue
        keys = list(INPUTS) + sorted({k for n in names for k in s[st][n]} - set(INPUTS))
        lines += [f"## {st}", "", "| system | licence | " + " | ".join(HEAD.get(k, k) for k in keys) + " |",
                  "|" + "---|" * (len(keys) + 2)]
        for n in names:
            v = s[st][n]
            lines.append(f"| {n} | {LICENCES[n]} | " + " | ".join(
                f"{v[i]['BLEU']:.1f} / {v[i]['chrF']:.1f}" if i in v else "-" for i in keys) + " |")
        lines.append("")
    if s.get("vs"):
        lines += ["Paired bootstrap on sentence chrF (first minus second, positive favours the first):", ""]
        lines += [f"- {v['set']}/{v['input']}: {v['a']} - {v['b']}: {v['mean']:+.2f} (95% CI {v['lo']:+.2f} to {v['hi']:+.2f})"
                  for v in s["vs"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def first_stage_script(tag, frames, batch):
    """Malayalam script from a first-stage model for the IN22 Manglish inputs; adds the columns to frames."""
    from translate import Translator as FirstStage
    folder = OUT_DIR / tag
    kind = "tiny" if (folder / "model.pt").exists() else "byt5"
    tr = FirstStage(kind, folder if kind == "tiny" else folder / "model", guard=False)
    keys = {}
    for st, f in frames.items():
        for name, col in FIRST_INPUTS:
            f[f"script_{tag}__{name}"] = tr.translate(f[col].fillna("").astype(str).tolist(), "ml", batch)
            keys[f"{short(tag)}_{name}"] = f"script_{tag}__{name}"
    del tr
    torch.cuda.empty_cache()
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", choices=list(SYSTEMS), required=True)
    ap.add_argument("--first", help="output tag of a first-stage model: score its script instead of the saved ones")
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
        frames = {s: f.head(a.max_eval).copy() for s, f in frames.items()}
    inputs = {"gold": "ml", **first_stage_script(a.first, frames, a.batch)} if a.first else INPUTS
    for name in a.systems:
        tr = Translator(name, device=a.device, beams=a.beams)
        for st, f in frames.items():
            refs = f["en"].astype(str).tolist()
            res = {}
            for key, col in inputs.items():
                if key == "gold" and a.first and name in summary.get(st, {}):
                    continue  # the real-Malayalam score does not depend on the first stage
                p = tr.translate(f[col].fillna("").astype(str).tolist(), a.batch)
                res[key] = en_scores(p, refs)
                print(f"[{name}] {st}/{key}: {res[key]}", flush=True)
                f[f"mt_{name}__{key}"] = p
                if a.max_eval and key == "gold":
                    for i in range(min(4, len(p))):
                        print("   ", refs[i], "=>", p[i], flush=True)
            if not a.max_eval:
                summary.setdefault(st, {}).setdefault(name, {}).update(res)
                saved = out / f"preds_mt_{st}.csv"
                old = pd.read_csv(saved) if saved.exists() else f[["ml", "en"]].copy()
                for key in res:
                    old[f"{name}__{key}"] = f[f"mt_{name}__{key}"].values
                for key, col in inputs.items():
                    if col.startswith(f"script_{a.first}"):
                        old[col] = f[col].values
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
        chrf = lambda col: sentence_chrf(f[col].fillna("").astype(str).tolist(), refs)
        names = [n for n in SYSTEMS if f"{n}__gold" in f]
        best = max(names, key=lambda n: summary[st][n]["tiny_typed"]["BLEU"])
        for n in names:
            if n != best:
                for k in ("gold", "tiny_typed"):
                    m, lo, hi = boot(chrf(f"{best}__{k}"), chrf(f"{n}__{k}"), rng)
                    summary["vs"].append({"set": st, "input": k, "a": best, "b": n, "mean": m, "lo": lo, "hi": hi})
        # newer first-stage models against the one the translators were chosen with, through the chosen translator
        for k in sorted(c[len(BEST) + 2:] for c in f.columns if c.startswith(f"{BEST}__") and c.endswith("_typed")):
            if k != "tiny_typed" and f"{BEST}__tiny_typed" in f:
                m, lo, hi = boot(chrf(f"{BEST}__{k}"), chrf(f"{BEST}__tiny_typed"), rng)
                summary["vs"].append({"set": st, "input": "typed", "a": f"{BEST} from {k[:-6]}",
                                      "b": f"{BEST} from tiny_en", "mean": m, "lo": lo, "hi": hi})
    path.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, out / "translators.md")
    return summary


if __name__ == "__main__":
    main()
