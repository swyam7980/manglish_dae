"""Score the English task and the code-mixed script task on IN22 (never trained on; see en_data.py).

  en     Manglish -> English on IN22-Conv (1,503 sentences) and IN22-Gen (1,024), from three inputs: plain romanised
         Malayalam, the same with the base noise, and code-mixed (some words already in English). Systems:
           ours_byt5_en        our model, en: task, directly
           pivot_ours_byt5     the earlier Manglish-ByT5 writes Malayalam script, then NLLB-200 translates it
           pivot_ours_byt5_en  the same pipeline with the new model's ml: task
           nllb_gold           NLLB-200 on the real Malayalam sentence: what the pipeline would reach with perfect script
         Paired bootstrap on mean sentence chrF between our direct model and each pipeline.
  mixed  ml: on the code-mixed inputs -> Malayalam script. Besides CER/chrF against the real sentence, the share of
         English words the output turned back into the right Malayalam word ("tomorrow" -> നാളെ), and the share it left
         in Latin letters.
NLLB-200-distilled-600M (CC BY-NC) is only a baseline here; nothing of it is redistributed. All decoding is greedy.
Writes results/ours_en/: preds_*.csv, en_summary.json, en_summary.md.
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
import sacrebleu
import torch

from config import PROC_DIR, RES_DIR, OUT_DIR
from extra_data import read_jsonl
from memory import run_with_backoff
from metrics import score
from translate import Translator

NLLB = "facebook/nllb-200-distilled-600M"
INPUTS = ("roman", "noisy", "mixed")
SETS = ("in22_conv", "in22_gen")
MODELS = {"ours_byt5": ("byt5", OUT_DIR / "ours_byt5" / "model"),
          "ours_byt5_en": ("byt5", OUT_DIR / "ours_byt5_en" / "model"),
          "ours_tiny": ("tiny", OUT_DIR / "ours_tiny"),
          "ours_tiny_en": ("tiny", OUT_DIR / "ours_tiny_en")}
DIRECT = ("ours_byt5_en", "ours_tiny_en")
PIVOT = ("ours_byt5", "ours_byt5_en", "ours_tiny_en")
LABELS = {"ours_byt5_en": "Manglish-ByT5 + English (ours), direct",
          "pivot_ours_byt5": "Manglish-ByT5 script, then NLLB-200",
          "pivot_ours_byt5_en": "Manglish-ByT5 + English script, then NLLB-200",
          "pivot_ours_tiny_en": "Manglish-Tiny + English script, then NLLB-200",
          "direct_ours_tiny_en": "Manglish-Tiny + English (ours), direct",
          "nllb_gold": "NLLB-200 on the real Malayalam (ceiling)",
          "ours_byt5": "Manglish-ByT5 (before English)", "ours_tiny": "Manglish-Tiny (before English)",
          "ours_tiny_en": "Manglish-Tiny + English (ours)"}
BOOT = 1000


class Nllb:
    def __init__(self, dev):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.dev = dev
        self.tok = AutoTokenizer.from_pretrained(NLLB, src_lang="mal_Mlym")
        dtype = torch.float16 if dev == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(NLLB, dtype=dtype).to(dev).eval()
        self.eng = self.tok.convert_tokens_to_ids("eng_Latn")

    def _chunk(self, chunk):
        x = self.tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.dev)
        with torch.no_grad():
            y = self.model.generate(**x, forced_bos_token_id=self.eng, max_length=256, num_beams=1, do_sample=False)
        return [p.strip() for p in self.tok.batch_decode(y, skip_special_tokens=True)]

    def translate(self, texts, batch=32):
        return run_with_backoff(self._chunk, list(texts), batch)[0]


def sentence_chrf(preds, refs):
    return np.array([sacrebleu.sentence_chrf(p, [r]).score for p, r in zip(preds, refs)])


def boot(a, b, rng, n=BOOT):
    """Mean of a - b with a 95% interval over sentence resamples (paired)."""
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    means = d[idx].mean(axis=1)
    return round(float(d.mean()), 2), round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)


def en_scores(preds, refs):
    return {"BLEU": round(sacrebleu.corpus_bleu(preds, [refs]).score, 2),
            "chrF": round(sacrebleu.corpus_chrf(preds, [refs]).score, 2)}


def swap_rates(preds, rows):
    """Share of English swaps turned into the right Malayalam word, and share left in Latin letters."""
    hit = latin = total = 0
    for p, r in zip(preds, rows):
        words = set(re.findall(r"[ഀ-ൿ]+", p))
        latin_words = set(re.findall(r"[a-z]+", p.lower()))
        for ml, en in r["swaps"]:
            total += 1
            hit += ml in words
            latin += en in latin_words
    return {"swaps": total, "translated_pct": round(100 * hit / max(1, total), 1),
            "left_in_latin_pct": round(100 * latin / max(1, total), 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["en", "mixed"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-eval", type=int, default=0, help="cap sentences per set (smoke tests)")
    ap.add_argument("--out", default="ours_en")
    ap.add_argument("--device", choices=["cuda", "cpu"])
    ap.add_argument("--direct", nargs="+", default=list(DIRECT), help="our models translating to English themselves")
    ap.add_argument("--pivot", nargs="+", default=list(PIVOT), help="our script models put in front of NLLB-200")
    a = ap.parse_args(argv)
    out = RES_DIR / a.out
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "en_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_jsonl(PROC_DIR / "ours_en_test.jsonl")
    by_set = {s: [r for r in rows if r["set"] == s][: a.max_eval or None] for s in SETS}
    rng = np.random.default_rng(0)
    present = {k: v for k, v in MODELS.items() if (v[1] / ("config.json" if v[0] == "byt5" else "model.pt")).exists()}

    if "en" in a.parts:
        # script first (one model in memory at a time), then NLLB over every script output, then our direct model
        script = {}
        for name in a.pivot:
            if name not in present:
                continue
            tr = Translator(*present[name], device=dev, guard=False)
            for s, rs in by_set.items():
                for inp in INPUTS:
                    script[(name, s, inp)] = tr.translate([r[inp] for r in rs], "ml", a.batch)
            del tr
            torch.cuda.empty_cache()
        nllb = Nllb(dev)
        preds = {}
        for (name, s, inp), ml in script.items():
            preds[(f"pivot_{name}", s, inp)] = nllb.translate(ml, a.batch)
        gold = {s: nllb.translate([r["ml"] for r in rs], a.batch) for s, rs in by_set.items()}
        del nllb
        torch.cuda.empty_cache()
        for name in a.direct:
            if name not in present:
                continue
            tr = Translator(*present[name], device=dev, guard=False)
            if "en" in tr.tasks:
                key = name if name == "ours_byt5_en" else f"direct_{name}"  # ours_byt5_en keeps its earlier key
                for s, rs in by_set.items():
                    for inp in INPUTS:
                        preds[(key, s, inp)] = tr.translate([r[inp] for r in rs], "en", a.batch)
            del tr
            torch.cuda.empty_cache()

        res = summary.setdefault("en", {})
        for s, rs in by_set.items():
            refs = [r["en"] for r in rs]
            res.setdefault(s, {})["nllb_gold"] = {"input": "real Malayalam", **en_scores(gold[s], refs)}
            csv = out / f"preds_en_{s}.csv"
            frame = pd.read_csv(csv).to_dict("list") if csv.exists() else {}  # keep columns of systems not rerun
            frame.update({"ml": [r["ml"] for r in rs], "en": refs, "nllb_gold": gold[s]})
            for inp in INPUTS:
                frame[f"in_{inp}"] = [r[inp] for r in rs]
            chrf = {}
            for (name, s2, inp), p in preds.items():
                if s2 != s:
                    continue
                res[s].setdefault(name, {})[inp] = en_scores(p, refs)
                frame[f"{name}__{inp}"] = p
                chrf[(name, inp)] = sentence_chrf(p, refs)
            for (name, s2, inp), ml in script.items():
                if s2 == s:
                    frame[f"script_{name}__{inp}"] = ml
            pd.DataFrame(frame).to_csv(csv, index=False)
            for name in res[s]:  # sentence chrF of systems scored in an earlier run, from their saved predictions
                for inp in INPUTS:
                    if (name, inp) not in chrf and f"{name}__{inp}" in frame:
                        chrf[(name, inp)] = sentence_chrf([str(x) for x in frame[f"{name}__{inp}"]], refs)
            vs = res[s].setdefault("vs", [])
            vs.clear()
            directs = [k for k in ("ours_byt5_en", "direct_ours_tiny_en") if k in res[s]]
            pivots = [k for k in res[s] if k.startswith("pivot_")]
            for inp in INPUTS:
                for d in directs:
                    for other in pivots:
                        if (d, inp) in chrf and (other, inp) in chrf:
                            m, lo, hi = boot(chrf[(d, inp)], chrf[(other, inp)], rng)
                            vs.append({"input": inp, "direct": d, "other": other, "mean": m, "lo": lo, "hi": hi})
                            print(f"[{s}/{inp}] sentence chrF, {d} - {other}: {m:+.2f} ({lo:+.2f} to {hi:+.2f})",
                                  flush=True)
            print(s, json.dumps({k: v for k, v in res[s].items() if k != "vs"}), flush=True)

    if "mixed" in a.parts:
        mixed = [r for rs in by_set.values() for r in rs]
        swapped = [r for r in mixed if r["swaps"]]
        res = summary.setdefault("mixed", {"n": len(mixed), "n_with_swaps": len(swapped)})
        res.update(n=len(mixed), n_with_swaps=len(swapped))
        frame = {"set": [r["set"] for r in swapped], "mixed": [r["mixed"] for r in swapped],
                 "ml": [r["ml"] for r in swapped], "swaps": [json.dumps(r["swaps"], ensure_ascii=False) for r in swapped]}
        for name in ("ours_byt5", "ours_byt5_en", "ours_tiny", "ours_tiny_en"):
            if name not in present:
                continue
            tr = Translator(*present[name], device=dev, guard=False)
            if "ml" not in tr.tasks:
                continue
            p_all = tr.translate([r["mixed"] for r in mixed], "ml", a.batch)
            p_plain = tr.translate([r["roman"] for r in mixed], "ml", a.batch)
            p_sw = [p for p, r in zip(p_all, mixed) if r["swaps"]]
            res[name] = {"mixed_all": score(p_all, [r["ml"] for r in mixed]),
                         "plain_all": score(p_plain, [r["ml"] for r in mixed]),
                         "mixed_with_swaps": score(p_sw, [r["ml"] for r in swapped]), **swap_rates(p_sw, swapped)}
            frame[name] = p_sw
            print(name, json.dumps(res[name]), flush=True)
            del tr
            torch.cuda.empty_cache()
        pd.DataFrame(frame).to_csv(out / "preds_mixed_ml.csv", index=False)

    summary_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, out / "en_summary.md")
    return summary


def write_markdown(s, path):
    lines = ["# English and code-mixed script", ""]
    if "en" in s:
        for st, r in s["en"].items():
            lines += [f"## Manglish -> English, {st} (BLEU / chrF, higher is better)", "",
                      "| system | " + " | ".join(INPUTS) + " |", "|" + "---|" * (len(INPUTS) + 1)]
            for name, v in r.items():
                if name in ("vs", "nllb_gold"):
                    continue
                lines.append(f"| {LABELS.get(name, name)} | " + " | ".join(
                    f"{v[i]['BLEU']:.1f} / {v[i]['chrF']:.1f}" if i in v else "-" for i in INPUTS) + " |")
            g = r.get("nllb_gold")
            if g:
                lines.append(f"| {LABELS['nllb_gold']} | {g['BLEU']:.1f} / {g['chrF']:.1f} (any input) | | |")
            if r.get("vs"):
                lines += ["", "Paired bootstrap on mean sentence chrF, direct model minus pipeline (positive favours ours):", ""]
                lines += [f"- {v['input']}: {LABELS.get(v.get('direct', 'ours_byt5_en'))} vs {LABELS.get(v['other'])}: "
                          f"{v['mean']:+.2f} (95% CI {v['lo']:+.2f} to {v['hi']:+.2f})" for v in r["vs"]]
            lines.append("")
    if "mixed" in s:
        m = s["mixed"]
        lines += [f"## Code-mixed Manglish -> Malayalam script ({m['n_with_swaps']} of {m['n']} sentences have English words)",
                  "", "| system | English words translated | left in Latin | CER / chrF (mixed) | CER / chrF (no English words) |",
                  "|---|---|---|---|---|"]
        for name, v in m.items():
            if isinstance(v, dict):
                lines.append(f"| {LABELS.get(name, name)} | {v['translated_pct']:.1f}% | {v['left_in_latin_pct']:.1f}% | "
                             f"{v['mixed_all']['CER']:.2f} / {v['mixed_all']['chrF']:.1f} | "
                             f"{v['plain_all']['CER']:.2f} / {v['plain_all']['chrF']:.1f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
