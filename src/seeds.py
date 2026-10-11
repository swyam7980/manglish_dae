"""Aggregate multi-seed results and paired-bootstrap model differences into results/report_data.json and seeds_summary.md."""
import json
import re

import numpy as np
import pandas as pd
import jiwer

from config import RES_DIR, OUT_DIR

SYSTEMS = {
    "mbart_large": "mBART-50",
    "byt5_base": "ByT5-base",
    "mt5_base_lr1e-3": "mT5-base (LR 1e-3, deviation)",
    "mt5_base": "mT5-base (LR 2e-4, as specified)",
    "mbart_large_plusmixed": "mBART-50 + Mixed-Code data",
    "byt5_base_plusmixed": "ByT5-base + Mixed-Code data",
    "mt5_base_plusmixed": "mT5-base + Mixed-Code data (LR 2e-4)",
    "mt5_base_lr1e-3_plusmixed": "mT5-base + Mixed-Code data (LR 1e-3)",
    "mbart_large_plusall": "mBART-50 + Mixed-Code + Dakshina",
    "byt5_base_plusall": "ByT5-base + Mixed-Code + Dakshina",
    "mt5_base_plusall": "mT5-base + Mixed-Code + Dakshina (LR 2e-4)",
    "mt5_base_lr1e-3_plusall": "mT5-base + Mixed-Code + Dakshina (LR 1e-3)",
}
METRICS = ["CER", "WER", "BLEU", "chrF"]
BOOT = 2000
PAIRS = [("mbart_large", "byt5_base"), ("mbart_large", "mt5_base_lr1e-3"), ("byt5_base", "mt5_base_lr1e-3"),
         ("mbart_large_plusall", "mbart_large"), ("byt5_base_plusall", "byt5_base"),
         ("byt5_base_plusall", "mbart_large_plusall")]
# the same questions on the larger combined test set (seed-42 predictions written by eval_combined.py)
COMBINED_PAIRS = [("mbart", "byt5"), ("mbart_plusall", "mbart"), ("byt5_plusall", "byt5"), ("byt5_plusall", "mbart_plusall"),
                  ("mt5_lr1e-3_plusall", "mt5_lr1e-3"), ("byt5_plusall", "mt5_lr1e-3_plusall")]
COMBINED_BOOT = 500
SHOW_EXAMPLES = 8


def seed_tags(base):
    tags = [base] + sorted(p.stem.replace("metrics_", "") for p in RES_DIR.glob(f"metrics_{base}_s*.json")
                           if re.search(r"_s\d+$", p.stem))
    return [t for t in tags if (RES_DIR / f"metrics_{t}.json").exists()]


def seed_of(tag):
    m = re.search(r"_s(\d+)$", tag)
    return int(m.group(1)) if m else 42


def bootstrap_diff(a, b, clean, rng, boot=BOOT):
    """Paired bootstrap of corpus CER(a) - CER(b) over validation sentences. Negative favours a."""
    n = len(clean)
    diffs = []
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        refs = [clean[i] for i in idx]
        diffs.append((jiwer.cer(refs, [a[i] for i in idx]) - jiwer.cer(refs, [b[i] for i in idx])) * 100)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def pattern_stats(tag):
    p = RES_DIR / f"analysis_{tag}.md"
    out = {}
    if p.exists():
        for k, v in re.findall(r"^- (\w+): ([\d.]+)$", p.read_text(encoding="utf-8"), flags=re.M):
            out[k] = float(v)
    return out


def main():
    data = {"systems": [], "baseline": json.loads((RES_DIR / "metrics_identity.json").read_text()), "pairs": [], "examples": []}
    preds = {}
    for base, label in SYSTEMS.items():
        tags = seed_tags(base)
        if not tags:
            continue
        runs = []
        for t in tags:
            m = json.loads((RES_DIR / f"metrics_{t}.json").read_text())
            runs.append({"seed": seed_of(t), **{k: m[k] for k in METRICS}})
        df = pd.DataFrame(runs)
        agg = {k: {"mean": float(df[k].mean()), "sd": float(df[k].std(ddof=1)) if len(df) > 1 else None} for k in METRICS}
        info = json.loads((OUT_DIR / base / "train_info.json").read_text())
        data["systems"].append({
            "key": base, "label": label, "n_seeds": len(runs), "runs": runs, "agg": agg,
            "train_min": info["minutes"], "peak_gpu_gb": info["peak_gpu_gb"], "trainable_pct": info["trainable_pct"],
            "precision": info["precision"], "loss_history": info["loss_history"], "patterns": pattern_stats(base),
            "train_rows": info.get("train_rows"), "micro_batch": info.get("micro_batch")})
        preds[base] = pd.read_csv(RES_DIR / f"preds_{base}.csv").fillna("")

    rng = np.random.default_rng(0)
    for a, b in PAIRS:
        if a in preds and b in preds:
            clean = preds[a]["clean"].tolist()
            mean, lo, hi = bootstrap_diff(preds[a]["pred"].tolist(), preds[b]["pred"].tolist(), clean, rng)
            data["pairs"].append({"a": a, "b": b, "mean": mean, "lo": lo, "hi": hi})

    data["pairs_combined"] = []
    cpreds = {}
    for name in {n for pair in COMBINED_PAIRS for n in pair}:
        p = RES_DIR / "combined_all" / f"preds_{name}.csv"
        if p.exists():
            cpreds[name] = pd.read_csv(p).fillna("")
    for a, b in COMBINED_PAIRS:
        if a in cpreds and b in cpreds:
            clean = cpreds[a]["clean"].tolist()
            mean, lo, hi = bootstrap_diff(cpreds[a]["pred"].tolist(), cpreds[b]["pred"].tolist(), clean, rng, COMBINED_BOOT)
            data["pairs_combined"].append({"a": a, "b": b, "mean": mean, "lo": lo, "hi": hi, "n": len(clean)})

    ref = next(iter(preds.values()))
    for i in range(min(SHOW_EXAMPLES, len(ref))):
        row = {"noisy": ref.noisy[i], "clean": ref.clean[i]}
        row.update({k: preds[k].pred[i] for k in preds})
        data["examples"].append(row)
    data["n_val"] = len(ref)

    (RES_DIR / "report_data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    lines = ["| system | seeds | CER | WER | BLEU | chrF |", "|---|---|---|---|---|---|"]
    for s in data["systems"]:
        cells = [f"{s['agg'][k]['mean']:.2f}" + (f" ± {s['agg'][k]['sd']:.2f}" if s["agg"][k]["sd"] is not None else "") for k in METRICS]
        lines.append(f"| {s['label']} | {s['n_seeds']} | " + " | ".join(cells) + " |")
    lines += ["", "Paired bootstrap on CER (seed 42, negative favours the first system):", ""]
    for p in data["pairs"]:
        lines.append(f"- {p['a']} - {p['b']}: {p['mean']:+.2f} (95% CI {p['lo']:+.2f} to {p['hi']:+.2f})")
    lines += ["", "Same comparisons on the combined test set (seed 42, negative favours the first system):", ""]
    for p in data["pairs_combined"]:
        lines.append(f"- {p['a']} - {p['b']}: {p['mean']:+.2f} (95% CI {p['lo']:+.2f} to {p['hi']:+.2f}, n={p['n']})")
    (RES_DIR / "seeds_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
