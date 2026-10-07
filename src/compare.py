"""Collect every metrics_*.json, rank the backbones by CER, write results/summary.csv and a chart."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import RES_DIR


def main():
    rows = []
    for p in sorted(RES_DIR.glob("metrics_*.json")):
        d = json.loads(p.read_text())
        d["system"] = p.stem.replace("metrics_", "")
        rows.append(d)
    if not rows:
        raise SystemExit("No metrics files found. Run evaluate.py first.")
    df = pd.DataFrame(rows)[["system", "CER", "WER", "BLEU", "chrF"]].sort_values("CER").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(RES_DIR / "summary.csv", index=False)
    print(df.to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, m in zip(ax, ["CER", "WER"]):
        a.barh(df["system"][::-1], df[m][::-1], color="#3769aa")
        a.set_title(m + " % (lower is better)")
    plt.tight_layout()
    plt.savefig(RES_DIR / "comparison.png", dpi=150)


if __name__ == "__main__":
    main()
