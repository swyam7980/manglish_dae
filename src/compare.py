"""Collect every metrics_*.json, rank baseline + backbones by CER, write results/summary.{csv,md} and a chart."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import RES_DIR, OUT_DIR


def main():
    rows = []
    for p in sorted(RES_DIR.glob("metrics_*.json")):
        tag = p.stem.replace("metrics_", "")
        if tag.endswith("_smoke"):
            continue
        d = json.loads(p.read_text())
        d["system"] = tag
        info = OUT_DIR / tag / "train_info.json"
        if info.exists():
            t = json.loads(info.read_text())
            d.update(trainable_M=round(t["trainable_params"] / 1e6, 2), train_min=t["minutes"],
                     peak_gpu_gb=t["peak_gpu_gb"], final_loss=round(t["last_logged_loss"], 4))
        rows.append(d)
    if not rows:
        raise SystemExit("No metrics files found. Run evaluate.py first.")
    cols = ["system", "CER", "WER", "BLEU", "chrF", "trainable_M", "train_min", "peak_gpu_gb", "final_loss"]
    df = pd.DataFrame(rows).reindex(columns=cols).sort_values("CER").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(RES_DIR / "summary.csv", index=False)
    (RES_DIR / "summary.md").write_text(df.fillna("-").to_markdown(index=False))
    print(df.to_string(index=False))

    base = df[df.system == "identity"]
    if not base.empty:
        for _, r in df[df.system != "identity"].iterrows():
            if r.CER >= base.CER.iloc[0] or r.WER >= base.WER.iloc[0]:
                print(f"WARNING: {r.system} does not beat the copy-input baseline on CER and WER: bug or undertrained, not a finding.")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, m in zip(ax, ["CER", "WER"]):
        a.barh(df["system"][::-1], df[m][::-1], color="#3769aa")
        a.set_title(m + " % (lower is better)")
    plt.tight_layout()
    plt.savefig(RES_DIR / "comparison.png", dpi=150)



if __name__ == "__main__":
    main()
