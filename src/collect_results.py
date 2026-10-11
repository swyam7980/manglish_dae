"""Copy the result files that reproduce the report into the repository's results/ folder.

Copies metrics, predictions, analyses, summaries and the study outputs. Does not copy trained adapters (outputs/),
the datasets (data/) or logs. Run with MDAE_WORKDIR set when the working data lives outside the repository.
"""
import shutil

from config import RES_DIR, PROC_DIR, ROOT

DEST = ROOT / "results"
TOP = ["metrics_*.json", "preds_*.csv", "analysis_*.md", "summary.csv", "summary.md", "seeds_summary.md",
       "report_data.json", "comparison.png"]
STUDY_DIRS = {"combined_all": ["*.csv", "*.json"], "decoding_all": ["decoding_detail.csv", "decoding_summary.*"],
              "robustness_all": ["*.csv", "*.json"]}
DATA_STATS = ["stats.json", "train_plus_all.stats.json"]


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def main():
    total, count = 0, 0
    for pattern in TOP:
        for f in sorted(RES_DIR.glob(pattern)):
            total += copy(f, DEST / f.name)
            count += 1
    for sub, patterns in STUDY_DIRS.items():
        for pattern in patterns:
            for f in sorted((RES_DIR / sub).glob(pattern)):
                total += copy(f, DEST / sub / f.name)
                count += 1
    for name in DATA_STATS:
        f = PROC_DIR / name
        if f.exists():
            total += copy(f, DEST / "data" / name)
            count += 1
    print(f"copied {count} files, {total / 1e6:.1f} MB into {DEST}")


if __name__ == "__main__":
    main()
