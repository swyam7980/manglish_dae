"""Load SeLLaI Mal-Eng, keep the Manglish class, clean, split 90/10 (seed 42), add noise, save JSONL."""
import argparse
import json
import random
import re
import sys

import pandas as pd

from config import RAW_DIR, PROC_DIR, SEED, VAL_FRACTION, MIN_CHARS, MAX_CHARS
from noise import corrupt_all

DEMO_WORDS = ("njan nale varum ente veedu evide aanu nee enthu parayunnu avan ivide illa "
              "ammaye ariyo sheri kollam adipoli povilla mone chechi super machane eda "
              "padam kandu nannayittund ente favourite aanu athu valare nalla movie").split()


def make_demo(path, n=2000):
    """Tiny synthetic file with the same columns as SeLLaI, only for testing the pipeline."""
    rng = random.Random(0)
    rows = []
    for _ in range(n):
        k = rng.randint(3, 10)
        rows.append({"sentence": " ".join(rng.choice(DEMO_WORDS) for _ in range(k)), "language": "Manglish"})
    for _ in range(n // 4):
        rows.append({"sentence": "this is a plain english sentence number %d" % rng.randint(0, 999), "language": "English"})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[demo] wrote synthetic file: {path}")


def read_any(path):
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def find_col(df, names):
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


def clean(s):
    s = re.sub(r"https?://\S+|@\w+", " ", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="path to the SeLLaI file (default: first csv/xlsx/tsv in data/raw)")
    ap.add_argument("--label", default="manglish", help="class name to keep (case-insensitive substring)")
    ap.add_argument("--demo", action="store_true", help="use a synthetic file to smoke-test the pipeline")
    a = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    if a.demo:
        path = RAW_DIR / "demo_sellai_like.csv"
        make_demo(path)
    elif a.file:
        from pathlib import Path
        path = Path(a.file)
    else:
        cands = [p for p in sorted(RAW_DIR.iterdir()) if p.suffix.lower() in (".csv", ".xlsx", ".xls", ".tsv") and not p.name.startswith("demo_")]
        if not cands:
            sys.exit(f"No data file in {RAW_DIR}. Download SeLLaI Mal-Eng from Mendeley Data (see handout, Step 3).")
        path = cands[0]
    print(f"Reading {path}")
    df = read_any(path)
    print("columns:", list(df.columns), "| rows:", len(df))

    scol = find_col(df, ["sentence", "text", "sentences"]) or df.columns[0]
    lcol = find_col(df, ["language", "label", "class", "category"])
    if lcol is not None:
        print("\nclass counts:\n" + df[lcol].astype(str).value_counts().to_string())
        keep = df[df[lcol].astype(str).str.lower().str.contains(a.label.lower(), regex=False)]
        if keep.empty:
            sys.exit(f"No rows match label '{a.label}'. Pick one from the class counts above with --label.")
    else:
        print("WARNING: no class column found, treating every row as Manglish.")
        keep = df
    sents = [clean(s) for s in keep[scol].dropna().tolist()]
    n0 = len(sents)
    sents = [s for s in sents if MIN_CHARS <= len(s) <= MAX_CHARS and re.search(r"[A-Za-z]", s)]
    sents = [s for s in sents if sum(c.isascii() for c in s) / len(s) > 0.95]
    seen, uniq = set(), []
    for s in sents:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    print(f"\nManglish rows: {n0} -> after length/script filter + dedupe: {len(uniq)}")

    rng = random.Random(SEED)
    idx = list(range(len(uniq)))
    rng.shuffle(idx)
    n_val = max(1, int(len(uniq) * VAL_FRACTION))
    val_i, train_i = idx[:n_val], idx[n_val:]
    train = [uniq[i] for i in train_i]
    val = [uniq[i] for i in val_i]
    for name, clean_list in (("train", train), ("val", val)):
        noisy = corrupt_all(clean_list, seed=1 if name == "train" else 2)
        with open(PROC_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for c, n in zip(clean_list, noisy):
                f.write(json.dumps({"clean": c, "noisy": n}, ensure_ascii=False) + "\n")
    print(f"train={len(train)}  val={len(val)}  -> {PROC_DIR}")
    print("example:", json.loads(open(PROC_DIR / "val.jsonl", encoding="utf-8").readline()))


if __name__ == "__main__":
    main()
