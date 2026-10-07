"""Load SeLLaI Mal-Eng, keep the Manglish class, clean, split 90/10 (seed 42), add noise, save JSONL."""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

from config import (RAW_DIR, PROC_DIR, SEED, VAL_FRACTION, MIN_CHARS, MAX_CHARS, MIN_ASCII_FRACTION,
                    NOISE_SEED_TRAIN, NOISE_SEED_VAL)
from noise import corrupt_all

DEMO_WORDS = ("njan nale varum ente veedu evide aanu nee enthu parayunnu avan ivide illa "
              "ammaye ariyo sheri kollam adipoli povilla mone chechi super machane eda "
              "padam kandu nannayittund ente favourite aanu athu valare nalla movie").split()


def make_demo(path, n=2000):
    """Synthetic file with the same columns as SeLLaI. Only for testing the pipeline, never for results."""
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
    """Strip URLs and @mentions, collapse whitespace, lower-case (proposal Sec. 6.2)."""
    s = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", str(s))
    return re.sub(r"\s+", " ", s).strip().lower()


def build_sentences(raw):
    """raw: list of strings of the chosen class. Returns (unique clean sentences, per-filter counts)."""
    stats = {"class_rows": len(raw)}
    sents = [clean(s) for s in raw]
    sents = [s for s in sents if s]
    stats["after_nonempty"] = len(sents)
    sents = [s for s in sents if len(s) >= MIN_CHARS and re.search(r"[a-z]", s)]
    stats["after_min_len_and_has_letters"] = len(sents)
    sents = [s for s in sents if len(s) <= MAX_CHARS]
    stats["after_max_len"] = len(sents)
    sents = [s for s in sents if sum(c.isascii() for c in s) / len(s) >= MIN_ASCII_FRACTION]
    stats["after_latin_script"] = len(sents)
    seen, uniq = set(), []
    for s in sents:
        if s not in seen:  # already lower-cased, so this is case-insensitive dedupe
            seen.add(s)
            uniq.append(s)
    stats["after_dedupe"] = len(uniq)
    return uniq, stats


def split_and_noise(uniq):
    rng = random.Random(SEED)
    idx = list(range(len(uniq)))
    rng.shuffle(idx)
    n_val = max(1, int(len(uniq) * VAL_FRACTION))
    val = [uniq[i] for i in idx[:n_val]]
    train = [uniq[i] for i in idx[n_val:]]
    assert not set(train) & set(val), "train/val overlap"
    return {"train": (train, corrupt_all(train, NOISE_SEED_TRAIN)),
            "val": (val, corrupt_all(val, NOISE_SEED_VAL))}


def write_splits(splits, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (clean_list, noisy) in splits.items():
        # newline="\n" keeps the files byte-identical across operating systems
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for c, n in zip(clean_list, noisy):
                f.write(json.dumps({"clean": c, "noisy": n}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="path to the SeLLaI file (default: first csv/xlsx/tsv in data/raw)")
    ap.add_argument("--label", default="manglish", help="class name to keep (case-insensitive exact match)")
    ap.add_argument("--demo", action="store_true", help="use a synthetic file to smoke-test the pipeline")
    a = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    if a.demo:
        path = RAW_DIR / "demo_sellai_like.csv"
        make_demo(path)
    elif a.file:
        path = Path(a.file)
        if not path.exists():
            sys.exit(f"File not found: {path}")
    else:
        cands = [p for p in sorted(RAW_DIR.iterdir())
                 if p.suffix.lower() in (".csv", ".xlsx", ".xls", ".tsv") and not p.name.startswith("demo_")]
        if not cands:
            sys.exit(f"No data file in {RAW_DIR}. Download SeLLaI Mal-Eng from Mendeley Data "
                     "(DOI 10.17632/5p4zbpy8wz.1, 'Download All') and put the file there. See data/README.md.")
        path = cands[0]
    print(f"Reading {path}")
    df = read_any(path)
    print("columns:", list(df.columns), "| rows:", len(df))

    scol = find_col(df, ["sentence", "text", "sentences"])
    lcol = find_col(df, ["language", "label", "class", "category"])
    if scol is None:
        sys.exit(f"No sentence column found among {list(df.columns)}; expected 'sentence'.")
    if lcol is None:
        sys.exit(f"No class column found among {list(df.columns)}; expected 'language'.")
    counts = df[lcol].astype(str).value_counts()
    print("\nclass counts:\n" + counts.to_string())
    mask = df[lcol].astype(str).str.strip().str.lower() == a.label.strip().lower()
    if not mask.any():
        sys.exit(f"No rows have class '{a.label}'. Pick the exact spelling from the class counts above with --label.")
    raw = df.loc[mask, scol].dropna().astype(str).tolist()

    uniq, stats = build_sentences(raw)
    print("\nfilter funnel:")
    for k, v in stats.items():
        print(f"  {k:32s} {v}")
    if len(uniq) < 100:
        sys.exit(f"Only {len(uniq)} usable sentences after filtering; refusing to continue.")

    splits = split_and_noise(uniq)
    write_splits(splits, PROC_DIR)
    stats.update(train=len(splits["train"][0]), val=len(splits["val"][0]),
                 class_counts={k: int(v) for k, v in counts.items()}, source_file=path.name)
    (PROC_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\ntrain={stats['train']}  val={stats['val']}  -> {PROC_DIR}")
    print("train/val sentence overlap: 0 (asserted)")
    print("sample clean -> noisy pairs:")
    for c, n in list(zip(*splits["val"]))[:5]:
        print(f"  {c!r}\n   -> {n!r}")


if __name__ == "__main__":
    main()
