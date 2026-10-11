"""Build a larger training file from several sources while keeping the validation set exactly as it is.

Sources: another SeLLaI class (e.g. Mixed-Code), the Romanised Malayalam sentences of the Dakshina dataset
(directory or .tar), or any plain-text file / CSV column. Every source goes through the same cleaning filters as the
Manglish class, sentences that appear in the validation set (exact or ignoring spaces and punctuation) are dropped,
duplicates across sources are removed, and noise is added with the training noise settings.

Every extra source also gives up --holdout-fraction of its sentences (default 10%) to a combined test set, so models
can be evaluated on all sources at once and no test sentence is ever trained on. The original 325 validation
sentences are part of that test set, so earlier results stay comparable.

Output: data/processed/train_plus_<tag>.jsonl and test_<tag>.jsonl (rows: clean, noisy, source) and a stats file.
"""
import argparse
import json
import random
import re
import sys
import tarfile
from pathlib import Path

import pandas as pd

from config import RAW_DIR, PROC_DIR, SEED, NOISE_SEED_TRAIN, NOISE_SEED_VAL
import sanitize
from data_prep import read_any, find_col
from noise import corrupt_all

EXTRA_NOISE_OFFSET = 7919  # extra sentences get their own noise stream, separate from the base training noise
HOLDOUT_FRACTION = 0.10
NL = chr(10)  # written files use LF on every operating system


def norm_key(s):
    """Spacing- and punctuation-insensitive form used to find near-duplicates of validation sentences."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_sellai_class(path, label):
    df = read_any(Path(path))
    scol = find_col(df, ["sentence", "text", "sentences"])
    lcol = find_col(df, ["language", "label", "class", "category"])
    if scol is None or lcol is None:
        sys.exit(f"{path}: expected 'sentence' and 'language' columns, found {list(df.columns)}")
    mask = df[lcol].astype(str).str.strip().str.lower() == label.strip().lower()
    if not mask.any():
        sys.exit(f"{path}: no rows with class '{label}'. Classes: {sorted(df[lcol].astype(str).unique())[:8]}")
    return df.loc[mask, scol].dropna().astype(str).tolist()


def load_text_file(path, column=None):
    p = Path(path)
    if p.suffix.lower() in (".csv", ".tsv", ".xlsx", ".xls"):
        df = read_any(p)
        col = column or find_col(df, ["sentence", "text", "sentences"]) or df.columns[0]
        return df[col].dropna().astype(str).tolist()
    return [line.rstrip("\n") for line in open(p, encoding="utf-8") if line.strip()]


def load_dakshina(path, lang="ml"):
    """Romanised sentences from a Dakshina directory or the dakshina_dataset_v1.0.tar archive.
    Reads the *.roman.txt files of the romanized folder only (streamed from the tar, nothing else is extracted)."""
    p = Path(path)
    lines, found = [], []
    if p.is_file() and tarfile.is_tarfile(p):
        with tarfile.open(p) as tf:
            for m in tf:
                if m.isfile() and f"/{lang}/romanized/" in m.name and m.name.endswith(".roman.txt"):
                    found.append(m.name)
                    lines += tf.extractfile(m).read().decode("utf-8").splitlines()
    elif p.is_dir():
        for f in sorted(p.rglob("*.roman.txt")):
            if f"/{lang}/romanized/" in f.as_posix():
                found.append(f.as_posix())
                lines += f.read_text(encoding="utf-8").splitlines()
    else:
        sys.exit(f"Dakshina path not found or not a directory/tar: {p}")
    if not lines:
        sys.exit(f"No '{lang}/romanized/*.roman.txt' files found under {p}. Expected the layout of dakshina_dataset_v1.0.")
    print(f"[dakshina] {len(lines)} lines from {len(found)} files: {[Path(x).name for x in found]}")
    return lines


def build_extra(sources, val_clean, base_train_clean, cap_per_source=0, holdout_fraction=0.0, split_long=()):
    """sources: {name: raw sentence list}. Returns (train rows, held-out rows, stats); rows are (sentence, source).
    Order is deterministic. Sentences equal to a validation sentence (exact or ignoring spaces/punctuation) and
    duplicates within or across sources are dropped before the train/held-out split, so the two never share one."""
    banned_exact = set(val_clean)
    banned_keys = {norm_key(s) for s in val_clean}
    seen = set(base_train_clean) | banned_exact
    seen_keys = {norm_key(s) for s in base_train_clean} | banned_keys
    train_rows, held_rows, stats = [], [], {}
    for name, raw in sources.items():
        uniq, funnel = sanitize.sanitize_many(raw, split_long=name in split_long)
        kept, dropped_val, dropped_dup = [], 0, 0
        for s in uniq:
            k = norm_key(s)
            if s in banned_exact or k in banned_keys:
                dropped_val += 1
            elif s in seen or k in seen_keys:
                dropped_dup += 1
            else:
                kept.append(s)
                seen.add(s)
                seen_keys.add(k)
        rng = random.Random(SEED)
        rng.shuffle(kept)
        if cap_per_source and len(kept) > cap_per_source:
            kept = kept[:cap_per_source]
        n_held = int(len(kept) * holdout_fraction)
        held, train = kept[:n_held], kept[n_held:]
        stats[name] = {**funnel, "dropped_val_overlap": dropped_val, "dropped_duplicates": dropped_dup,
                       "train_added": len(train), "held_out": len(held)}
        train_rows += [(s, name) for s in train]
        held_rows += [(s, name) for s in held]
    return train_rows, held_rows, stats


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline=NL) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + NL)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="name of the output file: train_plus_<tag>.jsonl")
    ap.add_argument("--sellai-file", help="SeLLaI file (default: first csv in data/raw)")
    ap.add_argument("--sellai-label", action="append", default=[], help="extra SeLLaI class to add (repeatable)")
    ap.add_argument("--dakshina", help="Dakshina directory or dakshina_dataset_v1.0.tar")
    ap.add_argument("--text-file", action="append", default=[], help="extra sentences: .txt or a csv/tsv/xlsx (repeatable)")
    ap.add_argument("--text-column", help="column name for --text-file when it is a table")
    ap.add_argument("--cap-per-source", type=int, default=0, help="keep at most N sentences from each source (0 = all)")
    ap.add_argument("--split-long", action="store_true", help="also cut over-long --text-file lines at clause boundaries")
    ap.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION,
                    help="share of each extra source reserved for the combined test set")
    a = ap.parse_args(argv)

    for need in ("train.jsonl", "val.jsonl"):
        if not (PROC_DIR / need).exists():
            sys.exit(f"{PROC_DIR / need} missing. Run data_prep.py first.")
    base_train = read_jsonl(PROC_DIR / "train.jsonl")
    val = read_jsonl(PROC_DIR / "val.jsonl")

    sources = {}
    if a.sellai_label:
        path = a.sellai_file or next((str(q) for q in sorted(RAW_DIR.iterdir())
                                      if q.suffix.lower() in (".csv", ".xlsx", ".tsv") and not q.name.startswith("demo_")), None)
        if not path:
            sys.exit(f"No SeLLaI file in {RAW_DIR}; pass --sellai-file.")
        for label in a.sellai_label:
            sources[f"sellai_{label.lower()}"] = load_sellai_class(path, label)
    if a.dakshina:
        sources["dakshina_ml"] = load_dakshina(a.dakshina)
    for t in a.text_file:
        sources[f"file_{Path(t).stem}"] = load_text_file(t, a.text_column)
    if not sources:
        sys.exit("No extra source given: use --sellai-label, --dakshina or --text-file.")

    split_long_sources = {"dakshina_ml"} | ({n for n in sources if n.startswith("file_")} if a.split_long else set())
    extra, held, stats = build_extra(sources, [r["clean"] for r in val], [r["clean"] for r in base_train],
                                     a.cap_per_source, a.holdout_fraction, split_long_sources)
    clean = [s for s, _ in extra]
    noisy = corrupt_all(clean, NOISE_SEED_TRAIN + EXTRA_NOISE_OFFSET)
    rows = [{"clean": r["clean"], "noisy": r["noisy"], "source": "sellai_manglish"} for r in base_train]
    rows += [{"clean": c, "noisy": n, "source": src} for c, n, (_, src) in zip(clean, noisy, extra)]
    random.Random(SEED).shuffle(rows)

    held_noisy = corrupt_all([s for s, _ in held], NOISE_SEED_VAL + EXTRA_NOISE_OFFSET)
    test = [{"clean": r["clean"], "noisy": r["noisy"], "source": "sellai_manglish"} for r in val]
    test += [{"clean": c, "noisy": n, "source": src} for c, n, (_, src) in zip([s for s, _ in held], held_noisy, held)]
    assert not {r["clean"] for r in rows} & {r["clean"] for r in test}, "train/test overlap"

    out = PROC_DIR / f"train_plus_{a.tag}.jsonl"
    write_rows(out, rows)
    write_rows(PROC_DIR / f"test_{a.tag}.jsonl", test)
    audit = sanitize.audit([r["clean"] for r in base_train + val])
    summary = {"base_manglish_audit_unchanged": audit, "base_train": len(base_train), "extra_added": len(extra), "train_total": len(rows),
               "test_total": len(test), "per_source": stats,
               "train_rows_by_source": pd.Series([r["source"] for r in rows]).value_counts().to_dict(),
               "test_rows_by_source": pd.Series([r["source"] for r in test]).value_counts().to_dict()}
    (PROC_DIR / f"train_plus_{a.tag}.stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out} and test_{a.tag}.jsonl")
    return summary


if __name__ == "__main__":
    main()
