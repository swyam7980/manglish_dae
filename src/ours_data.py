"""Malayalam-script targets for our own model, taken from the same split the comparison used.

Dakshina ships its romanised sentences line-aligned with the native-script originals (ml.romanized.rejoined.
{dev,test}.{roman,native}.txt). Each romanised line goes through the exact sanitiser that built train_plus_all, so a
clean Manglish sentence in our train or test file can be looked up and given its Malayalam original. Lines the
sanitiser split into clauses lose their alignment and get no script target.

Train pairs come only from training sentences and test pairs only from test sentences, so the split is the one every
other model was scored on. Dakshina's word lexicon (its own training split) is added as short word pairs.

Output: data/processed/ours_ml_train.jsonl (roman, native, kind), ours_ml_test.jsonl (clean, noisy, native) and
ours_ml.stats.json.
"""
import argparse
import json
import re
import sys
import tarfile
import unicodedata
from collections import defaultdict
from pathlib import Path

from config import PROC_DIR, RAW_DIR
import sanitize
from extra_data import read_jsonl, write_rows

LEX_CAP = 8000
SPLITS = ("dev", "test")
MALAYALAM = re.compile(r"[ഀ-ൿ]")


def tidy_native(s):
    """NFC, no stray outer punctuation, single spaces: the native counterpart of sanitize.tidy."""
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s.strip(" ,;:.!?-'\"")).strip()


def read_dakshina_members(path, names):
    """{name: text} for the requested romanized/lexicon files, from the tar or an extracted directory."""
    p, out = Path(path), {}
    if p.is_file() and tarfile.is_tarfile(p):
        with tarfile.open(p) as tf:
            for m in tf:
                base = m.name.rsplit("/", 1)[-1]
                if m.isfile() and base in names and "/ml/" in m.name:
                    out[base] = tf.extractfile(m).read().decode("utf-8")
    elif p.is_dir():
        for f in p.rglob("*"):
            if f.name in names and "/ml/" in f.as_posix():
                out[f.name] = f.read_text(encoding="utf-8")
    missing = set(names) - set(out)
    if missing:
        sys.exit(f"Missing Dakshina files under {p}: {sorted(missing)}")
    return out


def sentence_map(files):
    """clean Manglish sentence -> Malayalam original, for lines the sanitiser keeps whole."""
    mapping, n_pairs, n_kept = {}, 0, 0
    for split in SPLITS:
        roman = files[f"ml.romanized.rejoined.{split}.roman.txt"].splitlines()
        native = files[f"ml.romanized.rejoined.{split}.native.txt"].splitlines()
        assert len(roman) == len(native), f"{split}: roman/native line counts differ"
        for r, n in zip(roman, native):
            n_pairs += 1
            clean = sanitize.tidy(sanitize.normalize(r))
            nat = tidy_native(n)
            if sanitize.reject_reason(clean) or not MALAYALAM.search(nat):
                continue
            n_kept += 1
            mapping.setdefault(clean, nat)
    return mapping, {"aligned_lines": n_pairs, "usable_lines": n_kept, "unique_sentences": len(mapping)}


def lexicon_pairs(text, banned_roman, cap=LEX_CAP):
    """Most-attested romanisation of each native word (ties: first listed); words seen in test sentences are skipped."""
    best = defaultdict(lambda: ("", -1))
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        nat, rom, cnt = unicodedata.normalize("NFC", parts[0]), parts[1].strip().lower(), int(parts[2])
        if rom.isascii() and rom.isalpha() and cnt > best[nat][1]:
            best[nat] = (rom, cnt)
    pairs = [(rom, nat) for nat, (rom, _) in best.items() if rom and rom not in banned_roman]
    pairs.sort(key=lambda p: (len(p[0]), p[0]))  # deterministic, then thinned evenly across lengths
    step = max(1, len(pairs) // cap) if cap else 1
    return pairs[::step][:cap] if cap else pairs


def build(dakshina, train_file="train_plus_all.jsonl", test_file="test_all.jsonl", lex_cap=LEX_CAP):
    names = {f"ml.romanized.rejoined.{s}.{k}.txt" for s in SPLITS for k in ("roman", "native")}
    names.add("ml.translit.sampled.train.tsv")
    files = read_dakshina_members(dakshina, names)
    mapping, stats = sentence_map(files)

    train = read_jsonl(PROC_DIR / train_file)
    test = read_jsonl(PROC_DIR / test_file)
    tr = [{"roman": r["clean"], "native": mapping[r["clean"]], "kind": "sentence"}
          for r in train if r["source"] == "dakshina_ml" and r["clean"] in mapping]
    te = [{"clean": r["clean"], "noisy": r["noisy"], "native": mapping[r["clean"]]}
          for r in test if r["source"] == "dakshina_ml" and r["clean"] in mapping]
    test_words = {w for r in te for w in re.findall(r"[a-z]+", r["clean"])}
    lex = lexicon_pairs(files["ml.translit.sampled.train.tsv"], test_words, lex_cap)
    tr += [{"roman": r, "native": n, "kind": "word"} for r, n in lex]
    assert not {r["roman"] for r in tr} & {r["clean"] for r in te}, "script train/test overlap"
    stats.update(train_sentences=sum(r["kind"] == "sentence" for r in tr), train_words=len(lex), test_sentences=len(te),
                 dakshina_train_rows=sum(r["source"] == "dakshina_ml" for r in train),
                 dakshina_test_rows=sum(r["source"] == "dakshina_ml" for r in test))
    return tr, te, stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dakshina", default=str(RAW_DIR / "dakshina_dataset_v1.0.tar"))
    ap.add_argument("--lex-cap", type=int, default=LEX_CAP, help="most lexicon word pairs to add (0 = all)")
    a = ap.parse_args(argv)
    tr, te, stats = build(a.dakshina, lex_cap=a.lex_cap)
    write_rows(PROC_DIR / "ours_ml_train.jsonl", tr)
    write_rows(PROC_DIR / "ours_ml_test.jsonl", te)
    (PROC_DIR / "ours_ml.stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    main()
