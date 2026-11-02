"""Manglish -> English pairs (and code-mixed Manglish -> Malayalam pairs) from Malayalam-English parallel corpora.

Nobody publishes Manglish-English pairs, so we make them: the Malayalam side of each pair is written in Latin letters
the way people type Manglish, word by word, using the romanisations real people gave in Dakshina's lexicon (its own
training split; a word with several spellings keeps all of them with their counts, so each epoch can draw a different
one). Words the lexicon lacks go through a rule-based romaniser (`romanize_word`).

Code-mixing: Manglish writers drop English words in ("njan tomorrow varum"). A small dictionary mined from the pairs
(co-occurrence between Malayalam word forms and English words, see `mine_dictionary`) marks, for each pair, the
Malayalam words whose English equivalent is in that pair's English sentence. Training swaps some of them for the
English word, so the model learns to translate them: into English for the en: task, into the Malayalam word for ml:.

Sources (data/raw/; see data/README.md): BPCC human sets (seed, ILCI, Wiki, Massive, Daily), PMIndia, OPUS Wikimedia
and Tatoeba. Test: IN22-Conv and IN22-Gen (never trained on; pairs whose English or Malayalam match an IN22 sentence or
a Dakshina script-test sentence are dropped).

Output: data/processed/ours_en_train.jsonl  {ml, en, source, toks: [[[roman, count], ...], swap], ...]}
        data/processed/ours_en_test.jsonl   {set, ml, en, roman, noisy, mixed, swaps: [[malayalam, english], ...]}
        data/processed/ours_en.stats.json
"""
import argparse
import csv
import io
import json
import random
import re
import sys
import tarfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from config import PROC_DIR, RAW_DIR, SEED
from extra_data import read_jsonl, write_rows
from noise import corrupt
from ours_data import read_dakshina_members

csv.field_size_limit(10 ** 7)
MAX_ROMAN = 140           # characters of romanised Malayalam (the clean task's sentences are capped at 120)
MAX_EN = 160
MIN_WORDS = 2
TEST_SWAP_P = 0.3         # share of swappable words written in English in the code-mixed test inputs
# a Malayalam form maps to the English word found in at least P_E_GIVEN_M of its pairs (inflected forms each map
# to the same word, which Dice would penalise), seen together PAIR_MIN times, and not drowned in that word's pairs
P_E_GIVEN_M, P_M_GIVEN_E, PAIR_MIN, FREQ_MIN = 0.5, 0.02, 8, 8

# ---------------------------------------------------------------- Malayalam text
ZWNJ, ZWJ = chr(0x200C), chr(0x200D)
OLD_CHILLU = {base + "്" + ZWJ: new for base, new in
              {"ണ": "ൺ", "ന": "ൻ", "ര": "ർ", "ല": "ൽ", "ള": "ൾ", "ക": "ൿ"}.items()}
MALAYALAM = re.compile(r"[ഀ-ൿ]")


def norm_ml(s):
    """NFC, atomic chillus (old ZWJ forms are common in BPCC), no zero-width joiners, single spaces."""
    s = unicodedata.normalize("NFC", s)
    for old, new in OLD_CHILLU.items():
        s = s.replace(old, new)
    s = s.replace(ZWNJ, "").replace(ZWJ, "")
    s = s.replace("ൻ്റ", "ന്റ").replace("ൻറ", "ന്റ")  # the two other encodings of ന്റ
    return re.sub(r"\s+", " ", s).strip()


def norm_en(s):
    s = unicodedata.normalize("NFKC", s).replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()


def key(s):
    """Loose identity for de-duplication and test exclusion: letters and digits only, lower-cased."""
    return re.sub(r"[\W_]+", "", s.lower())


VOWELS = {"അ": "a", "ആ": "aa", "ഇ": "i", "ഈ": "ee", "ഉ": "u", "ഊ": "oo", "ഋ": "ru", "എ": "e", "ഏ": "e", "ഐ": "ai",
          "ഒ": "o", "ഓ": "o", "ഔ": "au"}
SIGNS = {"ാ": "aa", "ി": "i", "ീ": "ee", "ു": "u", "ൂ": "oo", "ൃ": "ru", "െ": "e", "േ": "e", "ൈ": "ai", "ൊ": "o",
         "ോ": "o", "ൌ": "au", "ൗ": "au"}
CONS = {"ക": "k", "ഖ": "kh", "ഗ": "g", "ഘ": "gh", "ങ": "ng", "ച": "ch", "ഛ": "chh", "ജ": "j", "ഝ": "jh", "ഞ": "nj",
        "ട": "t", "ഠ": "th", "ഡ": "d", "ഢ": "dh", "ണ": "n", "ത": "th", "ഥ": "th", "ദ": "d", "ധ": "dh", "ന": "n",
        "പ": "p", "ഫ": "f", "ബ": "b", "ഭ": "bh", "മ": "m", "യ": "y", "ര": "r", "റ": "r", "ല": "l", "ള": "l",
        "ഴ": "zh", "വ": "v", "ശ": "sh", "ഷ": "sh", "സ": "s", "ഹ": "h"}
CHILLU = {"ൺ": "n", "ൻ": "n", "ർ": "r", "ൽ": "l", "ൾ": "l", "ൿ": "k", "ം": "m", "ഃ": "h"}
VIRAMA = "്"
CLUSTERS = [("ന്റ", "nt"), ("റ്റ", "tt"), ("ങ്ങ", "ng"), ("ഞ്ഞ", "nj"), ("ക്ഷ", "ksh"), ("ന്ത", "nth"), ("ങ്ക", "nk")]
TIDY = [("thth", "tth"), ("chch", "cch"), ("aaa", "aa")]


def romanize_word(w):
    """Rule-based Manglish spelling of one Malayalam word: inherent 'a', final virama as 'u' (ഇന്ന് -> innu)."""
    for a, b in CLUSTERS:
        w = w.replace(a, "\x00" + b + "\x01")  # placeholder consonant cluster that still takes a vowel
    out, i, pending = [], 0, False        # pending: last consonant has no vowel yet
    while i < len(w):
        c = w[i]
        if c == "\x00":
            j = w.index("\x01", i)
            if pending:
                out.append("a")
            out.append(w[i + 1:j])
            pending, i = True, j + 1
            continue
        if c in CONS:
            if pending:
                out.append("a")
            out.append(CONS[c])
            pending = True
        elif c in SIGNS:
            out.append(SIGNS[c])
            pending = False
        elif c == VIRAMA:
            if i == len(w) - 1:
                out.append("u")
            pending = False
        else:
            if pending:
                out.append("a")
            pending = False
            out.append(VOWELS.get(c) or CHILLU.get(c) or (c.lower() if c.isascii() else ""))
        i += 1
    if pending:
        out.append("a")
    s = "".join(out)
    for a, b in TIDY:
        s = s.replace(a, b)
    return s


def load_lexicon(dakshina):
    """native word -> [[roman, count], ...] most-attested first, from Dakshina's training lexicon."""
    name = "ml.translit.sampled.train.tsv"
    text = read_dakshina_members(dakshina, {name})[name]
    lex = defaultdict(Counter)
    for line in text.splitlines():
        p = line.split("\t")
        if len(p) >= 3 and p[1].isascii() and p[1].isalpha():
            lex[norm_ml(p[0])][p[1].lower()] += int(p[2])
    return {w: [list(x) for x in c.most_common(4)] for w, c in lex.items()}


WORD = re.compile(r"[ഀ-ൿ]+|[A-Za-z]+|\d+(?:[.,]\d+)*")


def ml_words(s):
    return WORD.findall(s)


def roman_options(w, lex):
    if MALAYALAM.match(w):
        return lex.get(w) or [[romanize_word(w), 1]]
    return [[w.lower(), 1]]


def draw(opts, rng):
    total = sum(c for _, c in opts)
    r = rng.random() * total
    for rom, c in opts:
        r -= c
        if r < 0:
            return rom
    return opts[-1][0]


def render(toks, rng, swap_p=0.0):
    """One Manglish rendering of a tokenised pair: a sampled spelling per word, some swappable words in English."""
    out = []
    for opts, swap in toks:
        out.append(swap if swap and rng.random() < swap_p else draw(opts, rng))
    return " ".join(out)


# ---------------------------------------------------------------- sources
def read_tsv_pairs(path):
    """BPCC tsv files: header names the columns (order differs between subsets); eng_Latn/mal_Mlym in src/tgt."""
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            en, ml = (row["src"], row["tgt"]) if row.get("src_lang") == "eng_Latn" else (row["tgt"], row["src"])
            yield en, ml


def read_sources(raw):
    bpcc = raw / "bpcc"
    for sub in ("daily", "massive", "bpcc-seed-latest", "ilci", "wiki"):
        p = bpcc / sub / "mal_Mlym.tsv"
        if p.exists():
            for en, ml in read_tsv_pairs(p):
                yield "bpcc_" + ("seed" if sub.startswith("bpcc") else sub), en, ml
    p = raw / "pmindia" / "pmindia.v1.ml-en.tsv"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                yield "pmindia", parts[0], parts[1]
    for name in ("Tatoeba", "wikimedia"):
        d = raw / "opus" / name
        if (d / f"{name}.en-ml.en").exists():
            ens = (d / f"{name}.en-ml.en").read_text(encoding="utf-8").splitlines()
            mls = (d / f"{name}.en-ml.ml").read_text(encoding="utf-8").splitlines()
            for en, ml in zip(ens, mls):
                yield f"opus_{name.lower()}", en, ml


def read_in22(raw):
    """{set: [(en, ml), ...]} from the IN22 benchmark tarball (IN22-Conv repo ships both conv and gen)."""
    tar = raw / "IN22-Conv" / "IN22_benchmark.tar.gz"
    if not tar.exists():
        sys.exit(f"{tar} missing: download ai4bharat/IN22-Conv (see data/README.md).")
    files = {}
    with tarfile.open(tar) as tf:
        for m in tf:
            if m.isfile() and m.name.rsplit("/", 1)[-1] in ("test.eng_Latn", "test.mal_Mlym"):
                files[m.name] = tf.extractfile(m).read().decode("utf-8").splitlines()
    out = {}
    for s in ("conv", "gen"):
        en, ml = files[f"IN22_benchmark/{s}/test.eng_Latn"], files[f"IN22_benchmark/{s}/test.mal_Mlym"]
        out[f"in22_{s}"] = [(norm_en(e), norm_ml(m)) for e, m in zip(en, ml)]
    return out


# ---------------------------------------------------------------- code-mixing dictionary
STOP = set("""the a an and or but if of to in on at by for with from as is are was were be been being am do does did
have has had will would shall should can could may might must this that these those it its he she they them his her
their we us our you your i me my mine not no yes there here what which who whom whose when where why how all any
some each every very too also just only than then so such into over under about after before again more most other
own same few both up down out off one two three let lets get got go going come came make made take took give gave
said say says see saw know knew want please okay ok""".split())
EN_WORD = re.compile(r"[a-z]+")


def en_words(s):
    return {w for w in EN_WORD.findall(s.lower()) if len(w) >= 3 and w not in STOP}


def mine_dictionary(pairs):
    """Malayalam word form -> English word, by co-occurrence over the training pairs (one-best per form)."""
    m_df, e_df = Counter(), Counter()
    for en, ml in pairs:
        m_df.update(set(w for w in ml_words(ml) if MALAYALAM.match(w)))
        e_df.update(en_words(en))
    m_ok = {w for w, c in m_df.items() if c >= FREQ_MIN}
    e_ok = {w for w, c in e_df.items() if c >= FREQ_MIN}
    co = Counter()
    for en, ml in pairs:
        es = en_words(en) & e_ok
        if not es:
            continue
        for m in set(ml_words(ml)) & m_ok:
            for e in es:
                co[(m, e)] += 1
    best = {}
    for (m, e), c in sorted(co.items()):  # sorted: ties resolve the same way in every run
        if c < PAIR_MIN:
            continue
        p = c / m_df[m]
        if p >= P_E_GIVEN_M and c / e_df[e] >= P_M_GIVEN_E and p > best.get(m, ("", 0))[1]:
            best[m] = (e, p)
    return {m: e for m, (e, _) in best.items()}


def tokenise(en, ml, lex, dictionary):
    ews = set(EN_WORD.findall(en.lower()))
    toks = []
    for w in ml_words(ml):
        swap = dictionary.get(w, "")
        toks.append([roman_options(w, lex), swap if swap in ews else ""])
    return toks


# ---------------------------------------------------------------- build
def build(raw, dakshina, seed=SEED):
    in22 = read_in22(raw)
    banned = {key(x) for rows in in22.values() for pair in rows for x in pair}
    ml_test = PROC_DIR / "ours_ml_test.jsonl"
    if ml_test.exists():  # keep the script task's held-out Dakshina sentences out of training
        banned |= {key(norm_ml(r["native"])) for r in read_jsonl(ml_test)}
    stats = {"read": Counter(), "kept": Counter(), "dropped": Counter()}
    seen, pairs = set(), []
    for src, en, ml in read_sources(raw):
        stats["read"][src] += 1
        en, ml = norm_en(en), norm_ml(ml)
        if not MALAYALAM.search(ml) or not re.search(r"[A-Za-z]", en):
            stats["dropped"]["empty_or_wrong_script"] += 1
            continue
        if key(en) in banned or key(ml) in banned:
            stats["dropped"]["test_overlap"] += 1
            continue
        if len(ml_words(ml)) < MIN_WORDS or len(en) > MAX_EN:
            stats["dropped"]["too_short_or_long"] += 1
            continue
        k = (key(en), key(ml))
        if k in seen:
            stats["dropped"]["duplicate"] += 1
            continue
        seen.add(k)
        pairs.append((src, en, ml))
    lex = load_lexicon(dakshina)
    dictionary = mine_dictionary([(en, ml) for _, en, ml in pairs])
    rng = random.Random(f"{seed}-en-len")
    train = []
    for src, en, ml in pairs:
        toks = tokenise(en, ml, lex, dictionary)
        if len(render(toks, rng)) > MAX_ROMAN:
            stats["dropped"]["too_short_or_long"] += 1
            continue
        stats["kept"][src] += 1
        train.append({"ml": ml, "en": en, "source": src, "toks": toks})
    test = []
    for name, rows in in22.items():
        for i, (en, ml) in enumerate(rows):
            toks = tokenise(en, ml, lex, dictionary)
            r = random.Random(f"{seed}-{name}-{i}")
            roman = render(toks, r)
            mixed_rng = random.Random(f"{seed}-{name}-mix-{i}")
            mixed, swaps = [], []
            for w, (opts, swap) in zip(ml_words(ml), toks):
                if swap and mixed_rng.random() < TEST_SWAP_P:
                    mixed.append(swap)
                    swaps.append([w, swap])  # the Malayalam word a script output should contain
                else:
                    mixed.append(draw(opts, mixed_rng))
            test.append({"set": name, "ml": ml, "en": en, "roman": roman,
                         "noisy": corrupt(roman, random.Random(f"{seed}-{name}-noise-{i}")),
                         "mixed": " ".join(mixed), "swaps": swaps})
    n_tok = sum(len(r["toks"]) for r in train)
    stats.update(train_pairs=len(train), test_pairs=len(test), dictionary_size=len(dictionary),
                 lexicon_words=len(lex),
                 lexicon_coverage=round(sum(MALAYALAM.match(w) is not None and w in lex
                                            for r in train for w in ml_words(r["ml"])) / max(1, n_tok), 3),
                 swappable_pairs=sum(any(s for _, s in r["toks"]) for r in train),
                 test_with_swaps=sum(bool(r["swaps"]) for r in test))
    for k in ("read", "kept", "dropped"):
        stats[k] = dict(stats[k])
    return train, test, stats, dictionary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_DIR))
    ap.add_argument("--dakshina", default=str(RAW_DIR / "dakshina_dataset_v1.0.tar"))
    ap.add_argument("--test-only", action="store_true", help="rewrite only the test file (e.g. while training reads train)")
    a = ap.parse_args(argv)
    train, test, stats, dictionary = build(Path(a.raw), a.dakshina)
    write_rows(PROC_DIR / "ours_en_test.jsonl", test)
    if a.test_only:
        return stats
    write_rows(PROC_DIR / "ours_en_train.jsonl", train)
    (PROC_DIR / "ours_en_dictionary.json").write_text(json.dumps(dictionary, ensure_ascii=False, indent=0),
                                                       encoding="utf-8")
    (PROC_DIR / "ours_en.stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


if __name__ == "__main__":
    main()
