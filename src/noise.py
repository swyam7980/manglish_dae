"""Synthetic Manglish corruption: character repetition plus random duplication/deletion.
The robustness conditions at the bottom add heavier noise and two noise types the models never trained on."""
import random
import re

from config import P_REPEAT, P_DUPDEL


def corrupt(text: str, rng: random.Random, p_rep: float = P_REPEAT, p_dd: float = P_DUPDEL) -> str:
    """Repetition (p_rep): alphabetic char followed by 1-3 extra copies.
    Duplication/deletion (p_dd): alphabetic char doubled or dropped with equal chance."""
    out = []
    for ch in text:
        if not ch.isalpha():
            out.append(ch)
            continue
        r = rng.random()
        if r < p_rep:
            out.append(ch * (1 + rng.randint(1, 3)))
        elif r < p_rep + p_dd:
            if rng.random() < 0.5:
                out.append(ch * 2)
            # else: deletion, append nothing
        else:
            out.append(ch)
    noisy = "".join(out)
    return noisy if noisy.strip() else text


def corrupt_all(sentences, seed: int):
    """Deterministic per-sentence noise (seed + index), so every backbone sees identical corrupted data."""
    return [corrupt(s, random.Random(seed * 1_000_003 + i)) for i, s in enumerate(sentences)]


VOWELS = "aeiou"
# (pattern, replacement) spelling variants seen in romanised Malayalam; applied per occurrence with probability p
PHONETIC_RULES = [("aa", "a"), ("ee", "i"), ("oo", "u"), ("ii", "i"), ("th", "t"), ("zh", "l"), ("sh", "s"),
                  ("ch", "c"), ("kk", "k"), ("nn", "n"), ("ll", "l"), ("u", "oo"), ("i", "ee"), ("a", "aa")]


def drop_vowels(text: str, rng: random.Random, p: float = 0.5) -> str:
    """Remove each non-initial vowel of a word with probability p (texting style: 'nalla' -> 'nll')."""
    out = []
    for word in re.split(r"(\s+)", text):
        kept = [ch for i, ch in enumerate(word) if not (i > 0 and ch in VOWELS and rng.random() < p)]
        out.append("".join(kept))
    res = "".join(out)
    return res if res.strip() else text


def phonetic_variants(text: str, rng: random.Random, p: float = 0.5) -> str:
    """Rewrite common letter groups the way different writers romanise them ('thaazhvara' -> 'tazhvara')."""
    for pat, rep in PHONETIC_RULES:
        text = re.sub(pat, lambda m: rep if rng.random() < p else m.group(0), text)
    return text


CONDITIONS = {
    "base": None,
    "heavy": lambda t, r: corrupt(t, r, 0.20, 0.08),
    "extreme": lambda t, r: corrupt(t, r, 0.30, 0.12),
    "vowel_drop": lambda t, r: drop_vowels(t, r, 0.5),
    "phonetic": lambda t, r: phonetic_variants(t, r, 0.5),
    "phonetic_plus_base": lambda t, r: corrupt(phonetic_variants(t, r, 0.5), r),
}


def apply_condition(sentences, name: str, seed: int):
    """Deterministic noisy version of each sentence under a named robustness condition."""
    fn = CONDITIONS[name]
    if fn is None:
        raise ValueError("'base' uses the stored validation noise, not a generated one")
    return [fn(s, random.Random(seed * 1_000_003 + i)) or s for i, s in enumerate(sentences)]


if __name__ == "__main__":
    for s in ["povilla", "njan nale varum", "ente ammaye ariyo"]:
        print(s, "->", corrupt(s, random.Random(1)))
