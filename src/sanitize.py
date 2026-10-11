"""Sanitisation for extra data sources: turn raw lines into clean, lower-case, Latin-script reconstruction targets.

normalize() repairs what can be repaired (Unicode forms, HTML entities, control and zero-width characters, curly
quotes, emoji, URLs, mentions, bracketed references, repeated punctuation, whitespace). reject_reason() then names
the first rule a sentence breaks, so every dropped line is counted by reason and a few examples are kept for
inspection. Over-long lines can be split into clauses first.

The original Manglish class is not re-filtered with these rules, so the validation set stays exactly what earlier
results were scored on; audit() reports how many of its sentences these rules would have dropped.
"""
import html
import re
import unicodedata
from collections import Counter

from config import MIN_CHARS, MAX_CHARS

WHITESPACE_CONTROLS = frozenset(chr(c) for c in (9, 10, 11, 12, 13))
MIN_WORDS = 2
MAX_WORD_LEN = 40   # Malayalam is agglutinative: long romanised words are normal
MIN_LETTER_SHARE = 0.60
MAX_DIGIT_SHARE = 0.30
MAX_INITIAL_SHARE = 0.25
MIN_UNIQUE_TOKEN_SHARE = 0.5
NO_VOWEL_LEN = 6
EXAMPLES_PER_REASON = 5

_QUOTES = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"',
           "–": "-", "—": "-", "−": "-", "…": "...", " ": " "}
_URL = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+|@\w+|#\w+")
_BRACKET_REF = re.compile(r"\[\s*\d+\s*\]|\(\s*\d+\s*\)")
_RUN_PUNCT = re.compile(r"([^\w\s])\1+")
_ELONGATED = re.compile(r"([^\W\d_])\1{2,}")
_VOWEL_FREE = re.compile(r"\b[^aeiou\W\d_]{%d,}\b" % NO_VOWEL_LEN)
_CLAUSE_END = re.compile(r"(?<=[,;:.!?])\s+")


def normalize(s):
    """Repair a raw line. Returns the lower-cased text, possibly empty."""
    s = html.unescape(str(s))
    s = unicodedata.normalize("NFKC", s)
    s = "".join(_QUOTES.get(c, c) for c in s)
    s = "".join(" " if c in WHITESPACE_CONTROLS else c for c in s)  # these separate words; other controls are deleted below
    s = "".join(c for c in s if unicodedata.category(c) not in ("Cc", "Cf", "Co", "Cs", "So", "Sk"))
    s = s.replace('"', " ")
    s = _URL.sub(" ", s)
    s = _BRACKET_REF.sub(" ", s)
    s = _RUN_PUNCT.sub(r"\1", s)
    s = re.sub(r"\s+", " ", s).strip(" -_*~|'")
    return s.lower()


def reject_reason(s, max_chars=MAX_CHARS):
    """First rule broken by an already-normalised sentence, or None when it is acceptable."""
    if not s:
        return "empty"
    if len(s) < MIN_CHARS:
        return "too_short"
    if not re.search(r"[a-z]", s):
        return "no_letters"
    if len(s) > max_chars:
        return "too_long"
    if not s.isascii():
        return "non_latin_script"
    tokens = re.findall(r"[a-z0-9']+", s)
    letters = [c for c in s if c.isalpha()]
    non_space = [c for c in s if not c.isspace()]
    digits = sum(c.isdigit() for c in s)
    if len(re.findall(r"[a-z]+", s)) < MIN_WORDS:
        return "too_few_words"
    if (len(letters) + digits) / len(non_space) < MIN_LETTER_SHARE:
        return "symbol_heavy"
    if digits / max(1, digits + len(letters)) > MAX_DIGIT_SHARE:
        return "digit_heavy"
    if _ELONGATED.search(s):
        return "elongated_letters"
    if any(len(t) > MAX_WORD_LEN for t in tokens):
        return "gibberish_token"
    if _VOWEL_FREE.search(s):
        return "gibberish_token"
    if tokens and len(tokens[-1]) == 1 and tokens[-1].isalpha():
        return "trailing_initial"  # sentence splitters cut Wikipedia text after initials, leaving a truncated fragment
    if sum(len(t) == 1 and t.isalpha() for t in tokens) / max(1, len(tokens)) > MAX_INITIAL_SHARE:
        return "initials_heavy"
    if len(tokens) >= 6 and len(set(tokens)) / len(tokens) < MIN_UNIQUE_TOKEN_SHARE:
        return "repetitive"
    return None


def split_clauses(s, max_chars=MAX_CHARS):
    """Cut an over-long sentence at clause boundaries and re-join neighbouring clauses up to max_chars."""
    parts = [p.strip() for p in _CLAUSE_END.split(s) if p.strip()]
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + 1 + len(p) > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        out.append(cur)
    return out


def tidy(s):
    """Trailing and leading punctuation that survives clause splitting carries no meaning for the target."""
    return re.sub(r"\s+", " ", s.strip(" ,;:.!?-'")).strip()


def sanitize_many(raw, split_long=False, max_chars=MAX_CHARS):
    """raw: iterable of lines. Returns (unique clean sentences in input order, stats).
    stats: input, pieces after splitting, kept, unique, rejected counts by reason, up to 5 examples per reason."""
    reasons, examples = Counter(), {}
    kept, seen, n_in, n_pieces = [], set(), 0, 0
    for line in raw:
        n_in += 1
        text = normalize(line)
        pieces = split_clauses(text, max_chars) if (split_long and len(text) > max_chars) else [text]
        for piece in pieces:
            n_pieces += 1
            piece = tidy(piece)
            why = reject_reason(piece, max_chars)
            if why:
                reasons[why] += 1
                bucket = examples.setdefault(why, [])
                if len(bucket) < EXAMPLES_PER_REASON:
                    bucket.append(piece[:100] or str(line)[:100])
            elif piece in seen:
                reasons["duplicate"] += 1
            else:
                seen.add(piece)
                kept.append(piece)
    stats = {"input_lines": n_in, "pieces": n_pieces, "kept_unique": len(kept),
             "rejected_by_reason": dict(reasons.most_common()), "examples": examples}
    return kept, stats


def audit(sentences, max_chars=MAX_CHARS):
    """How many already-clean sentences would these rules drop? Used on the original Manglish class (not changed)."""
    reasons = Counter()
    for s in sentences:
        why = reject_reason(s, max_chars)
        if why:
            reasons[why] += 1
    return {"sentences": len(sentences), "would_drop": sum(reasons.values()), "by_reason": dict(reasons.most_common())}
