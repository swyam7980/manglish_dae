"""Tests for the sanitisation pipeline used on extra data sources."""
import pytest

import sanitize


@pytest.mark.parametrize("raw,expected", [
    ("  Oru   VELUTHA\tChristmas   Tree  ", "oru velutha christmas tree"),
    ("nalla &amp; velutha movie", "nalla & velutha movie"),
    ("\u201cnalla\u201d movie\u2014super", "nalla movie-super"),
    ("line one" + chr(10) + "line" + chr(9) + "two" + chr(13) + chr(10) + "end", "line one line two end"),
    ("zero\u200bwidth\u00a0space\u0007bell", "zerowidth spacebell"),
    ("super movie \U0001F600\U0001F44D aanu", "super movie aanu"),
    ("see https://x.com/a?b=1 and www.site.org now @user #tag ok", "see and now ok"),
    ("ref one [1] and (23) end", "ref one and end"),
    ("what!!! really??? ...", "what! really? ."),
    ("\ufb01nal \uff21\uff22\uff23", "final abc"),     # NFKC: ligature and full-width forms
])
def test_normalize_repairs(raw, expected):
    assert sanitize.normalize(raw) == expected


@pytest.mark.parametrize("sentence,reason", [
    ("", "empty"), ("ab", "too_short"), ("123 456", "no_letters"),
    ("word " * 40, "too_long"), ("oru caf\u00e9 aanu", "non_latin_script"),
    ("\u0d2e\u0d32\u0d2f\u0d3e\u0d33\u0d02 text", "non_latin_script"),
    ("lavender", "too_few_words"), ("a1 b2 c3 !!! ??? ###", "symbol_heavy"),
    ("1234 5678 9012 ab cd", "digit_heavy"), ("sooooper movie aanu", "elongated_letters"),
    ("ab" * 21 + " aanu", "gibberish_token"), ("this strngthsss aanu", "elongated_letters"),
    ("ente brthrsss aanu", "elongated_letters"), ("kshthrmpl ente veedu", "gibberish_token"),
    ("a b c d e f g", "trailing_initial"), ("aalappuzha jillayile nedumudiyil p", "trailing_initial"),
    ("n m k nalla veedu evide", "initials_heavy"),
    ("aanu aanu aanu aanu aanu ente", "repetitive"),
])
def test_reject_reasons(sentence, reason):
    assert sanitize.reject_reason(sentence) == reason


@pytest.mark.parametrize("sentence", [
    "oru velutha christmas tree", "sun soaking cheyyunna person", "7 elephants kunnin cheruviloode nadakkunnu",
    "jersey dharicha women's table tennis player", "pattyavishayamakkiyittundathre ente veedu",
])
def test_good_sentences_pass(sentence):
    assert sanitize.reject_reason(sentence) is None


def test_split_clauses_respects_limit_and_loses_nothing():
    text = ", ".join(f"clause number {i} aanu ithu" for i in range(12))
    parts = sanitize.split_clauses(text, 60)
    assert len(parts) > 1 and all(len(p) <= 60 for p in parts)
    assert " ".join(parts).replace(" ", "") == text.replace(" ", "")


def test_sanitize_many_counts_every_line_and_keeps_examples():
    raw = ["Oru velutha tree.", "oru velutha tree", "lavender", "sooooper movie aanu", "ok cheyyunna person!!", "\U0001F600"]
    kept, st = sanitize.sanitize_many(raw)
    assert kept == ["oru velutha tree", "ok cheyyunna person"]
    r = st["rejected_by_reason"]
    assert r == {"too_few_words": 1, "elongated_letters": 1, "duplicate": 1, "empty": 1} or sum(r.values()) == 4
    assert st["input_lines"] == 6 and st["kept_unique"] == 2
    assert st["examples"]["elongated_letters"] == ["sooooper movie aanu"]


def test_sanitize_many_splits_long_lines_only_when_asked():
    words = "kadal vayal mala puzha veedu kada school paalam road theru".split()
    long_line = ", ".join(f"oru {w} nalla {v} kaanunnu" for w, v in zip(words, reversed(words)))
    kept_no, st_no = sanitize.sanitize_many([long_line])
    kept_yes, _ = sanitize.sanitize_many([long_line], split_long=True)
    assert kept_no == [] and st_no["rejected_by_reason"]["too_long"] == 1
    assert len(kept_yes) >= 2 and all(len(k) <= 120 for k in kept_yes)


def test_audit_reports_without_changing_anything():
    out = sanitize.audit(["oru velutha tree", "sooooper movie aanu"])
    assert out == {"sentences": 2, "would_drop": 1, "by_reason": {"elongated_letters": 1}}
