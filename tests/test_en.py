"""The English task: Malayalam normalisation and romanisation, the code-mixing dictionary, en:/code-mixed ml: examples,
and vocab compatibility for Tiny models saved before the task existed. CPU only, no downloads."""
import json
import random

ZWJ = chr(0x200D)


def test_norm_ml_unifies_chillu_and_nta_encodings():
    from en_data import norm_ml
    assert norm_ml("അവന്" + ZWJ) == "അവൻ"            # old ZWJ chillu -> atomic chillu
    assert norm_ml("എന്" + ZWJ + "റെ") == "എന്റെ"     # old ZWJ form of ന്റ
    assert norm_ml("എൻ്റെ") == "എന്റെ"                # Unicode 13 form of ന്റ
    assert norm_ml("  ഞാൻ   വരും ") == "ഞാൻ വരും"


def test_romanize_word_reads_like_manglish():
    from en_data import romanize_word
    cases = {"ഞാൻ": "njaan", "നാളെ": "naale", "വരും": "varum", "എന്റെ": "ente", "ഇന്ന്": "innu",
             "പറ്റില്ല": "pattilla", "എങ്ങനെ": "engane", "കുട്ടികൾ": "kuttikal", "വീട്": "veetu"}
    assert {w: romanize_word(w) for w in cases} == cases


def test_mine_dictionary_maps_inflected_forms_to_one_english_word():
    from en_data import mine_dictionary
    pairs = ([("I will come tomorrow", "ഞാൻ നാളെ വരും")] * 6 + [("tomorrow's meeting", "നാളത്തെ യോഗം")] * 5
             + [("see you tomorrow", "നാളത്തെ കാണാം")] * 4
             + [("I will come today", "ഞാൻ ഇന്ന് വരും")] * 9 + [("the meeting is today", "യോഗം ഇന്ന്")] * 9
             + [("rare words", "അപൂർവ്വം")] * 3)
    d = mine_dictionary(pairs)
    assert d.get("നാളത്തെ") == "tomorrow" and d.get("ഇന്ന്") == "today" and d.get("യോഗം") == "meeting"
    assert "അപൂർവ്വം" not in d  # seen fewer than FREQ_MIN times


def test_render_samples_spellings_and_swaps_english():
    from en_data import render, tokenise
    lex = {"ഞാൻ": [["njan", 3], ["njaan", 1]], "നാളെ": [["naale", 2], ["nale", 2]]}
    toks = tokenise("I will come tomorrow", "ഞാൻ നാളെ വരും", lex, {"നാളെ": "tomorrow"})
    assert toks[1][1] == "tomorrow" and toks[2][0] == [["varum", 1]]  # വരും: not in the lexicon -> rules
    rng = random.Random(0)
    plain = {render(toks, rng) for _ in range(50)}
    assert plain <= {"njan naale varum", "njan nale varum", "njaan naale varum", "njaan nale varum"} and len(plain) > 1
    assert render(toks, rng, swap_p=1.0).split()[1] == "tomorrow"
    # a dictionary word whose English equivalent is not in this pair's English is never swapped
    assert tokenise("I will come", "ഞാൻ നാളെ വരും", lex, {"നാളെ": "tomorrow"})[1][1] == ""


def test_build_examples_adds_en_and_code_mixed_ml():
    from en_data import tokenise
    from train_ours import build_examples, sample_rows
    row = {"ml": "ഞാൻ നാളെ വരും", "en": "I will come tomorrow",
           "toks": tokenise("I will come tomorrow", "ഞാൻ നാളെ വരും", {}, {"നാളെ": "tomorrow"})}
    ex = build_examples([], [], 3, seed=1, en_rows=[row] * 20, mlx_rows=[row] * 20)
    en = [e for e in ex if e["src"].startswith("en: ")]
    mlx = [e for e in ex if e["src"].startswith("ml: ")]
    assert len(en) == len(mlx) == 60
    assert {e["tgt"] for e in en} == {"I will come tomorrow"} and {e["tgt"] for e in mlx} == {"ഞാൻ നാളെ വരും"}
    assert any("tomorrow" in e["src"] for e in mlx) and any("tomorrow" not in e["src"] for e in en)
    assert ex == build_examples([], [], 3, seed=1, en_rows=[row] * 20, mlx_rows=[row] * 20)  # deterministic
    assert sample_rows(list(range(10)), 3, 1, "x") == sample_rows(list(range(10)), 3, 1, "x")
    assert sample_rows(list(range(10)), 0, 1, "x") == list(range(10))


def test_old_tiny_vocab_still_loads(tmp_path):
    from tiny_model import CharVocab
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps({"chars": ["a", "b"], "task_tokens": {"clean": 4, "ml": 5}, "n_special": 6}))
    old = CharVocab.load(p)
    assert old.tasks == ("clean", "ml") and len(old) == 8 and old.index["a"] == 6
    assert old.decode([6, 7, 2, 6]) == "ab"
    new = CharVocab.from_texts(["ab"])
    assert new.tasks == ("clean", "ml", "en") and new.encode_src("a", "en", 5)[0] == 6 and new.index["a"] == 7
    assert CharVocab.from_texts(["ab"], tasks=("clean", "ml")).tasks == ("clean", "ml")


def test_swap_rates_counts_translated_and_latin():
    from eval_en import swap_rates
    rows = [{"swaps": [["നാളെ", "tomorrow"]]}, {"swaps": [["മഴ", "rain"]]}, {"swaps": [["ഇന്ന്", "today"]]}]
    r = swap_rates(["ഞാൻ നാളെ വരും", "rain പെയ്യും", "ഞാൻ ടുഡേ"], rows)
    assert r == {"swaps": 3, "translated_pct": 33.3, "left_in_latin_pct": 33.3}
