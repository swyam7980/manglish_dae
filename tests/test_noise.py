import random

from noise import corrupt, corrupt_all

SENTS = ["njan nale varum", "ente ammaye ariyo", "adipoli padam machane", "povilla mone"] * 50


def test_deterministic():
    assert corrupt_all(SENTS, 1) == corrupt_all(SENTS, 1)


def test_seeds_differ():
    assert corrupt_all(SENTS, 1) != corrupt_all(SENTS, 2)


def test_noisy_differs_but_is_recognisable():
    noisy = corrupt_all(SENTS, 1)
    changed = sum(n != c for n, c in zip(noisy, SENTS)) / len(SENTS)
    assert changed > 0.5
    # collapsing runs of repeated letters should land close to the original
    import re
    close = sum(re.sub(r"(.)\1+", r"\1", n)[:3] == re.sub(r"(.)\1+", r"\1", c)[:3] for n, c in zip(noisy, SENTS))
    assert close / len(SENTS) > 0.6


def test_non_alpha_untouched_and_never_empty():
    rng = random.Random(0)
    assert corrupt("123 !!", rng) == "123 !!"
    for i in range(500):
        assert corrupt("a", random.Random(i), p_rep=0.0, p_dd=1.0).strip()


def test_repetition_only_adds_1_to_3_copies():
    out = corrupt("a", random.Random(0), p_rep=1.0, p_dd=0.0)
    assert set(out) == {"a"} and 2 <= len(out) <= 4
