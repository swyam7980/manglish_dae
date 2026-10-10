"""Unit tests for the multi-source data builder, memory fail-safes, new noise conditions and the length guard."""
import io
import random
import tarfile

import pytest
import torch


def test_build_extra_removes_val_overlap_and_duplicates():
    from extra_data import build_extra
    val = ["oru velutha christmas tree"]
    base = ["ente veedu evide aanu"]
    sources = {
        "a": ["Oru Velutha Christmas Tree", "oru-velutha christmas tree!", "nalla movie aanu", "nalla movie aanu", "ente veedu evide aanu"],
        "b": ["nalla  movie aanu", "vere sentence und", "x"],
    }
    rows, stats = build_extra(sources, val, base)
    got = [s for s, _ in rows]
    assert got == ["nalla movie aanu", "vere sentence und"]          # val look-alikes, in-source and cross-source dups, too short: gone
    assert stats["a"]["dropped_val_overlap"] == 2
    assert stats["a"]["added"] == 1 and stats["b"]["added"] == 1
    assert dict((s, src) for s, src in rows)["vere sentence und"] == "b"


def test_build_extra_cap_is_deterministic():
    from extra_data import build_extra
    src = {"a": [f"sentence number {chr(97 + i)}{chr(97 + j)} aanu" for i in range(10) for j in range(10)]}
    r1, _ = build_extra(src, [], [], cap_per_source=20)
    r2, _ = build_extra(src, [], [], cap_per_source=20)
    assert len(r1) == 20 and r1 == r2


def test_dakshina_loader_dir_and_tar(tmp_path):
    from extra_data import load_dakshina
    rel = "dakshina_dataset_v1.0/ml/romanized/ml.romanized.rejoined.dev.roman.txt"
    other = "dakshina_dataset_v1.0/ta/romanized/ta.romanized.rejoined.dev.roman.txt"
    (tmp_path / "d" / "dakshina_dataset_v1.0" / "ml" / "romanized").mkdir(parents=True)
    (tmp_path / "d" / "dakshina_dataset_v1.0" / "ta" / "romanized").mkdir(parents=True)
    (tmp_path / "d" / rel).write_text("malayalam oru\nmalayalam randu\n", encoding="utf-8")
    (tmp_path / "d" / other).write_text("tamil onru\n", encoding="utf-8")
    assert load_dakshina(tmp_path / "d") == ["malayalam oru", "malayalam randu"]
    tar_path = tmp_path / "dak.tar"
    with tarfile.open(tar_path, "w") as tf:
        for name, text in ((rel, "malayalam oru\nmalayalam randu\n"), (other, "tamil onru\n")):
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    assert load_dakshina(tar_path) == ["malayalam oru", "malayalam randu"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        load_dakshina(empty)                                            # no ml/romanized files


def test_noise_conditions_are_deterministic_and_different():
    import noise
    sents = ["thaazhvarayile manju malakal", "nalla velutha christmas tree"] * 3
    for name in noise.CONDITIONS:
        if name == "base":
            continue
        a, b = noise.apply_condition(sents, name, 11), noise.apply_condition(sents, name, 11)
        assert a == b and a != sents
    assert noise.drop_vowels("nalla", random.Random(0), 1.0) == "nll"   # first letter is kept, vowels removed
    with pytest.raises(ValueError):
        noise.apply_condition(sents, "base", 1)


def test_pick_micro_batch_prefers_largest_that_fits():
    import memory

    def tries(limit):
        def fn(bs):
            if bs > limit:
                raise torch.cuda.OutOfMemoryError("out of memory")
        return fn
    assert memory.pick_micro_batch(tries(8), 16, None, lambda: 0) == 8
    assert memory.pick_micro_batch(tries(3), 16, None, lambda: 0) == 2
    assert memory.pick_micro_batch(tries(0), 16, None, lambda: 0) == 1
    # fits without error but peak is over the VRAM budget -> must go smaller
    peaks = {16: 9e9, 8: 7e9, 4: 4e9, 2: 2e9, 1: 1e9}
    chosen = {}

    def fn(bs):
        chosen["last"] = bs
    assert memory.pick_micro_batch(fn, 16, 5e9, lambda: peaks[chosen["last"]]) == 4
    assert memory.next_smaller(16) == 8 and memory.next_smaller(2) == 1 and memory.next_smaller(1) is None


def test_run_with_backoff_halves_on_oom():
    import memory
    sizes = []

    def fn(chunk):
        sizes.append(len(chunk))
        if len(chunk) > 4:
            raise torch.cuda.OutOfMemoryError("out of memory")
        return [x * 2 for x in chunk]
    out, bs = memory.run_with_backoff(fn, list(range(20)), 16)
    assert out == [x * 2 for x in range(20)] and bs == 4
    assert sizes[0] == 16 and 8 in sizes


def test_length_guard_and_collapse():
    import generation
    assert generation.collapse_runs("coooool boyyyy ok") == "cool boyy ok"
    noisy = ["wine glaaaass kayyyyyil piddicchirrrrikkunna old man", "oru velutha dog"]
    preds = ["old man", "oru velutha dog"]
    out, n = generation.length_guard(preds, noisy)
    assert n == 1 and out[0] == generation.collapse_runs(noisy[0]) and out[1] == preds[1]
