"""Our own models: script pair building, the noise mix, Manglish-Tiny (train, ONNX export parity, inference) and the
Manglish-ByT5 training/merge path on a tiny random ByT5. CPU only, no downloads."""
import json
import random

import pytest
import torch

ML_ROWS = [("njan nale varum", "ഞാൻ നാളെ വരും"), ("ente veedu", "എന്റെ വീട്"), ("kerala sarkkar", "കേരള സർക്കാർ"),
           ("nalla mazha", "നല്ല മഴ"), ("avan vannu", "അവൻ വന്നു"), ("ithu nalla pusthakam", "ഇത് നല്ല പുസ്തകം")]
CLEAN = ["njan nale varum", "ente amma veettil undu", "beach sidil jogging cheyyunna aalukal", "valare nalla padam",
         "avan schoolil poyi", "mazha peyyunnu"] * 4


@pytest.fixture(scope="module")
def data():
    from config import PROC_DIR
    from extra_data import write_rows
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    write_rows(PROC_DIR / "train_plus_all.jsonl", [{"clean": c, "noisy": c, "source": "s"} for c in CLEAN])
    write_rows(PROC_DIR / "ours_ml_train.jsonl", [{"roman": r, "native": n, "kind": "sentence"} for r, n in ML_ROWS])
    return PROC_DIR


def test_sentence_map_uses_the_training_sanitiser():
    from ours_data import sentence_map
    files = {"ml.romanized.rejoined.dev.roman.txt": "Njan Nale Varum.\nx\n",
             "ml.romanized.rejoined.dev.native.txt": "ഞാൻ നാളെ വരും.\nഎ\n",
             "ml.romanized.rejoined.test.roman.txt": "", "ml.romanized.rejoined.test.native.txt": ""}
    mapping, stats = sentence_map(files)
    assert mapping == {"njan nale varum": "ഞാൻ നാളെ വരും"}  # lower-cased, outer punctuation gone on both sides
    assert stats["aligned_lines"] == 2 and stats["usable_lines"] == 1


def test_lexicon_keeps_most_attested_spelling_and_skips_test_words():
    from ours_data import lexicon_pairs
    text = "അവൻ\tavan\t3\nഅവൻ\tavann\t1\nമഴ\tmazha\t2\nവീട്\tveedu\t2\n"
    pairs = lexicon_pairs(text, banned_roman={"veedu"}, cap=0)
    assert sorted(pairs) == [("avan", "അവൻ"), ("mazha", "മഴ")]


def test_build_examples_fresh_noise_per_epoch_and_deterministic():
    from train_ours import build_examples, TASK_PREFIX
    ml = [{"roman": r, "native": n} for r, n in ML_ROWS]
    a = build_examples(CLEAN, ml, 2, seed=1)
    assert a == build_examples(CLEAN, ml, 2, seed=1)
    assert len(a) == 2 * (len(CLEAN) + len(ml))
    assert {e["src"].split(":")[0] + ": " for e in a} == {TASK_PREFIX["clean"], TASK_PREFIX["ml"]}  # no en rows given
    first, second = a[: len(CLEAN)], a[len(CLEAN) + len(ml): 2 * len(CLEAN) + len(ml)]
    assert [e["tgt"] for e in first] == [e["tgt"] for e in second]
    assert [e["src"] for e in first] != [e["src"] for e in second]
    assert build_examples(CLEAN, ml, 1, seed=1, first_epoch=1) == a[len(CLEAN) + len(ml):]


def test_is_oom_matches_driver_out_of_memory():
    from memory import is_oom
    assert is_oom(RuntimeError("CUDA error: out of memory"))
    assert not is_oom(RuntimeError("shape mismatch"))


def test_tiny_model_shapes_and_vocab_roundtrip():
    from tiny_model import CharVocab, TinySeq2Seq, pad_batch, EOS
    v = CharVocab.from_texts(["abc ", "ഞാൻ"])
    ids = v.encode_tgt("ab ഞാ", 50)
    assert ids[-1] == EOS and v.decode(ids) == "ab ഞാ"
    m = TinySeq2Seq(len(v), d_model=32, heads=2, enc_layers=1, dec_layers=1, ff=64).eval()
    src = pad_batch([v.encode_src("abc", "clean", 50), v.encode_src("a", "ml", 50)])
    assert m(src, torch.ones(2, 5, dtype=torch.long)).shape == (2, 5, len(v))
    assert m.greedy(src, 7).shape[0] == 2


def test_padding_does_not_change_tiny_outputs():
    from tiny_model import CharVocab, TinySeq2Seq, pad_batch
    torch.manual_seed(0)
    v = CharVocab.from_texts(["abcdef "])
    m = TinySeq2Seq(len(v), d_model=32, heads=2, enc_layers=1, dec_layers=1, ff=64).eval()
    short = v.encode_src("ab", "clean", 50)
    alone = m.greedy(pad_batch([short]), 6)[0].tolist()
    batched = m.greedy(pad_batch([short, v.encode_src("abcdef abc", "clean", 50)]), 6)[0].tolist()
    assert alone == batched[: len(alone)]


ARCH = ["--d-model", "32", "--heads", "2", "--enc-layers", "1", "--dec-layers", "1", "--ff", "64"]


@pytest.fixture(scope="module")
def tiny_trained(data):
    import train_tiny
    info = train_tiny.main(["--epochs", "2", "--batch", "8", "--device", "cpu", "--tag", "ours_tiny_t", *ARCH])
    return info


def test_tiny_trains_and_keeps_best_checkpoint(tiny_trained):
    from config import OUT_DIR
    assert len(tiny_trained["history"]) == 2
    assert all(h["dev_loss"] == h["dev_loss"] for h in tiny_trained["history"])
    for f in ("model.pt", "vocab.json", "config.json", "train_info.json"):
        assert (OUT_DIR / "ours_tiny_t" / f).exists()


def test_tiny_onnx_export_matches_pytorch(tiny_trained):
    pytest.importorskip("onnxruntime")
    import export_tiny
    rep = export_tiny.main(["--tag", "ours_tiny_t"])
    assert all(r["same"] for r in rep["parity_fp32"])
    assert set(rep["sizes_mb"]) == {"encoder.onnx", "decoder.onnx", "encoder.int8.onnx", "decoder.int8.onnx"}


def test_translator_tiny_both_tasks(tiny_trained):
    from config import OUT_DIR
    from translate import Translator
    tr = Translator("tiny", OUT_DIR / "ours_tiny_t", device="cpu")
    for task in ("clean", "ml"):
        out = tr.translate(["Njan Nale Varum", "mazha"], task)
        assert len(out) == 2 and all(isinstance(s, str) for s in out)


def test_ours_byt5_trains_merges_and_reloads(data, tmp_path):
    import tiny_models
    import train_ours
    from config import OUT_DIR
    from translate import Translator
    from noise import corrupt
    corpus = [corrupt(c, random.Random(i)) for i, c in enumerate(CLEAN * 10)]  # enough text for the toy tokenizers
    paths = tiny_models.build(tmp_path, corpus + [n for _, n in ML_ROWS])
    info = train_ours.main(["--base-path", paths["byt5"], "--max-steps", "2", "--epochs", "1", "--tag", "ours_byt5_t"])
    assert info["tasks"] == ["clean", "ml"] and info["examples"] == len(CLEAN) + len(ML_ROWS)
    model_dir = OUT_DIR / "ours_byt5_t" / "model"
    cfg = json.loads((model_dir / "config.json").read_text())
    assert "peft_type" not in cfg  # merged: an ordinary model folder
    tr = Translator("byt5", model_dir, device="cpu")
    out = tr.translate(["njaaan naaale varummm"], "clean")
    assert len(out) == 1  # random weights: only the wiring is checked; the length guard keeps it non-empty
    assert out[0]
