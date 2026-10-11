"""End-to-end wiring test for all three backbones on tiny random models (no downloads, CPU, ~1 minute)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    work = Path(os.environ["MDAE_WORKDIR"])
    r = subprocess.run([sys.executable, str(SRC / "data_prep.py"), "--demo"], capture_output=True, text=True, cwd=SRC,
                       env={**os.environ, "MDAE_WORKDIR": str(work)})
    assert r.returncode == 0, r.stderr
    from config import PROC_DIR
    texts = []
    for split in ("train", "val"):
        for line in open(PROC_DIR / f"{split}.jsonl", encoding="utf-8"):
            d = json.loads(line)
            texts += [d["clean"], d["noisy"]]
    import tiny_models
    root = tmp_path_factory.mktemp("tiny")
    return tiny_models.build(root, texts)


@pytest.mark.parametrize("key", ["mbart", "mt5", "byt5"])
def test_train_save_reload_evaluate(prepared, key):
    import train
    from evaluate import evaluate
    from config import OUT_DIR, RES_DIR
    info = train.main(["--model", key, "--size", "tiny", "--base-path", prepared[key], "--max-train", "64",
                       "--max-steps", "4", "--tag-suffix", "_t"])
    assert 0 < info["trainable_pct"] < 60          # toy models: LoRA is a big share; the real <5% check is in smoke_test.py
    assert all(map(lambda x: x == x and abs(x) < 1e6, info["loss_history"]))
    ad = OUT_DIR / (("mbart_large" if key == "mbart" else f"{key}_tiny") + "_t") / "adapter"
    assert (ad / "adapter_config.json").exists()
    assert not list(ad.glob("pytorch_model*")) and not list(ad.glob("model*.bin"))  # adapter only, no base weights
    r1 = evaluate(key, "tiny", max_val=16, base_path=prepared[key], tag_suffix="_t")
    r2 = evaluate(key, "tiny", max_val=16, base_path=prepared[key], tag_suffix="_t")
    assert r1 == r2                                  # greedy decoding is deterministic
    assert (RES_DIR / f"preds_{'mbart_large' if key == 'mbart' else key + '_tiny'}_t.csv").exists()


def test_baseline_compare_analyze(prepared):
    from evaluate import baseline
    import compare, analyze
    res = baseline()
    assert 0 < res["CER"] < 60                       # noise at p=0.10/0.04 gives a clearly non-zero but modest CER
    compare.main()
    analyze.main()
    from config import RES_DIR
    assert (RES_DIR / "summary.csv").exists() and (RES_DIR / "comparison.png").exists()


def test_train_oom_backoff_and_extra_train_file(prepared, monkeypatch):
    """A first out-of-memory error must halve the micro-batch, keep the effective batch and still finish."""
    import torch
    import train
    from config import PROC_DIR, OUT_DIR, EFFECTIVE_BATCH
    rows = [json.loads(line) for line in open(PROC_DIR / "train.jsonl", encoding="utf-8")][:48]
    with open(PROC_DIR / "train_plus_unit.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, "source": "unit"}) + "\n")
    real, calls = train.Seq2SeqTrainer.train, {"n": 0}

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.cuda.OutOfMemoryError("out of memory")
        return real(self, *a, **k)
    monkeypatch.setattr(train.Seq2SeqTrainer, "train", flaky)
    info = train.main(["--model", "mt5", "--size", "tiny", "--base-path", prepared["mt5"], "--train-file",
                       "train_plus_unit.jsonl", "--max-steps", "3", "--auto-batch", "--resume", "--save-steps", "2",
                       "--tag-suffix", "_oom"])
    from memory import next_smaller
    assert len(info["oom_retries"]) == 1                                # the injected OOM, handled once
    first = info["oom_retries"][0]
    assert first["to"] == next_smaller(first["from"]) == info["micro_batch"]   # 16 -> 8 with CUDA (auto-batch), 2 -> 1 on CPU
    assert info["micro_batch"] * info["grad_accum"] == EFFECTIVE_BATCH
    assert info["train_rows"] == 48 and info["train_file"] == "train_plus_unit.jsonl"
    assert (OUT_DIR / "mt5_tiny_oom" / "adapter" / "adapter_config.json").exists()
    assert not (OUT_DIR / "mt5_tiny_oom" / "ckpt").exists()             # checkpoints are cleaned up after a finished run


def test_studies_run_end_to_end_on_tiny_models(prepared):
    """eval_combined, decode_study and robustness must run, write their files and agree with plain evaluation."""
    import pandas as pd
    import decode_study, eval_combined, robustness, train
    from config import PROC_DIR, RES_DIR
    from evaluate import evaluate
    train.main(["--model", "mt5", "--size", "tiny", "--base-path", prepared["mt5"], "--max-train", "64",
                "--max-steps", "3", "--tag-suffix", "_st"])
    rows = [json.loads(line) for line in open(PROC_DIR / "val.jsonl", encoding="utf-8")][:40]
    with open(PROC_DIR / "test_unit.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({**r, "source": "src_a" if i % 2 else "src_b"}) + "\n")
    common = ["--systems", "mt5:_st", "--size", "tiny", "--base-path", prepared["mt5"], "--eval-file", "test_unit.jsonl"]
    det = eval_combined.main([*common, "--out", "unit_combined"])
    assert set(det.source) == {"ALL", "src_a", "src_b"} and {"copy_input", "rule_collapse_runs", "mt5_st"} <= set(det.system)
    assert (RES_DIR / "unit_combined" / "combined_summary.json").exists()
    mt5_all = det[(det.system == "mt5_st") & (det.source == "ALL")].iloc[0]
    assert mt5_all.n == 40
    plain = evaluate("mt5", "tiny", max_val=40, base_path=prepared["mt5"], tag_suffix="_st")
    assert abs(mt5_all.CER - plain["CER"]) < 1e-9                         # same adapter, same rows, same greedy decoding
    dec = decode_study.main([*common, "--max-eval", "20", "--out", "unit_decoding"])
    assert {"beam1", "beam1+guard", "beam4", "beam4+guard", "rule"} <= set(dec.config)
    rob = robustness.main([*common, "--max-eval", "20", "--out", "unit_robust"])
    assert {"base", "heavy", "extreme", "vowel_drop", "phonetic", "phonetic_plus_base"} == set(rob.condition)
    copy_cer = rob[rob.system == "copy_input"].set_index("condition").CER
    assert copy_cer["extreme"] > copy_cer["heavy"] > copy_cer["base"]    # heavier noise really is heavier
