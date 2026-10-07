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
