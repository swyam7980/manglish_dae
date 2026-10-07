import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


def run_prep(work, *args):
    env = {**os.environ, "MDAE_WORKDIR": str(work)}
    return subprocess.run([sys.executable, str(SRC / "data_prep.py"), *args], capture_output=True, text=True, env=env, cwd=SRC)


def rows(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def test_demo_prep_properties(tmp_path):
    r = run_prep(tmp_path, "--demo")
    assert r.returncode == 0, r.stderr
    tr, va = rows(tmp_path / "data/processed/train.jsonl"), rows(tmp_path / "data/processed/val.jsonl")
    assert abs(len(va) / (len(tr) + len(va)) - 0.10) < 0.01
    assert not {x["clean"] for x in tr} & {x["clean"] for x in va}
    assert all(x["clean"] == x["clean"].lower() for x in tr + va)
    assert sum(x["clean"] != x["noisy"] for x in va) / len(va) > 0.5
    assert "english" not in " ".join(x["clean"] for x in tr + va)  # other classes dropped


def test_rerun_is_byte_identical(tmp_path):
    run_prep(tmp_path, "--demo")
    a = [(tmp_path / f"data/processed/{n}.jsonl").read_bytes() for n in ("train", "val")]
    run_prep(tmp_path, "--demo")
    b = [(tmp_path / f"data/processed/{n}.jsonl").read_bytes() for n in ("train", "val")]
    assert a == b


def test_cleaning_filters():
    from data_prep import build_sentences
    raw = ["Njan  NALE varum http://x.co", "njan nale varum", "ab", "", "ente @user veedu", "നമസ്കാരം നമസ്കാരം",
           "x" * 200, "!!!"]
    out, stats = build_sentences(raw)
    assert out == ["njan nale varum", "ente veedu"]
    assert stats["after_dedupe"] == 2


def test_missing_data_and_bad_label_messages(tmp_path):
    r = run_prep(tmp_path)
    assert r.returncode != 0 and "No data file" in r.stderr + r.stdout
    run_prep(tmp_path, "--demo")
    r = run_prep(tmp_path, "--file", str(tmp_path / "data/raw/demo_sellai_like.csv"), "--label", "nonexistent")
    assert r.returncode != 0 and "class counts" in r.stdout + r.stderr
