"""Single entry point: runs the whole pipeline in order. Usage: python run_all.py [--size small|base] [--smoke]"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(*args):
    cmd = [sys.executable, str(HERE / args[0]), *args[1:]]
    print("\n>>>", " ".join(Path(c).name if i == 1 else c for i, c in enumerate(cmd[1:], 1)), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base", help="mT5/ByT5 size tier (use the same for both)")
    ap.add_argument("--file", help="SeLLaI data file (default: first file in data/raw)")
    ap.add_argument("--models", nargs="+", default=["mbart", "mt5", "byt5"])
    ap.add_argument("--smoke", action="store_true", help="run the Phase E smoke test before full training")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    a = ap.parse_args()

    if not a.skip_data:
        run("data_prep.py", *(["--file", a.file] if a.file else []))
    run("check_lengths.py", "--size", a.size)
    run("evaluate.py", "--baseline")
    if a.smoke:
        run("smoke_test.py", "--size", a.size, "--models", *a.models)
    for m in a.models:
        run("train.py", "--model", m, "--size", a.size, *(["--grad-ckpt"] if a.grad_ckpt else []))
        run("evaluate.py", "--model", m, "--size", a.size)
    run("compare.py")
    run("analyze.py")


if __name__ == "__main__":
    main()
