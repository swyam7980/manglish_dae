"""Everything for our own models, in order, skipping steps whose output already exists:

  data -> train_byt5 -> train_tiny -> export -> eval -> demo

    python run_ours.py                    # the lot
    python run_ours.py --steps eval demo  # just those
Logs go to logs/ours_<step>.log.
"""
import argparse
import os
import subprocess
import sys
import time

from config import PROC_DIR, OUT_DIR, RES_DIR, WORK

STEPS = {
    "data": (["ours_data.py"], PROC_DIR / "ours_ml_test.jsonl"),
    "train_byt5": (["train_ours.py", "--auto-batch", "--grad-ckpt", "--resume"], OUT_DIR / "ours_byt5" / "model" / "config.json"),
    "train_tiny": (["train_tiny.py"], OUT_DIR / "ours_tiny" / "train_info.json"),
    "export": (["export_tiny.py"], OUT_DIR / "ours_tiny" / "onnx" / "export_report.json"),
    "eval": (["eval_ours.py"], RES_DIR / "ours" / "ours_summary.json"),
    "demo": (["build_demo.py"], OUT_DIR / "demo_site" / "index.html"),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    ap.add_argument("--force", action="store_true", help="run steps even when their output exists")
    a = ap.parse_args(argv)
    logs = WORK / "logs"
    logs.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",  # Malayalam in a cp1252 console
           "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1")}
    times = {}
    for name in a.steps:
        cmd, done = STEPS[name]
        if done.exists() and not a.force:
            print(f"[skip] {name}: {done.name} exists", flush=True)
            continue
        t0 = time.time()
        print(f"[run] {name}: {' '.join(cmd)}", flush=True)
        with open(logs / f"ours_{name}.log", "w", encoding="utf-8") as log:
            r = subprocess.run([sys.executable, *cmd], stdout=log, stderr=subprocess.STDOUT, env=env,
                               cwd=os.path.dirname(os.path.abspath(__file__)))
        times[name] = round((time.time() - t0) / 60, 1)
        if r.returncode != 0 or not done.exists():
            sys.exit(f"[fail] {name} (exit {r.returncode}); see {logs / f'ours_{name}.log'}")
        print(f"[done] {name} in {times[name]} min", flush=True)
    return times


if __name__ == "__main__":
    main()
