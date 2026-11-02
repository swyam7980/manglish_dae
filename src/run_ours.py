"""Everything for our own models, in order, skipping steps whose output already exists:

  data -> train_byt5 -> train_tiny -> export -> eval -> demo                       (clean: and ml:)
  en_data -> train_byt5_en -> train_tiny_en -> export_en -> eval_en -> eval -> demo   (adds en:, see en_data.py)

    python run_ours.py                    # the lot
    python run_ours.py --steps eval demo  # just those
    python run_ours.py --steps en_data train_byt5_en train_tiny_en export_en eval_en eval demo --force-eval
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
    "en_data": (["en_data.py"], PROC_DIR / "ours_en_test.jsonl"),
    "train_byt5_en": (["train_ours.py", "--base-path", str(OUT_DIR / "ours_byt5" / "model"), "--en-file",
                       "ours_en_train.jsonl", "--en-max", "140000", "--epochs", "1", "--tag", "ours_byt5_en",
                       "--auto-batch", "--grad-ckpt", "--resume"], OUT_DIR / "ours_byt5_en" / "model" / "config.json"),
    "train_tiny_en": (["train_tiny.py", "--en-file", "ours_en_train.jsonl", "--en-max", "140000", "--tag", "ours_tiny_en",
                       "--d-model", "384", "--heads", "6", "--enc-layers", "6", "--dec-layers", "4", "--ff", "1536",
                       "--epochs", "10"], OUT_DIR / "ours_tiny_en" / "train_info.json"),
    "export_en": (["export_tiny.py", "--tag", "ours_tiny_en"], OUT_DIR / "ours_tiny_en" / "onnx" / "export_report.json"),
    "eval_en": (["eval_en.py"], RES_DIR / "ours_en" / "en_summary.json"),
}
REDO = {"eval", "demo"}  # summaries that new models extend: --force-eval reruns them


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    ap.add_argument("--force", action="store_true", help="run steps even when their output exists")
    ap.add_argument("--force-eval", action="store_true", help="rerun eval and demo even when their output exists")
    a = ap.parse_args(argv)
    logs = WORK / "logs"
    logs.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",  # Malayalam in a cp1252 console
           "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1")}
    times = {}
    for name in a.steps:
        cmd, done = STEPS[name]
        if done.exists() and not a.force and not (a.force_eval and name in REDO):
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
