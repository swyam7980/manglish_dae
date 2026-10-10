"""Run the whole follow-up programme in one resumable queue.

Order: extra-data training runs (more data first), extra seeds, then the decoding study and robustness test, then
the aggregate tables. Every step is skipped when its result already exists, a failing step is logged and the queue
carries on, and training runs checkpoint (--resume) so re-running this script after a crash or power cut continues.

  python run_more.py                       # SeLLaI Mixed-Code extra data
  python run_more.py --dakshina PATH       # also builds and trains on SeLLaI + Dakshina
  python run_more.py --dry-run             # print the plan only
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from config import OUT_DIR, RES_DIR, PROC_DIR, WORK

HERE = Path(__file__).resolve().parent
LOGS = WORK / "logs"
MODELS = ["mbart", "byt5", "mt5"]
# (tag suffix, extra train.py args) per backbone for runs on extra data
MT5_VARIANTS = [("_plus{tag}", []), ("_lr1e-3_plus{tag}", ["--lr", "1e-3"])]


class Queue:
    def __init__(self, dry):
        self.dry, self.failed, self.skipped, self.ran = dry, [], 0, 0
        LOGS.mkdir(parents=True, exist_ok=True)

    def run(self, name, args, done=None):
        if done is not None and done.exists():
            self.skipped += 1
            print(f"[skip] {name} (found {done.name})", flush=True)
            return True
        print(f"[run ] {name}: {' '.join(args)}", flush=True)
        if self.dry:
            return True
        t0 = time.time()
        with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
            rc = subprocess.run([sys.executable, str(HERE / args[0]), *args[1:]], cwd=HERE, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        mins = (time.time() - t0) / 60
        if rc == 0:
            self.ran += 1
            print(f"[ ok ] {name} ({mins:.1f} min)", flush=True)
        else:
            self.failed.append(name)
            print(f"[FAIL] {name} (exit {rc}, {mins:.1f} min) see {LOGS / (name + '.log')}", flush=True)
        return rc == 0

    def train_and_eval(self, model, suffix, train_args):
        tag = f"{'mbart_large' if model == 'mbart' else model + '_base'}{suffix}"
        ok = self.run(f"train_{tag}", ["train.py", "--model", model, "--resume", "--tag-suffix", suffix, *train_args],
                      OUT_DIR / tag / "train_info.json")
        if ok or (OUT_DIR / tag / "adapter" / "adapter_config.json").exists():
            self.run(f"eval_{tag}", ["evaluate.py", "--model", model, "--tag-suffix", suffix], RES_DIR / f"metrics_{tag}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dakshina", help="Dakshina directory or dakshina_dataset_v1.0.tar to add as a second extra source")
    ap.add_argument("--seeds", nargs="*", type=int, default=[45, 46], help="extra seeds for mbart and byt5 (default settings)")
    ap.add_argument("--plus-seeds", nargs="*", type=int, default=[], help="extra seeds for the mixed-data runs of mbart/byt5")
    ap.add_argument("--no-seeds", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    q = Queue(a.dry_run)
    t0 = time.time()

    q.run("data_plus_mixed", ["extra_data.py", "--tag", "mixed", "--sellai-label", "Mixed-Code"],
          PROC_DIR / "train_plus_mixed.jsonl")
    fast = ["--auto-batch"]

    for m in MODELS:
        if m == "mt5":
            for suf, extra in MT5_VARIANTS:
                q.train_and_eval(m, suf.format(tag="mixed"), ["--train-file", "train_plus_mixed.jsonl", *fast, *extra])
        else:
            q.train_and_eval(m, "_plusmixed", ["--train-file", "train_plus_mixed.jsonl", *fast])

    if not a.no_seeds:
        for s in a.seeds:
            for m in ("mbart", "byt5"):
                q.train_and_eval(m, f"_s{s}", ["--seed", str(s)])
    for s in a.plus_seeds:
        for m in ("mbart", "byt5"):
            q.train_and_eval(m, f"_plusmixed_s{s}", ["--train-file", "train_plus_mixed.jsonl", *fast, "--seed", str(s)])

    systems = ["mbart", "byt5", "mt5:_lr1e-3", "mbart:_plusmixed", "byt5:_plusmixed", "mt5:_plusmixed", "mt5:_lr1e-3_plusmixed"]
    if a.dakshina:
        q.run("data_plus_all", ["extra_data.py", "--tag", "all", "--sellai-label", "Mixed-Code", "--dakshina", a.dakshina],
              PROC_DIR / "train_plus_all.jsonl")
        for m in MODELS:
            if m == "mt5":
                for suf, extra in MT5_VARIANTS:
                    q.train_and_eval(m, suf.format(tag="all"), ["--train-file", "train_plus_all.jsonl", *fast, *extra])
            else:
                q.train_and_eval(m, "_plusall", ["--train-file", "train_plus_all.jsonl", *fast])
        systems += ["mbart:_plusall", "byt5:_plusall", "mt5:_plusall", "mt5:_lr1e-3_plusall"]

    q.run("decode_study", ["decode_study.py", "--systems", *systems])
    q.run("robustness", ["robustness.py", "--systems", *systems])
    q.run("compare", ["compare.py"])
    q.run("analyze", ["analyze.py"])
    q.run("seeds", ["seeds.py"])

    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min: {q.ran} ran, {q.skipped} skipped, {len(q.failed)} failed {q.failed}")
    return 1 if q.failed else 0


if __name__ == "__main__":
    sys.exit(main())
