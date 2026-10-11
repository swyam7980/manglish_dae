"""Run the follow-up programme as a resumable queue, in phases.

  seeds     extra training seeds (45, 46) for mbart and byt5 with the original settings
  data      build train_plus_<tag> and test_<tag> from SeLLaI Manglish + Mixed-Code (+ Dakshina when given)
  train     train every backbone on the combined training file (largest micro-batch that fits), then score on the
            original validation set
  combined  score every trained adapter on the combined multi-source test set, by source
  studies   decoding study and robustness test on a subsample of the combined test set
  report    comparison tables

A step is skipped when its result exists, a failing step is logged and the queue carries on, and training
checkpoints (--resume) so re-running after a crash or power cut continues.

  python run_more.py --phases seeds
  python run_more.py --dakshina D:/.../data/raw/dakshina_dataset_v1.0.tar --phases data train combined studies report
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from config import OUT_DIR, RES_DIR, PROC_DIR, WORK

HERE = Path(__file__).resolve().parent
LOGS = WORK / "logs"
PHASES = ["seeds", "data", "train", "combined", "studies", "report"]
BASE_SYSTEMS = ["mbart", "byt5", "mt5:_lr1e-3"]
STUDY_SAMPLE = 600


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
    ap.add_argument("--phases", nargs="+", choices=PHASES, default=PHASES)
    ap.add_argument("--tag", default="all", help="name of the combined dataset: train_plus_<tag> / test_<tag>")
    ap.add_argument("--dakshina", help="Dakshina directory or dakshina_dataset_v1.0.tar to add as an extra source")
    ap.add_argument("--no-mixed", action="store_true", help="leave out the SeLLaI Mixed-Code class")
    ap.add_argument("--seeds", nargs="*", type=int, default=[45, 46])
    ap.add_argument("--plus-seeds", nargs="*", type=int, default=[], help="extra seeds for the combined-data runs of mbart/byt5")
    ap.add_argument("--sample", type=int, default=STUDY_SAMPLE, help="test sentences used by the decoding and robustness studies")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    q = Queue(a.dry_run)
    t0 = time.time()
    tag, train_file, test_file = a.tag, f"train_plus_{a.tag}.jsonl", f"test_{a.tag}.jsonl"
    fast = ["--auto-batch"]
    mt5_variants = [(f"_plus{tag}", []), (f"_lr1e-3_plus{tag}", ["--lr", "1e-3"])]
    plus_systems = ["mbart:_plus" + tag, "byt5:_plus" + tag, "mt5:_plus" + tag, f"mt5:_lr1e-3_plus{tag}"]
    systems = BASE_SYSTEMS + plus_systems

    if "seeds" in a.phases:
        for s in a.seeds:
            for m in ("mbart", "byt5"):
                q.train_and_eval(m, f"_s{s}", ["--seed", str(s)])

    if "data" in a.phases:
        args = ["extra_data.py", "--tag", tag]
        if not a.no_mixed:
            args += ["--sellai-label", "Mixed-Code"]
        if a.dakshina:
            args += ["--dakshina", a.dakshina]
        if "--sellai-label" in args or a.dakshina:
            q.run(f"data_{tag}", args, PROC_DIR / train_file)

    if "train" in a.phases:
        for m in ("mbart", "byt5"):
            q.train_and_eval(m, f"_plus{tag}", ["--train-file", train_file, *fast])
        for suf, extra in mt5_variants:
            q.train_and_eval("mt5", suf, ["--train-file", train_file, *fast, *extra])
        for s in a.plus_seeds:
            for m in ("mbart", "byt5"):
                q.train_and_eval(m, f"_plus{tag}_s{s}", ["--train-file", train_file, *fast, "--seed", str(s)])

    if "combined" in a.phases:
        q.run("eval_combined", ["eval_combined.py", "--eval-file", test_file, "--systems", *systems, "--out", f"combined_{tag}"])

    if "studies" in a.phases:
        common = ["--systems", *systems, "--eval-file", test_file, "--max-eval", str(a.sample)]
        q.run("decode_study", ["decode_study.py", *common, "--out", f"decoding_{tag}"])
        q.run("robustness", ["robustness.py", *common, "--out", f"robustness_{tag}"])

    if "report" in a.phases:
        q.run("compare", ["compare.py"])
        q.run("analyze", ["analyze.py"])
        q.run("seeds", ["seeds.py"])

    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min: {q.ran} ran, {q.skipped} skipped, {len(q.failed)} failed {q.failed}")
    return 1 if q.failed else 0


if __name__ == "__main__":
    sys.exit(main())
