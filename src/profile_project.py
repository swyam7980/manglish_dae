"""Profile the datasets and the compute used, for the results page.

Writes results/data/dataset_profile.json (names, providers, licences, sizes and sentence lengths per source) and
results/run_times.json (one row per training run from outputs/*/train_info.json, plus the longer evaluation and
study steps taken from run_more.py console logs passed with --queue-log).
"""
import argparse
import json
import re
import statistics
import subprocess

from config import OUT_DIR, PROC_DIR, RES_DIR

DATASETS = {
    "sellai_manglish": {
        "name": "SeLLaI Mal-Eng, Manglish class", "provider": "Mendeley Data (SeLLaI Mal-Eng, 2024)",
        "licence": "CC BY 4.0", "url": "https://data.mendeley.com/datasets/5p4zbpy8wz/1",
        "register": "image captions, Latin-script Malayalam-English", "raw_key": "Manglish"},
    "sellai_mixed-code": {
        "name": "SeLLaI Mal-Eng, Mixed-Code class", "provider": "Mendeley Data (same file as above)",
        "licence": "CC BY 4.0", "url": "https://data.mendeley.com/datasets/5p4zbpy8wz/1",
        "register": "longer image captions, more Malayalam and more English mixed in", "raw_key": "Mixed-Code"},
    "dakshina_ml": {
        "name": "Dakshina, Malayalam romanised sentences", "provider": "Google Research (Roark et al., 2020)",
        "licence": "CC BY-SA 4.0", "url": "https://github.com/google-research-datasets/dakshina",
        "register": "formal Wikipedia sentences romanised by hand, no English mixing", "raw_key": None},
}
NOT_USED = [
    {"name": "SeLLaI Mal-Eng, English class", "rows": 10030, "why": "plain English, not Manglish"},
    {"name": "DravidianCodeMix (Malayalam-English comments)", "rows": None,
     "why": "comments are already noisy, so they cannot serve as clean targets"},
]


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def profile_datasets(tag="all"):
    train = read_jsonl(PROC_DIR / f"train_plus_{tag}.jsonl")
    test = read_jsonl(PROC_DIR / f"test_{tag}.jsonl")
    stats = json.loads((PROC_DIR / f"train_plus_{tag}.stats.json").read_text(encoding="utf-8"))
    base = json.loads((PROC_DIR / "stats.json").read_text(encoding="utf-8"))
    out = []
    for key, meta in DATASETS.items():
        tr = [r["clean"] for r in train if r["source"] == key]
        te = [r["clean"] for r in test if r["source"] == key]
        both = tr + te
        if key == "sellai_manglish":
            raw, kept = base["class_rows"], base["after_dedupe"]
        else:
            ps = stats["per_source"][key]
            raw, kept = ps["input_lines"], ps["kept_unique"]
        out.append({**meta, "key": key, "raw_rows": raw, "clean_rows": kept, "train": len(tr), "test": len(te),
                    "avg_chars": round(statistics.mean(len(s) for s in both), 1),
                    "avg_words": round(statistics.mean(len(s.split()) for s in both), 1),
                    "total_chars": sum(len(s) for s in both)})
    return {"datasets": out, "not_used": NOT_USED, "train_total": len(train), "test_total": len(test)}


def gpu_name():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
        return r.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def profile_runs(queue_logs):
    runs = []
    for d in sorted(OUT_DIR.iterdir()):
        info = d / "train_info.json"
        if not info.exists() or d.name.endswith("_smoke") or d.name.endswith("_probe"):
            continue
        t = json.loads(info.read_text())
        m = re.search(r"_s(\d+)$", d.name)
        runs.append({"tag": d.name, "system": re.sub(r"_s\d+$", "", d.name), "seed": int(m.group(1)) if m else t.get("seed", 42),
                     "train_rows": t.get("train_rows") or 2934, "steps": t["steps"], "minutes": t["minutes"],
                     "peak_gpu_gb": t["peak_gpu_gb"], "micro_batch": t.get("micro_batch") or 2, "grad_accum": t.get("grad_accum") or 8,
                     "precision": t["precision"], "learning_rate": t.get("learning_rate", 2e-4), "trainable_pct": t["trainable_pct"]})
    steps = []
    for path in queue_logs:
        for line in open(path, encoding="utf-8", errors="replace"):
            m = re.match(r"\[ ok \] (\w[\w\-.]*) \(([\d.]+) min\)", line)
            if m and float(m.group(2)) >= 2 and not m.group(1).startswith("train_"):
                steps.append({"name": m.group(1), "minutes": float(m.group(2))})
    return {"hardware": gpu_name(), "runs": runs, "steps": steps,
            "training_minutes": round(sum(r["minutes"] for r in runs), 1),
            "step_minutes": round(sum(s["minutes"] for s in steps), 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="all")
    ap.add_argument("--queue-log", nargs="*", default=[], help="console output of run_more.py, for the study step times")
    a = ap.parse_args(argv)
    (RES_DIR / "data").mkdir(parents=True, exist_ok=True)
    prof = profile_datasets(a.tag)
    (RES_DIR / "data" / "dataset_profile.json").write_text(json.dumps(prof, indent=1), encoding="utf-8")
    runs = profile_runs(a.queue_log)
    (RES_DIR / "run_times.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
    for d in prof["datasets"]:
        print(f"{d['name']:44s} raw {d['raw_rows']:>6,} clean {d['clean_rows']:>6,} train {d['train']:>6,} test {d['test']:>5,} "
              f"avg {d['avg_chars']} chars / {d['avg_words']} words")
    print(f"{len(runs['runs'])} training runs, {runs['training_minutes'] / 60:.1f} h training + {runs['step_minutes'] / 60:.1f} h evaluation and studies on {runs['hardware']}")
    return prof, runs


if __name__ == "__main__":
    main()
