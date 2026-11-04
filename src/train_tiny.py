"""Train Manglish-Tiny from scratch on the same rows and noise mix as Manglish-ByT5.

Each epoch draws fresh noise (train_ours.build_examples), so the small model sees many corruptions of every sentence.
The checkpoint with the lowest loss on a fixed slice of noisy training-style examples is kept (the test sets are never
looked at during training). Output: outputs/ours_tiny/{model.pt, vocab.json, config.json, train_info.json}.

--en-file adds the en: task and code-mixed ml: examples exactly as train_ours.py does (see en_data.py); give it a
larger model, e.g. --tag ours_tiny_en --d-model 384 --heads 6 --enc-layers 6 --dec-layers 4 --ff 1536 --epochs 10.
"""
import argparse
import json
import math
import random
import time

import torch
from torch import nn

from config import PROC_DIR, OUT_DIR, SEED
from tiny_model import DEFAULTS, PAD, BOS, CharVocab, TinySeq2Seq, pad_batch
from train import read_jsonl
from train_ours import build_examples, sample_rows, MLX_SWAP_MIX, SCRIPT_SWAP_MIX, TASK_PREFIX

MAX_SRC = 200
MAX_TGT = 200
EPOCHS = 40
BATCH = 128
LR = 1e-3
DEV_ROWS = 1000
TAG = "ours_tiny"


def split_prefix(src):
    """'clean: text' -> ('clean', 'text'): the ByT5 examples carry their task as a text prefix."""
    for task, pre in TASK_PREFIX.items():
        if src.startswith(pre):
            return task, src[len(pre):]
    raise ValueError(src[:20])


def to_ids(examples, vocab):
    out = []
    for e in examples:
        task, text = split_prefix(e["src"])
        out.append((vocab.encode_src(text, task, MAX_SRC), vocab.encode_tgt(e["tgt"], MAX_TGT)))
    return out


def batches(pairs, size, rng):
    """Length-bucketed batches: sort a shuffled pool by length, cut it into batches, shuffle the batches."""
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    pool = 50 * size
    out = []
    for s in range(0, len(idx), pool):
        chunk = sorted(idx[s:s + pool], key=lambda i: len(pairs[i][1]))
        out += [chunk[k:k + size] for k in range(0, len(chunk), size)]
    rng.shuffle(out)
    return out


def loss_on(model, pairs, crit, dev, size=256):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for k in range(0, len(pairs), size):
            src, tgt = zip(*pairs[k:k + size])
            loss, toks = step_loss(model, src, tgt, crit, dev)
            tot += loss.item() * toks
            n += toks
    model.train()
    return tot / n


def step_loss(model, src, tgt, crit, dev):
    s = pad_batch(list(src)).to(dev)
    t = pad_batch([[BOS] + x for x in tgt]).to(dev)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        logits = model(s, t[:, :-1])
    y = t[:, 1:]
    return crit(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)), int(y.ne(PAD).sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", default="train_plus_all.jsonl")
    ap.add_argument("--ml-file", default="ours_ml_train.jsonl")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--en-file", help="en_data.py output: adds the en: task and code-mixed ml: examples")
    ap.add_argument("--en-max", type=int, default=0, help="English pairs per epoch (0 = all)")
    ap.add_argument("--mlx-max", type=int, default=40000, help="code-mixed ml: examples per epoch")
    ap.add_argument("--script-only", action="store_true",
                    help="with --en-file: no en: task; every pair becomes an ml: example, plain or code-mixed")
    ap.add_argument("--init-from", help="output tag of a trained Tiny to continue from (its vocabulary and architecture)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--device", choices=["cuda", "cpu"], help="default: cuda when available")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    clean_rows = [r["clean"] for r in read_jsonl(PROC_DIR / a.train_file)]
    ml_rows = read_jsonl(PROC_DIR / a.ml_file)
    en_rows, mlx_rows = [], []
    if a.en_file:
        all_en = read_jsonl(PROC_DIR / a.en_file)
        if a.script_only:
            mlx_rows = sample_rows(all_en, a.mlx_max, a.seed, "mlx")
        else:
            en_rows = sample_rows(all_en, a.en_max, a.seed, "en")
            mlx_rows = sample_rows([r for r in all_en if any(sw for _, sw in r["toks"])], a.mlx_max, a.seed, "mlx")
    if a.max_train:
        clean_rows, ml_rows = clean_rows[: a.max_train], ml_rows[: a.max_train]
        en_rows, mlx_rows = en_rows[: a.max_train], mlx_rows[: a.max_train]
    tasks = ("clean", "ml", "en") if a.en_file and not a.script_only else ("clean", "ml")
    # the vocabulary covers every character the targets can contain, plus the Malayalam block
    malayalam = [chr(c) for c in range(0x0D00, 0x0D80)] + [chr(0x200C), chr(0x200D)]  # + zero-width (non-)joiner
    init = OUT_DIR / a.init_from if a.init_from else None
    if init:  # keep the old ids so the trained weights still line up; unseen characters map to <unk>
        vocab = CharVocab.load(init / "vocab.json")
        for k, v in json.loads((init / "config.json").read_text(encoding="utf-8"))["arch"].items():
            setattr(a, k, v)
    else:
        vocab = CharVocab.from_texts(clean_rows + [r["native"] for r in ml_rows] + [r["roman"] for r in ml_rows]
                                     + malayalam + [r["en"] for r in en_rows] + [r["ml"] for r in mlx_rows], tasks=tasks)
    mix = SCRIPT_SWAP_MIX if a.script_only else MLX_SWAP_MIX
    epoch_examples = lambda ep: build_examples(clean_rows, ml_rows, 1, a.seed, first_epoch=ep, en_rows=en_rows,
                                               mlx_rows=mlx_rows, mlx_mix=mix)
    dev_pairs = to_ids(epoch_examples(10_000), vocab)
    dev_pairs = random.Random(a.seed).sample(dev_pairs, min(DEV_ROWS, len(dev_pairs)))

    arch = {k: getattr(a, k) for k in DEFAULTS}
    model = TinySeq2Seq(len(vocab), **arch).to(dev)
    if init:
        model.load_state_dict(torch.load(init / "model.pt", map_location=dev))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[tiny] {n_params / 1e6:.2f}M parameters, vocab {len(vocab)}, device {dev}", flush=True)
    per_epoch = len(clean_rows) + len(ml_rows) + len(en_rows) + len(mlx_rows)
    total_steps = a.epochs * math.ceil(per_epoch / a.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    warm = max(1, int(0.04 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.02, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))
    crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    out = OUT_DIR / a.tag
    out.mkdir(parents=True, exist_ok=True)
    vocab.save(out / "vocab.json")
    cfg = {"arch": arch, "max_src": MAX_SRC, "max_tgt": MAX_TGT, "task_prefix": {t: TASK_PREFIX[t] for t in tasks},
           "tasks": list(tasks), "params": n_params}
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    history, best, t0, step = [], math.inf, time.time(), 0
    model.train()
    for ep in range(a.epochs):
        pairs = to_ids(epoch_examples(ep), vocab)
        run, seen = 0.0, 0
        for b in batches(pairs, a.batch, rng):
            src, tgt = zip(*(pairs[i] for i in b))
            loss, _ = step_loss(model, src, tgt, crit, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run += loss.item()
            seen += 1
        dev_loss = loss_on(model, dev_pairs, crit, dev)
        history.append({"epoch": ep + 1, "train_loss": round(run / seen, 4), "dev_loss": round(dev_loss, 4)})
        print(f"[tiny] epoch {ep + 1}/{a.epochs} train {run / seen:.4f} dev {dev_loss:.4f} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if dev_loss < best:
            best = dev_loss
            torch.save(model.state_dict(), out / "model.pt")
    info = {"params": n_params, "vocab": len(vocab), "epochs": a.epochs, "steps": step, "batch": a.batch, "lr": a.lr,
            "best_dev_loss": round(best, 4), "minutes": round((time.time() - t0) / 60, 2), "seed": a.seed,
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None,
            "clean_rows": len(clean_rows), "ml_rows": len(ml_rows), "en_rows": len(en_rows), "mlx_rows": len(mlx_rows),
            "tasks": list(tasks), "arch": arch, "init_from": a.init_from, "script_only": a.script_only,
            "mlx_swap_mix": mix if mlx_rows else None, "history": history}
    (out / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("DONE", {k: v for k, v in info.items() if k != "history"})
    return info


if __name__ == "__main__":
    main()
