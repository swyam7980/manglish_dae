"""Use our trained models: clean noisy Manglish, write it in Malayalam script, or translate it to English.

    python translate.py --task clean "njaaan naaale varummm"
    python translate.py --task ml --model tiny "njan nale varum"
    python translate.py --task en "njan tomorrow varum"
    python translate.py --task clean --file in.txt --out out.txt

--model byt5 loads outputs/ours_byt5_en/model when it exists (the model that also does en:), else outputs/ours_byt5/model
(or --model-dir, e.g. a folder downloaded from the Hugging Face Hub); --model tiny loads outputs/ours_tiny_en when it
exists, else outputs/ours_tiny. Clean-task outputs from ByT5 go through the length guard from the comparison.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

from config import OUT_DIR
from generation import length_guard
from memory import run_with_backoff
from tiny_model import load_tiny, pad_batch

TASKS = ("clean", "ml", "en")
BYT5_DIRS = [OUT_DIR / "ours_byt5_en" / "model", OUT_DIR / "ours_byt5" / "model"]
TINY_DIRS = [OUT_DIR / "ours_tiny_en", OUT_DIR / "ours_tiny"]
DEFAULT_DIRS = {"byt5": next((d for d in BYT5_DIRS if (d / "config.json").exists()), BYT5_DIRS[-1]),
                "tiny": next((d for d in TINY_DIRS if (d / "model.pt").exists()), TINY_DIRS[-1])}
BYT5_PREFIX = {"clean": "clean: ", "ml": "ml: ", "en": "en: "}
BYT5_MAX_SRC, BYT5_MAX_TGT = 192, 384


class Translator:
    """One loaded model. translate(texts, task) -> list of outputs, in input order."""

    def __init__(self, kind="byt5", model_dir=None, device=None, beams=1, guard=True):
        self.kind, self.beams, self.guard = kind, beams, guard
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        folder = Path(model_dir or DEFAULT_DIRS[kind])
        if kind == "byt5":
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            if not (folder / "config.json").exists():
                sys.exit(f"No model in {folder}. Train with train_ours.py or download it (see README).")
            dtype = torch.bfloat16 if self.dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
            self.tok = AutoTokenizer.from_pretrained(folder)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(folder, torch_dtype=dtype).to(self.dev).eval()
        elif kind == "tiny":
            if not (folder / "model.pt").exists():
                sys.exit(f"No model in {folder}. Train with train_tiny.py or download it (see README).")
            self.model, self.vocab, self.cfg = load_tiny(folder, self.dev)
        else:
            raise ValueError(kind)
        self.tasks = self._tasks(folder)

    def _tasks(self, folder):
        """Tasks this model was trained for, from its train_info.json (a model without one does clean and ml)."""
        if self.kind == "tiny":
            return tuple(t for t in TASKS if t in self.vocab.tasks)
        for info in (folder / "train_info.json", folder.parent / "train_info.json"):
            if info.exists():
                return tuple(json.loads(info.read_text())["tasks"])
        return ("clean", "ml")

    def _byt5(self, chunk, task):
        x = self.tok([BYT5_PREFIX[task] + s for s in chunk], return_tensors="pt", padding=True, truncation=True,
                     max_length=BYT5_MAX_SRC).to(self.dev)
        with torch.no_grad():
            y = self.model.generate(**x, max_length=BYT5_MAX_TGT, num_beams=self.beams, do_sample=False)
        return [p.strip() for p in self.tok.batch_decode(y, skip_special_tokens=True)]

    def _tiny(self, chunk, task):
        src = pad_batch([self.vocab.encode_src(s, task, self.cfg["max_src"]) for s in chunk]).to(self.dev)
        out = self.model.greedy(src, self.cfg["max_tgt"])
        return [self.vocab.decode(row.tolist()).strip() for row in out]

    def translate(self, texts, task="clean", batch=32):
        if task not in self.tasks:
            raise ValueError(f"this model does the tasks {self.tasks}, not {task!r}")
        texts = [t.strip().lower() for t in texts]
        fn = self._byt5 if self.kind == "byt5" else self._tiny
        preds, _ = run_with_backoff(lambda c: fn(c, task), texts, batch)
        if task == "clean" and self.guard:
            preds, _ = length_guard(preds, texts)
        return preds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="sentences to translate (or use --file)")
    ap.add_argument("--task", choices=TASKS, default="clean")
    ap.add_argument("--model", choices=list(DEFAULT_DIRS), default="byt5")
    ap.add_argument("--model-dir", help="model folder (default: the one train_ours.py / train_tiny.py wrote)")
    ap.add_argument("--file", help="one sentence per line")
    ap.add_argument("--out", help="write outputs here, one per line (default: print)")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--device", choices=["cuda", "cpu"])
    ap.add_argument("--no-guard", action="store_true", help="turn off the length guard for the clean task")
    a = ap.parse_args(argv)
    texts = list(a.text)
    if a.file:
        texts += [l.rstrip("\n") for l in open(a.file, encoding="utf-8") if l.strip()]
    if not texts:
        ap.error("give sentences as arguments or with --file")
    tr = Translator(a.model, a.model_dir, a.device, a.beams, not a.no_guard)
    preds = tr.translate(texts, a.task)
    if a.out:
        Path(a.out).write_text("\n".join(preds) + "\n", encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        for s, p in zip(texts, preds):
            print(f"{s}\t->\t{p}")
    return preds


if __name__ == "__main__":
    main()
