"""Load a saved adapter once and reconstruct sentences, backing off the batch size on out-of-memory.

Also holds the two non-neural helpers used in the decoding study: a rule-based cleaner (collapse letter runs) and a
length guard that replaces a suspiciously short model output with that rule-based cleaning.
"""
import re
import sys
from types import SimpleNamespace

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from config import OUT_DIR, MAX_LEN
from memory import run_with_backoff, free_cuda
from models import load_base, input_prefix, gen_kwargs, run_tag

GUARD_MIN_RATIO = 0.6


def adapter_dir(model_key, size="base", tag_suffix=""):
    return OUT_DIR / (run_tag(model_key, size) + tag_suffix) / "adapter"


def load_adapter(model_key, size="base", tag_suffix="", base_path=None):
    """Tokenizer and weights come from disk (the saved adapter), not from the training process."""
    ad = adapter_dir(model_key, size, tag_suffix)
    if not (ad / "adapter_config.json").exists():
        sys.exit(f"No adapter at {ad}. Train first: python train.py --model {model_key} --size {size}")
    tok = AutoTokenizer.from_pretrained(ad)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = PeftModel.from_pretrained(load_base(model_key, size, base_path), ad).to(dev).eval()
    return SimpleNamespace(tok=tok, model=model, dev=dev, pre=input_prefix(model_key), gk=gen_kwargs(model_key, tok))


def release(loaded):
    del loaded.model
    free_cuda()


def generate(loaded, noisy, beams=1, batch=16):
    """Greedy (beams=1) or beam search reconstruction of every string in `noisy`."""
    def one(chunk):
        x = loaded.tok([loaded.pre + s for s in chunk], return_tensors="pt", padding=True, truncation=True,
                       max_length=MAX_LEN).to(loaded.dev)
        with torch.no_grad():
            y = loaded.model.generate(**x, max_length=MAX_LEN, num_beams=beams, do_sample=False, **loaded.gk)
        return [p.strip() for p in loaded.tok.batch_decode(y, skip_special_tokens=True)]

    preds, _ = run_with_backoff(one, list(noisy), batch)
    return preds


def collapse_runs(text, keep=2):
    """Rule-based cleaner: any run of 3+ identical letters becomes `keep` letters ('cooool' -> 'cool')."""
    return re.sub(r"([^\W\d_])\1{2,}", lambda m: m.group(1) * keep, text)


def length_guard(preds, noisy, min_ratio=GUARD_MIN_RATIO):
    """Replace outputs far shorter than the input (the model dropped words) with the rule-based cleaning of the input.
    Length is compared on letters only, against the input after run collapsing, so elongation does not hide a drop."""
    out, replaced = [], 0
    for p, n in zip(preds, noisy):
        fallback = collapse_runs(n)
        if len(p.replace(" ", "")) < min_ratio * len(fallback.replace(" ", "")):
            out.append(fallback)
            replaced += 1
        else:
            out.append(p)
    return out, replaced


def parse_systems(items):
    """'mbart' -> ('mbart', ''), 'mt5:_lr1e-3' -> ('mt5', '_lr1e-3'): a backbone plus its run-name variant."""
    out = []
    for it in items:
        key, _, variant = it.partition(":")
        out.append((key, variant))
    return out


def seed_suffixes(model_key, size="base", variant=""):
    """Tag suffixes of every trained seed of one system, seed 42 first: ['_lr1e-3', '_lr1e-3_s43', ...]."""
    base = run_tag(model_key, size) + variant
    found = []
    if (OUT_DIR / base / "adapter" / "adapter_config.json").exists():
        found.append((42, variant))
    for d in sorted(OUT_DIR.glob(base + "_s*")):
        m = re.search(r"_s(\d+)$", d.name)
        if m and (d / "adapter" / "adapter_config.json").exists():
            found.append((int(m.group(1)), d.name[len(run_tag(model_key, size)):]))
    return sorted(found)
