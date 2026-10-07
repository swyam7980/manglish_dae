"""Sanity-check a backbone BEFORE LoRA: loss of the plain pretrained model on noisy -> clean pairs, in the
configured precision and in fp32, plus a few greedy generations. Separates 'precision problem' from
'model is just far from this task'."""
import argparse
import gc
import json

import torch
from transformers import AutoModelForSeq2SeqLM

from config import PROC_DIR, MAX_LEN
from models import load_tokenizer, input_prefix, gen_kwargs, model_name, pick_dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mbart", "mt5", "byt5"])
    ap.add_argument("--size", default="base")
    ap.add_argument("--base-path")
    ap.add_argument("--rows", type=int, default=16)
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(PROC_DIR / "val.jsonl", encoding="utf-8")][: a.rows]
    tok = load_tokenizer(a.model, a.size, a.base_path)
    pre, gk = input_prefix(a.model), gen_kwargs(a.model, tok)
    x = tok([pre + r["noisy"] for r in rows], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    y = tok(text_target=[r["clean"] for r in rows], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    labels = y["input_ids"].masked_fill(y["input_ids"] == tok.pad_token_id, -100)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    configured, _ = pick_dtype(a.model)
    for dtype in dict.fromkeys([configured, torch.float32]):  # configured precision first, then fp32
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name(a.model, a.size, a.base_path), dtype=dtype).to(dev).eval()
        with torch.no_grad():
            out = model(**x.to(dev), labels=labels.to(dev))
            gen = model.generate(**x, max_length=MAX_LEN, num_beams=1, do_sample=False, **gk)
        loss = out.loss.item()
        print(f"\n[{a.model} {str(dtype).replace('torch.', '')}] mean token loss, no LoRA: {loss:.3f}"
              f"{'  <-- NaN/inf' if loss != loss or abs(loss) == float('inf') else ''}")
        for r, g in list(zip(rows, tok.batch_decode(gen, skip_special_tokens=True)))[:3]:
            print(f"  noisy: {r['noisy']!r}\n  clean: {r['clean']!r}\n  pred : {g!r}")
        del model, out, gen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
