"""Export Manglish-Tiny to ONNX for the browser demo: encoder.onnx + decoder.onnx (+ int8 copies) and vocab.json.

The browser runs greedy decoding itself: encode once, then call the decoder with the growing prefix until </s>.
A parity check decodes a few sentences with PyTorch and with onnxruntime and fails if they differ.
Output: outputs/ours_tiny/onnx/.
"""
import argparse
import json
import shutil
import sys

import torch

from config import OUT_DIR
from tiny_model import BOS, EOS, EncoderOnly, DecoderStep, load_tiny

OPSET = 17
PARITY_SENTENCES = [("clean", "njaaan naaale varummm"), ("clean", "ente ammaye ariyoo"),
                    ("ml", "njan nale varum"), ("ml", "kerala sarkkar"), ("ml", "njan tomorrow varum"),
                    ("en", "njan tomorrow varum"), ("en", "ente veedu evideyaanu")]  # en only if the model has it


def export(folder, out):
    model, vocab, cfg = load_tiny(folder)
    out.mkdir(parents=True, exist_ok=True)
    src = torch.tensor([vocab.encode_src("njan nale varum", "clean", cfg["max_src"])])
    mem = model.encode(src)
    tgt = torch.tensor([[BOS, 10, 11]])
    kw = {"opset_version": OPSET, "do_constant_folding": True, "dynamo": False}
    torch.onnx.export(EncoderOnly(model).eval(), (src,), out / "encoder.onnx", input_names=["src"],
                      output_names=["memory"], dynamic_axes={"src": {1: "s"}, "memory": {1: "s"}}, **kw)
    torch.onnx.export(DecoderStep(model).eval(), (tgt, mem), out / "decoder.onnx", input_names=["tgt", "memory"],
                      output_names=["logits"], dynamic_axes={"tgt": {1: "t"}, "memory": {1: "s"}}, **kw)
    shutil.copy(folder / "vocab.json", out / "vocab.json")
    (out / "config.json").write_text(json.dumps({**cfg, "bos": BOS, "eos": EOS}, indent=2), encoding="utf-8")
    return model, vocab, cfg


def quantize(out):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    for name in ("encoder", "decoder"):
        quantize_dynamic(out / f"{name}.onnx", out / f"{name}.int8.onnx", weight_type=QuantType.QInt8)


def ort_greedy(enc, dec, ids, max_len):
    import numpy as np
    mem = enc.run(None, {"src": np.array([ids], dtype=np.int64)})[0]
    ys = [BOS]
    for _ in range(max_len):
        logits = dec.run(None, {"tgt": np.array([ys], dtype=np.int64), "memory": mem})[0]
        nxt = int(logits[0].argmax())
        if nxt == EOS:
            break
        ys.append(nxt)
    return ys[1:]


def parity(model, vocab, cfg, out, suffix=""):
    import onnxruntime as ort
    enc = ort.InferenceSession(str(out / f"encoder{suffix}.onnx"))
    dec = ort.InferenceSession(str(out / f"decoder{suffix}.onnx"))
    rows = []
    for task, text in PARITY_SENTENCES:
        if task not in vocab.tasks:
            continue
        ids = vocab.encode_src(text, task, cfg["max_src"])
        ref = vocab.decode(model.greedy(torch.tensor([ids]), cfg["max_tgt"])[0].tolist())
        got = vocab.decode(ort_greedy(enc, dec, ids, cfg["max_tgt"]))
        rows.append({"task": task, "input": text, "torch": ref, "onnx": got, "same": ref == got})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ours_tiny")
    a = ap.parse_args(argv)
    folder = OUT_DIR / a.tag
    if not (folder / "model.pt").exists():
        sys.exit(f"No trained model in {folder}. Run train_tiny.py first.")
    out = folder / "onnx"
    model, vocab, cfg = export(folder, out)
    quantize(out)
    fp32 = parity(model, vocab, cfg, out)
    int8 = parity(model, vocab, cfg, out, ".int8")
    sizes = {p.name: round(p.stat().st_size / 1e6, 2) for p in sorted(out.glob("*.onnx"))}
    report = {"sizes_mb": sizes, "parity_fp32": fp32, "parity_int8": int8,
              "int8_agreement": sum(r["same"] for r in int8) / len(int8)}
    (out / "export_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not all(r["same"] for r in fp32):
        sys.exit("fp32 ONNX output differs from PyTorch; do not publish this export.")
    return report


if __name__ == "__main__":
    main()
