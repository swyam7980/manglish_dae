"""Save a 16-bit copy of a second-stage translator in outputs/<name>_fp16 (IndicTrans2 1B: 4.1 GB -> 2.0 GB).

Windows maps a whole safetensors file into memory to load it, and with a small page file the 4 GB float32 file of
IndicTrans2 1B fails with "The paging file is too small" (os error 1455). The weights run in float16 on the GPU anyway,
so this copy changes nothing in the output. The original is read with plain file reads, one tensor at a time, so the
conversion itself never maps the large file. translators.py uses the copy when it exists.

    python src/convert_translator.py              # it2_1b
"""
import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from config import OUT_DIR
from translators import SYSTEMS, read_safetensors


def convert(name):
    src = Path(snapshot_download(SYSTEMS[name][1], local_files_only=True))
    out = OUT_DIR / f"{name}_fp16"
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name not in ("model.safetensors", "pytorch_model.bin"):
            shutil.copy(f, out / f.name)  # config, tokenizer files and the model's own code
    tensors = {k: (t.half() if t.is_floating_point() else t) for k, t in read_safetensors(src / "model.safetensors")}
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    cfg["torch_dtype"] = "float16"
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="it2_1b", choices=list(SYSTEMS))
    out = convert(ap.parse_args(argv).name)
    print(f"wrote {out} ({(out / 'model.safetensors').stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
