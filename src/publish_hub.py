"""Prepare the two free-hosting folders; uploading them needs your Hugging Face login, so this script only builds them.

  outputs/hub/    a Hugging Face model repo: byt5/ (merged Manglish-ByT5), tiny/ (Manglish-Tiny + ONNX), README.md
  outputs/space/  a Hugging Face Space: by default the static browser demo (free); --space-sdk gradio builds a Gradio
                  app that downloads both models from that repo (Gradio Spaces on free CPU need a PRO account)

Then, once (with a write token from https://huggingface.co/settings/tokens):
  hf auth login
  hf upload <user>/manglish-translator outputs/hub . --repo-type model
  hf repos create <user>/manglish-translator-demo --type space --space-sdk static
  hf upload <user>/manglish-translator-demo outputs/space . --repo-type space
"""
import argparse
import json
import shutil
import sys

from config import ROOT, OUT_DIR, RES_DIR

SPACE_SRC = ["translate.py", "tiny_model.py", "generation.py", "memory.py", "config.py", "models.py"]
SPACE_REQUIREMENTS = ["torch", "transformers>=4.46", "peft", "sentencepiece", "huggingface_hub", "gradio>=4.44"]
TINY_FILES = ["model.pt", "vocab.json", "config.json", "train_info.json"]


def results_table():
    p = RES_DIR / "ours" / "ours_summary.md"
    return p.read_text(encoding="utf-8").split("\n", 2)[-1] if p.exists() else "_Run eval_ours.py to fill this in._\n"


def model_card(repo, github):
    return f"""---
language: [ml, en]
license: cc-by-sa-4.0
tags: [manglish, malayalam, transliteration, text-normalization, byt5, code-mixed]
base_model: google/byt5-base
---
# Manglish translator

Two models that clean noisy romanised Malayalam (Manglish) and write it in Malayalam script.

| folder | model | size | use |
|---|---|---|---|
| `byt5/` | Manglish-ByT5: google/byt5-base fine-tuned with LoRA, adapter merged | ~1.2 GB | best quality |
| `tiny/` | Manglish-Tiny: 6.5M-parameter character transformer trained from scratch | 26 MB (PyTorch), ~7 MB (int8 ONNX) | CPU / browser |

Prefix the input with the task: `clean: ` (noisy Manglish to clean Manglish) or `ml: ` (Manglish to Malayalam script).

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import snapshot_download
path = snapshot_download("{repo}", allow_patterns=["byt5/*"]) + "/byt5"
tok, model = AutoTokenizer.from_pretrained(path), AutoModelForSeq2SeqLM.from_pretrained(path)
x = tok(["clean: njaaan naaale varummm", "ml: njan nale varum"], return_tensors="pt", padding=True)
print(tok.batch_decode(model.generate(**x, max_length=384), skip_special_tokens=True))
```

Manglish-Tiny needs the small model code from the project repository ({github}): `python src/translate.py --model tiny
--model-dir <download>/tiny --task ml "njan nale varum"`, or the ONNX files in `tiny/onnx/` with onnxruntime.

## Training data
The same split used to compare mBART-50, mT5 and ByT5 in the project: 18,548 training sentences from SeLLaI Manglish
and Mixed-Code (CC BY 4.0) and Dakshina romanised Malayalam (CC BY-SA 4.0), with synthetic noise drawn fresh every
epoch; plus 5,792 Dakshina sentence pairs and 8,000 lexicon words for the script task. No test sentence was trained on.
Because the training data includes CC BY-SA material, the weights are released under CC BY-SA 4.0.

## Results on the project's held-out test sets
{results_table()}
## Limits
Lower-cased output; trained on short sentences (up to 120 characters); script output follows Dakshina's Wikipedia
style. The noise is synthetic, so real typing errors may differ.
"""


def space_readme(repo, sdk):
    app = "app_file: app.py\n" if sdk == "gradio" else ""
    how = ("Runs both models on the Space's CPU." if sdk == "gradio" else
           "Manglish-Tiny runs in your browser (onnxruntime-web); nothing is sent to a server.")
    return f"""---
title: Manglish Translator
emoji: 🌴
colorFrom: red
colorTo: gray
sdk: {sdk}
{app}pinned: false
license: cc-by-sa-4.0
---
Clean noisy Manglish or write it in Malayalam script. {how} Weights: https://huggingface.co/{repo}
"""


def build_hub(out, repo, github):
    byt5, tiny = OUT_DIR / "ours_byt5" / "model", OUT_DIR / "ours_tiny"
    if not (byt5 / "config.json").exists() or not (tiny / "model.pt").exists():
        sys.exit("Train both models first (train_ours.py, train_tiny.py, export_tiny.py).")
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(byt5, out / "byt5")
    (out / "tiny").mkdir(parents=True)
    for f in TINY_FILES:
        shutil.copy(tiny / f, out / "tiny" / f)
    if (tiny / "onnx").exists():
        shutil.copytree(tiny / "onnx", out / "tiny" / "onnx")
    (out / "README.md").write_text(model_card(repo, github), encoding="utf-8")


def build_static_space(out, repo):
    """The browser demo as a static Space (free; Gradio Spaces on free CPU need a PRO account)."""
    site = OUT_DIR / "demo_site"
    if not (site / "index.html").exists():
        sys.exit("Build the demo first (build_demo.py).")
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(site, out)
    (out / "README.md").write_text(space_readme(repo, "static"), encoding="utf-8")


def build_space(out, repo):
    shutil.rmtree(out, ignore_errors=True)
    (out / "src").mkdir(parents=True)
    for f in SPACE_SRC:
        shutil.copy(ROOT / "src" / f, out / "src" / f)
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    (out / "app.py").write_text(app.replace('os.environ.get("MANGLISH_HUB_REPO", "")',
                                            f'os.environ.get("MANGLISH_HUB_REPO", "{repo}")'), encoding="utf-8")
    (out / "requirements.txt").write_text("\n".join(SPACE_REQUIREMENTS) + "\n", encoding="utf-8")
    (out / "README.md").write_text(space_readme(repo, "gradio"), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Hugging Face model repo id, e.g. yourname/manglish-translator")
    ap.add_argument("--github", default="https://github.com/swyam7980/manglish_dae")
    ap.add_argument("--parts", nargs="+", default=["hub", "space"])
    ap.add_argument("--space-sdk", choices=["static", "gradio"], default="static",
                    help="static: free browser demo (Tiny); gradio: both models on CPU, needs Hugging Face PRO")
    a = ap.parse_args(argv)
    if "hub" in a.parts:
        build_hub(OUT_DIR / "hub", a.repo, a.github)
    if "space" in a.parts:
        (build_static_space if a.space_sdk == "static" else build_space)(OUT_DIR / "space", a.repo)
    print(json.dumps({"hub": str(OUT_DIR / "hub"), "space": str(OUT_DIR / "space")}, indent=1))


if __name__ == "__main__":
    main()
