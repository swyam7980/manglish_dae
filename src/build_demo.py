"""Assemble the browser demo: demo/index.html + the int8 ONNX files of Manglish-Tiny -> outputs/demo_site/.

publish_pages.py --demo outputs/demo_site then puts it on the gh-pages branch under demo/, next to the results page.
--results, --space and --weights add links to the results page, the Hugging Face Space and the model repo
(written to links.json).
"""
import argparse
import json
import shutil
import sys

from config import ROOT, OUT_DIR

MODEL_FILES = ["encoder.int8.onnx", "decoder.int8.onnx", "vocab.json", "config.json"]


def build(onnx_dir, out, space="", weights="", results=""):
    missing = [f for f in MODEL_FILES if not (onnx_dir / f).exists()]
    if missing:
        sys.exit(f"Missing {missing} in {onnx_dir}. Run train_tiny.py and export_tiny.py first.")
    shutil.rmtree(out, ignore_errors=True)
    (out / "model").mkdir(parents=True)
    shutil.copy(ROOT / "demo" / "index.html", out / "index.html")
    for f in MODEL_FILES:
        shutil.copy(onnx_dir / f, out / "model" / f)
    (out / "links.json").write_text(json.dumps({"space": space, "weights": weights, "results": results}), encoding="utf-8")
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())


def main(argv=None):
    ap = argparse.ArgumentParser()
    tiny = next((t for t in ("ours_tiny_en", "ours_tiny") if (OUT_DIR / t / "onnx" / "export_report.json").exists()),
                "ours_tiny")
    ap.add_argument("--onnx-dir", default=str(OUT_DIR / tiny / "onnx"), help="default: the English model's when exported")
    ap.add_argument("--out", default=str(OUT_DIR / "demo_site"))
    ap.add_argument("--space", default="", help="URL of the Hugging Face Space hosting this demo")
    ap.add_argument("--results", default="", help="URL of the results page (default: the parent folder)")
    ap.add_argument("--weights", default="", help="URL of the Hugging Face model repo")
    a = ap.parse_args(argv)
    from pathlib import Path
    files = build(Path(a.onnx_dir), Path(a.out), a.space, a.weights, a.results)
    print(f"demo site in {a.out}: {files}")
    return files


if __name__ == "__main__":
    main()
