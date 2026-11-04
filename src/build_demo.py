"""Assemble the browser demo: demo/index.html -> outputs/demo_site/. The site holds no model; visitors load
Manglish-Tiny's int8 ONNX files (outputs/ours_tiny_en/onnx) from their own PC with the page's folder picker.

publish_pages.py --demo outputs/demo_site then puts it on the gh-pages branch under demo/, next to the results page.
--results adds a link to the results page (written to links.json).
"""
import argparse
import json
import shutil

from config import ROOT, OUT_DIR


def build(out, results=""):
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shutil.copy(ROOT / "demo" / "index.html", out / "index.html")
    (out / "links.json").write_text(json.dumps({"results": results}), encoding="utf-8")
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "demo_site"))
    ap.add_argument("--results", default="", help="URL of the results page (default: the parent folder)")
    a = ap.parse_args(argv)
    from pathlib import Path
    files = build(Path(a.out), a.results)
    print(f"demo site in {a.out}: {files}")
    return files


if __name__ == "__main__":
    main()
