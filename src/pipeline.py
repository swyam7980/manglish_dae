"""Manglish -> English in two steps: our model writes Malayalam script, a dedicated translator turns it into English.
eval_translators.py picked the translator (results/ours_en/translators.md).

    python src/pipeline.py "naale mazha peyyumo" "ente veedu evideyaanu"
    python src/pipeline.py --first byt5 --file in.txt --out out.txt

Needs the isolated packages in .deps_mt (see requirements-translator.txt) and the translator's weights in the Hugging
Face cache, or --translator-dir with a downloaded copy.
"""
import argparse
from pathlib import Path

import deps_mt  # noqa: F401  (must come before transformers)
from translate import Translator as FirstStage
from translators import BEST, Translator as SecondStage


class EnglishPipeline:
    def __init__(self, first="tiny", first_dir=None, translator=BEST, translator_dir=None, device=None, beams=5):
        self.first = FirstStage(first, first_dir, device=device, guard=False)
        self.second = SecondStage(translator, device=device, beams=beams, path=translator_dir)

    def translate(self, lines, batch=16):
        """Returns (Malayalam script, English), each in input order."""
        script = self.first.translate(lines, "ml", batch)
        return script, self.second.translate(script, batch)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--out")
    ap.add_argument("--first", choices=["tiny", "byt5"], default="tiny")
    ap.add_argument("--first-dir")
    ap.add_argument("--translator", default=BEST)
    ap.add_argument("--translator-dir")
    ap.add_argument("--device", choices=["cuda", "cpu"])
    a = ap.parse_args(argv)
    lines = list(a.text)
    if a.file:
        lines += [l.strip() for l in Path(a.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        ap.error("give some text or --file")
    _, english = EnglishPipeline(a.first, a.first_dir, a.translator, a.translator_dir, a.device).translate(lines)
    if a.out:
        Path(a.out).write_text("\n".join(english) + "\n", encoding="utf-8")
    else:
        print("\n".join(english))
    return english


if __name__ == "__main__":
    main()
