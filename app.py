"""Local web interface for our models (Gradio): everything runs on this PC, from files on this PC.

    python app.py                      # http://127.0.0.1:7860
    python app.py --models tiny        # only the small first-stage model
    python app.py --translator-dir D:/models/indictrans2     # a downloaded copy of the translator

First-stage folders: outputs/ours_byt5_en/model (else outputs/ours_byt5/model) and outputs/ours_tiny_en (else
outputs/ours_tiny). "Manglish → English" writes Malayalam script with the chosen model, then translates it with the
second-stage translator (see src/pipeline.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import deps_mt  # noqa: E402,F401  (must come before gradio and transformers)
import gradio as gr  # noqa: E402

from translate import DEFAULT_DIRS, Translator  # noqa: E402
from translators import BEST, SYSTEMS, Translator as SecondStage  # noqa: E402

TASKS = {"Clean noisy Manglish": "clean", "Manglish → Malayalam script": "ml", "Manglish → English": "en"}
EXAMPLES = [["njaaan naaale varummm", "Clean noisy Manglish"],
            ["ente ammaye ariyooo", "Clean noisy Manglish"],
            ["njan tomorrow varum", "Manglish → Malayalam script"],
            ["njan nale varum", "Manglish → Malayalam script"],
            ["naale mazha peyyumo", "Manglish → English"],
            ["ente veedu evideyaanu ennu ariyaamo", "Manglish → English"]]
MAX_LINES = 200


def build(kinds, translator=BEST, translator_dir=None):
    first, second = {}, {}

    def get(kind):
        if kind not in first:
            first[kind] = Translator(kind, DEFAULT_DIRS[kind])
        return first[kind]

    def get_second():
        if "t" not in second:
            second["t"] = SecondStage(translator, path=translator_dir)
        return second["t"]

    def run(text, task_label, kind):
        lines = [l for l in (text or "").splitlines() if l.strip()][:MAX_LINES]
        if not lines:
            return "", ""
        task = TASKS[task_label]
        if task == "en":
            script = get(kind).translate(lines, "ml")
            return "\n".join(get_second().translate(script)), "\n".join(script)
        return "\n".join(get(kind).translate(lines, task)), ""

    with gr.Blocks(title="Manglish translator") as demo:
        gr.Markdown("# Manglish translator\nClean up noisy romanised Malayalam, write it in Malayalam script, or "
                    "translate it to English. One sentence per line. Everything runs on this PC.")
        with gr.Row():
            src = gr.Textbox(label="Manglish", lines=6, placeholder="njaaan naaale varummm")
            out = gr.Textbox(label="Output", lines=6)
        script = gr.Textbox(label="Malayalam script used for the English (first step)", lines=3)
        with gr.Row():
            task = gr.Radio(list(TASKS), value="Clean noisy Manglish", label="Task")
            kind = gr.Radio(kinds, value=kinds[0], label="First-stage model",
                            info="tiny: small, instant, best script scores. byt5: the large model, slower on CPU.")
        gr.Button("Translate", variant="primary").click(run, [src, task, kind], [out, script])
        gr.Examples(EXAMPLES, [src, task])
    return demo


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(DEFAULT_DIRS), default=["tiny", "byt5"])
    ap.add_argument("--translator", choices=list(SYSTEMS), default=BEST, help="it2_dist is a quarter of the size")
    ap.add_argument("--translator-dir", help="folder with the translator's weights (default: the Hugging Face cache)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args(argv)
    build(a.models, a.translator, a.translator_dir).launch(server_name=a.host, server_port=a.port)


if __name__ == "__main__":
    main()
