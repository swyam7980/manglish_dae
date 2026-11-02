"""Local web interface for our models (Gradio). Also the entry point of the Hugging Face Space.

    python app.py                      # http://127.0.0.1:7860
    python app.py --models tiny        # only the small model (no 1.2 GB download)

Model folders: outputs/ours_byt5_en/model (else outputs/ours_byt5/model) and outputs/ours_tiny_en (else
outputs/ours_tiny). When they are missing (as on a fresh clone or the Space) and --hub-repo is given, the weights are
downloaded from the Hugging Face Hub.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr  # noqa: E402

from translate import DEFAULT_DIRS, Translator  # noqa: E402

HUB_REPO = os.environ.get("MANGLISH_HUB_REPO", "")
HUB_SUBDIRS = {"byt5": "byt5", "tiny": "tiny"}
TASKS = {"Clean noisy Manglish": "clean", "Manglish → Malayalam script": "ml", "Manglish → English": "en"}
EXAMPLES = [["njaaan naaale varummm", "Clean noisy Manglish"],
            ["ente ammaye ariyooo", "Clean noisy Manglish"],
            ["njan tomorrow varum", "Manglish → Malayalam script"],
            ["njan nale varum", "Manglish → Malayalam script"],
            ["njan tomorrow varum", "Manglish → English"],
            ["ente veedu evideyaanu ennu ariyaamo", "Manglish → English"]]
MAX_LINES = 200


def model_dir(kind, hub_repo):
    local = DEFAULT_DIRS[kind]
    if (local / ("config.json" if kind == "byt5" else "model.pt")).exists() or not hub_repo:
        return local
    from huggingface_hub import snapshot_download
    root = snapshot_download(hub_repo, allow_patterns=[f"{HUB_SUBDIRS[kind]}/*"])
    return Path(root) / HUB_SUBDIRS[kind]


def build(kinds, hub_repo):
    loaded = {}

    def get(kind):
        if kind not in loaded:
            loaded[kind] = Translator(kind, model_dir(kind, hub_repo))
        return loaded[kind]

    def run(text, task_label, kind):
        lines = [l for l in (text or "").splitlines() if l.strip()][:MAX_LINES]
        if not lines:
            return ""
        try:
            return "\n".join(get(kind).translate(lines, TASKS[task_label]))
        except ValueError as e:  # e.g. a model trained before the English task
            raise gr.Error(str(e))

    with gr.Blocks(title="Manglish translator") as demo:
        gr.Markdown("# Manglish translator\nClean up noisy romanised Malayalam, write it in Malayalam script, or "
                    "translate it to English. One sentence per line.")
        with gr.Row():
            src = gr.Textbox(label="Manglish", lines=6, placeholder="njaaan naaale varummm")
            out = gr.Textbox(label="Output", lines=6)
        with gr.Row():
            task = gr.Radio(list(TASKS), value="Clean noisy Manglish", label="Task")
            kind = gr.Radio(kinds, value=kinds[0], label="Model",
                            info="byt5: best quality, slower on CPU. tiny: small and instant.")
        gr.Button("Translate", variant="primary").click(run, [src, task, kind], out)
        gr.Examples(EXAMPLES, [src, task])
    return demo


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(DEFAULT_DIRS), default=list(DEFAULT_DIRS))
    ap.add_argument("--hub-repo", default=HUB_REPO, help="Hugging Face model repo to download missing weights from")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args(argv)
    build(a.models, a.hub_repo).launch(server_name=a.host, server_port=a.port)


if __name__ == "__main__":
    main()
