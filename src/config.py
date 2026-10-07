"""Single source of truth for every setting shared by all three backbones."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
RES_DIR = ROOT / "results"

SEED = 42
VAL_FRACTION = 0.10
MIN_CHARS = 3
MAX_CHARS = 120          # keeps every sentence under 128 tokens/bytes, so ByT5 is not truncated
MAX_LEN = 128

P_REPEAT = 0.10          # per-character chance of elongation (1-3 extra copies)
P_DUPDEL = 0.04          # per-character chance of duplication or deletion
NOISE_SEED = 1234

EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-4

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

MODELS = {
    "mbart": {"base": "facebook/mbart-large-50-many-to-many-mmt", "targets": ["q_proj", "k_proj", "v_proj", "out_proj"]},
    "mt5":   {"base": "google/mt5-{size}", "targets": ["q", "k", "v", "o"]},
    "byt5":  {"base": "google/byt5-{size}", "targets": ["q", "k", "v", "o"]},
}
MT5_PREFIX = "reconstruct clean Manglish: "
