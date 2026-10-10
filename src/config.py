"""Single source of truth for every setting shared by all three backbones."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# MDAE_WORKDIR relocates data/outputs/results (used by the tests so they never touch real data).
WORK = Path(os.environ.get("MDAE_WORKDIR", ROOT))
RAW_DIR = WORK / "data" / "raw"
PROC_DIR = WORK / "data" / "processed"
OUT_DIR = WORK / "outputs"
RES_DIR = WORK / "results"

# ---- data -------------------------------------------------------------------------------------
SEED = 42                # train/val split and training seed
VAL_FRACTION = 0.10
MIN_CHARS = 3
MAX_CHARS = 120          # ByT5 works on bytes; Latin text keeps chars ~= bytes, so nothing hits 128
MAX_LEN = 128
MIN_ASCII_FRACTION = 0.95

# ---- noise (proposal: "fixed" probabilities, values are our defaults) -------------------------
P_REPEAT = 0.10          # per-character chance of elongation (1-3 extra copies)
P_DUPDEL = 0.04          # per-character chance of duplication or deletion
NOISE_SEED_TRAIN = 1
NOISE_SEED_VAL = 2

# ---- training budget, identical for every model ------------------------------------------------
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8           # effective batch 16
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM
LR = 2e-4
WARMUP_RATIO = 0.03

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ---- backbones ---------------------------------------------------------------------------------
# mBART-50 only exists in one size; mT5/ByT5 take --size small|base and must use the same tier.
MODELS = {
    "mbart": {"base": "facebook/mbart-large-50-many-to-many-mmt", "targets": ["q_proj", "k_proj", "v_proj", "out_proj"]},
    "mt5":   {"base": "google/mt5-{size}", "targets": ["q", "k", "v", "o"]},
    "byt5":  {"base": "google/byt5-{size}", "targets": ["q", "k", "v", "o"]},
}
MT5_PREFIX = "reconstruct clean Manglish: "
# T5-family models overflow in fp16, so they train in bf16 (fp32 if the GPU lacks bf16).
# mBART keeps fp16 as in the proposal. This is a deliberate, documented deviation.
FP16_UNSAFE = {"mt5", "byt5"}
