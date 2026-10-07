"""Backbone + tokenizer + LoRA wiring, identical rank/alpha/dropout for all three models."""
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import MODELS, LORA_R, LORA_ALPHA, LORA_DROPOUT, MT5_PREFIX, FP16_UNSAFE


def model_name(key, size, base_path=None):
    """Hub id, or a local directory when base_path is given (offline machines, Kaggle input datasets)."""
    return base_path or MODELS[key]["base"].format(size=size)


def run_tag(key, size):
    """Folder name for a run's outputs. mBART has a single size."""
    return f"{key}_{size if key != 'mbart' else 'large'}"


def pick_dtype(key):
    """Returns (weights dtype, Trainer precision flags). T5-family overflows in fp16 -> bf16, else fp32."""
    if not torch.cuda.is_available():
        return torch.float32, {}
    if key in FP16_UNSAFE:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, {"bf16": True}
        return torch.float32, {}
    return torch.float32, {"fp16": True}


def load_tokenizer(key, size, base_path=None):
    name = model_name(key, size, base_path)
    if key == "mbart":
        return AutoTokenizer.from_pretrained(name, src_lang="en_XX", tgt_lang="en_XX")
    return AutoTokenizer.from_pretrained(name)


def input_prefix(key):
    return MT5_PREFIX if key == "mt5" else ""


def gen_kwargs(key, tok):
    if key == "mbart":
        # Latin-script Manglish has no mBART-50 language code; en_XX is used for source and target.
        return {"forced_bos_token_id": tok.convert_tokens_to_ids("en_XX")}
    return {}


def load_base(key, size, base_path=None):
    dtype, _ = pick_dtype(key)
    return AutoModelForSeq2SeqLM.from_pretrained(model_name(key, size, base_path), dtype=dtype)


def build_lora_model(key, size, base_path=None, grad_ckpt=False):
    model = load_base(key, size, base_path)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
                     lora_dropout=LORA_DROPOUT, target_modules=MODELS[key]["targets"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
