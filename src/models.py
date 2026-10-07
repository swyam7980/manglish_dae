"""Backbone + tokenizer + LoRA wiring, identical rank/alpha/dropout for all three models."""
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import MODELS, LORA_R, LORA_ALPHA, LORA_DROPOUT, MT5_PREFIX


def model_name(key, size):
    return MODELS[key]["base"].format(size=size)


def pick_dtype(key):
    """mT5 overflows in fp16, so use bf16 when the GPU supports it, else fp32."""
    if not torch.cuda.is_available():
        return torch.float32, {}
    if key == "mt5":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, {"bf16": True}
        return torch.float32, {}
    return torch.float32, {"fp16": True}


def load_tokenizer(key, size):
    name = model_name(key, size)
    if key == "mbart":
        return AutoTokenizer.from_pretrained(name, src_lang="en_XX", tgt_lang="en_XX")
    return AutoTokenizer.from_pretrained(name)


def input_prefix(key):
    return MT5_PREFIX if key == "mt5" else ""


def gen_kwargs(key, tok):
    if key == "mbart":
        return {"forced_bos_token_id": tok.convert_tokens_to_ids("en_XX")}
    return {}


def build_lora_model(key, size):
    dtype, _ = pick_dtype(key)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name(key, size), torch_dtype=dtype)
    cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
                     lora_dropout=LORA_DROPOUT, target_modules=MODELS[key]["targets"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model
