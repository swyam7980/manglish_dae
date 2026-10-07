"""Builds tiny randomly-initialised mBART/mT5/ByT5 checkpoints locally, so the full pipeline can be
exercised offline. The weights are meaningless; this only tests wiring (LoRA targets, tokenizers, save/reload)."""
import sentencepiece as spm
from transformers import (ByT5Tokenizer, MBart50Tokenizer, MBartConfig, MBartForConditionalGeneration,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)


def _train_spm(corpus, prefix, vocab_size, **kw):
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(prefix), vocab_size=vocab_size,
                                   model_type="unigram", character_coverage=1.0, **kw)
    return str(prefix) + ".model"


def build(root, texts):
    corpus = root / "corpus.txt"
    corpus.write_text("\n".join(texts), encoding="utf-8")
    paths = {}

    mt5_dir = root / "tiny_mt5"
    sp = _train_spm(corpus, root / "mt5_spm", 100, pad_id=0, eos_id=1, unk_id=2, bos_id=-1)
    tok = T5Tokenizer(vocab_file=sp, extra_ids=0)
    cfg = T5Config(vocab_size=len(tok), d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4,
                   decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    T5ForConditionalGeneration(cfg).save_pretrained(mt5_dir)
    tok.save_pretrained(mt5_dir)
    paths["mt5"] = str(mt5_dir)

    byt5_dir = root / "tiny_byt5"
    tok = ByT5Tokenizer(extra_ids=0)
    cfg = T5Config(vocab_size=len(tok) + 5, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4,
                   decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    T5ForConditionalGeneration(cfg).save_pretrained(byt5_dir)
    tok.save_pretrained(byt5_dir)
    paths["byt5"] = str(byt5_dir)

    mbart_dir = root / "tiny_mbart"
    sp = _train_spm(corpus, root / "mbart_spm", 100)
    tok = MBart50Tokenizer(vocab_file=sp, src_lang="en_XX", tgt_lang="en_XX")
    cfg = MBartConfig(vocab_size=len(tok) + 5, d_model=32, encoder_layers=2, decoder_layers=2, encoder_attention_heads=4,
                      decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64, max_position_embeddings=256,
                      pad_token_id=1, bos_token_id=0, eos_token_id=2, decoder_start_token_id=2)
    MBartForConditionalGeneration(cfg).save_pretrained(mbart_dir)
    tok.save_pretrained(mbart_dir)
    paths["mbart"] = str(mbart_dir)
    return paths
