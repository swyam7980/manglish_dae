"""Manglish-Tiny: a small character-level encoder-decoder transformer trained from scratch.

It does the same two tasks as Manglish-ByT5 (a task token picks clean Manglish or Malayalam script) and is small
enough (~7M parameters, a few MB as int8 ONNX) to run inside a web page. Characters are the tokens, so the vocabulary
is just the letters, digits and punctuation of the training data plus the Malayalam block.
"""
import json
import math

import torch
from torch import nn

PAD, BOS, EOS, UNK = 0, 1, 2, 3
TASK_TOKENS = {"clean": 4, "ml": 5}
N_SPECIAL = 6
MAX_POS = 512
DEFAULTS = {"d_model": 256, "heads": 4, "enc_layers": 4, "dec_layers": 3, "ff": 1024, "dropout": 0.1}


class CharVocab:
    def __init__(self, chars):
        self.chars = list(chars)
        self.index = {c: i + N_SPECIAL for i, c in enumerate(self.chars)}

    @classmethod
    def from_texts(cls, texts):
        return cls(sorted({c for t in texts for c in t}))

    @classmethod
    def load(cls, path):
        return cls(json.loads(open(path, encoding="utf-8").read())["chars"])

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars, "pad": PAD, "bos": BOS, "eos": EOS, "unk": UNK,
                       "task_tokens": TASK_TOKENS, "n_special": N_SPECIAL}, f, ensure_ascii=False)

    def __len__(self):
        return N_SPECIAL + len(self.chars)

    def encode_src(self, text, task, max_len):
        return [TASK_TOKENS[task]] + [self.index.get(c, UNK) for c in text[: max_len - 1]]

    def encode_tgt(self, text, max_len):
        return [self.index.get(c, UNK) for c in text[: max_len - 1]] + [EOS]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS:
                break
            if i >= N_SPECIAL:
                out.append(self.chars[i - N_SPECIAL])
        return "".join(out)


NEG = -1e9  # additive mask value; finite so a fully padded row cannot turn into NaN


class Attention(nn.Module):
    """Plain multi-head attention. Written out (not nn.MultiheadAttention) so the ONNX export keeps every sequence
    length dynamic; the legacy exporter freezes nn.MultiheadAttention's reshape to the traced length."""
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(dropout)

    def split(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.h, self.dk).transpose(1, 2)

    def forward(self, x, mem, bias=None):
        q, k, v = self.split(self.q(x)), self.split(self.k(mem)), self.split(self.v(mem))
        att = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        if bias is not None:
            att = att + bias
        y = self.drop(att.softmax(-1)) @ v
        return self.o(y.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.h * self.dk))


class FeedForward(nn.Sequential):
    def __init__(self, d, ff, dropout):
        super().__init__(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))


class EncoderLayer(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att, self.ff, self.drop = Attention(d, heads, dropout), FeedForward(d, ff, dropout), nn.Dropout(dropout)

    def forward(self, x, bias):
        h = self.n1(x)
        x = x + self.drop(self.att(h, h, bias))
        return x + self.drop(self.ff(self.n2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_att, self.cross = Attention(d, heads, dropout), Attention(d, heads, dropout)
        self.ff, self.drop = FeedForward(d, ff, dropout), nn.Dropout(dropout)

    def forward(self, x, mem, self_bias, mem_bias):
        h = self.n1(x)
        x = x + self.drop(self.self_att(h, h, self_bias))
        x = x + self.drop(self.cross(self.n2(x), mem, mem_bias))
        return x + self.drop(self.ff(self.n3(x)))


def pad_bias(pad):
    """(B, S) bool padding mask -> additive (B, 1, 1, S) bias, or None."""
    return None if pad is None else pad[:, None, None, :].float() * NEG


class TinySeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=256, heads=4, enc_layers=4, dec_layers=3, ff=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_POS, d_model)
        self.enc_layers = nn.ModuleList(EncoderLayer(d_model, heads, ff, dropout) for _ in range(enc_layers))
        self.dec_layers = nn.ModuleList(DecoderLayer(d_model, heads, ff, dropout) for _ in range(dec_layers))
        self.enc_norm, self.dec_norm = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.tok.weight  # tied input/output embeddings
        # small init: embed() scales by sqrt(d_model) and the tied output layer must start with small logits
        nn.init.normal_(self.tok.weight, std=d_model ** -0.5)
        with torch.no_grad():
            self.tok.weight[PAD].zero_()

    def embed(self, ids):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        return self.emb_drop(self.tok(ids) * math.sqrt(self.d_model) + self.pos(pos))

    def encode(self, src, src_pad=None):
        x, bias = self.embed(src), pad_bias(src_pad)
        for layer in self.enc_layers:
            x = layer(x, bias)
        return self.enc_norm(x)

    def decode(self, tgt_in, memory, src_pad=None):
        # Target padding needs no mask: it only follows real tokens, and the causal mask hides later positions.
        n = tgt_in.shape[1]
        causal = torch.triu(torch.ones(n, n, device=tgt_in.device), diagonal=1)[None, None] * NEG
        x, mem_bias = self.embed(tgt_in), pad_bias(src_pad)
        for layer in self.dec_layers:
            x = layer(x, memory, causal, mem_bias)
        return self.out(self.dec_norm(x))

    def forward(self, src, tgt_in):
        src_pad = src.eq(PAD)
        return self.decode(tgt_in, self.encode(src, src_pad), src_pad)

    @torch.no_grad()
    def greedy(self, src, max_len=400):
        """Batch greedy decoding (no cache: the model is small and outputs are short)."""
        src_pad = src.eq(PAD)
        memory = self.encode(src, src_pad)
        ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            nxt = self.decode(ys, memory, src_pad)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], 1)
            done |= nxt.eq(EOS)
            if done.all():
                break
        return ys[:, 1:]


class EncoderOnly(nn.Module):
    """ONNX export wrapper: src ids -> encoder memory (batch of one, no padding)."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, src):
        return self.m.encode(src)


class DecoderStep(nn.Module):
    """ONNX export wrapper: (target prefix, memory) -> logits for the next character."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, tgt_in, memory):
        return self.m.decode(tgt_in, memory)[:, -1]


def pad_batch(seqs, value=PAD):
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (n - len(s)) for s in seqs], dtype=torch.long)


def load_tiny(folder, device="cpu"):
    """(model, vocab, config) from a saved folder: model.pt + vocab.json + config.json."""
    from pathlib import Path
    folder = Path(folder)
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    vocab = CharVocab.load(folder / "vocab.json")
    model = TinySeq2Seq(len(vocab), **cfg["arch"])
    model.load_state_dict(torch.load(folder / "model.pt", map_location=device))
    return model.to(device).eval(), vocab, cfg
