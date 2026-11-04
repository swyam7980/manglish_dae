"""Malayalam-script -> English translators for the two-model pipeline (Manglish -> script -> English).

IndicTrans2's custom code needs transformers 4.x while the project runs 5.x, so run anything here with an isolated
folder first on the path:  pip install --target .deps_mt transformers==4.46.3 tokenizers==0.20.3 huggingface_hub==0.26.2
sacremoses indic-nlp-library-itt requests regex joblib click  ; set PYTHONPATH=.deps_mt
IndicProcessor below ports the steps of IndicTransToolkit 1.1.1 (a Cython package) that Malayalam-to-English uses.
"""
import unicodedata

import torch

from memory import run_with_backoff

SYSTEMS = {"it2_dist": ("indictrans", "ai4bharat/indictrans2-indic-en-dist-200M"),
           "it2_1b": ("indictrans", "ai4bharat/indictrans2-indic-en-1B"),
           "opus": ("marian", "Helsinki-NLP/opus-mt-ml-en"),
           "madlad": ("madlad", "google/madlad400-3b-mt"),
           "nllb": ("nllb", "facebook/nllb-200-distilled-600M")}
LICENCES = {"it2_dist": "MIT", "it2_1b": "MIT", "opus": "Apache-2.0", "madlad": "Apache-2.0", "nllb": "CC BY-NC 4.0"}
BEST = "it2_1b"  # highest scores on IN22 among the redistributable translators (results/ours_en/translators.md)


class IndicProcessor:
    """Punctuation and numeral normalisation, placeholders for numbers and URLs, Indic tokenising, and transliteration
    of the Malayalam to Devanagari, which is the form IndicTrans2 was trained on."""

    def __init__(self):
        import regex as re
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        from indicnlp.tokenize import indic_tokenize
        from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
        from sacremoses import MosesDetokenizer
        self.re, self.tok, self.xl = re, indic_tokenize, UnicodeIndicTransliterator()
        self.norm = IndicNormalizerFactory().get_normalizer("ml")
        self.detok = MosesDetokenizer(lang="en")
        self.punc = [(r"\r", ""), (r"\(\s*", "("), (r"\s*\)", ")"), (r"\s:\s?", ":"), (r"\s;\s?", ";"),
                     (r"[`´‘‚’]", "'"), (r"[„“”«»]", '"'), (r"[–—]", "-"), (r" %", "%"),
                     (r" ([?!;])", r"\1"), (r"[ ]{2,}", " "), (r"\) ([\.!:?;,])", r")\1"), (r"(\d) %", r"\1%"),
                     (r"\"([,\.]+)", r'\1"'), (r"(\d) (\d)", r"\1.\2")]
        self.patterns = [("email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"),
                         ("url", r"\b(?<![\w/.])(?:(?:https?|ftp)://)?(?:(?:[\w-]+\.)+(?!\.))(?:[\w/\-?#&=%.]+)+(?!\.\w+)\b"),
                         ("num", r"(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|\d+[-\/.,:']\d+[-\/.,:'+]\d+(?:\.\d+)?"
                                 r"|\d+[-\/.:'+]\d+(?:\.\d+)?)"),
                         ("other", r"[A-Za-z0-9]*[#|@]\w+")]

    def _placeholders(self, text):
        found, n = {}, 1
        for kind, pat in self.patterns:
            for m in sorted(set(self.re.findall(pat, text))):
                if (kind == "url" and len(m.replace(".", "")) < 4) or (
                        kind == "num" and len(m.replace(" ", "").replace(".", "").replace(":", "")) < 4):
                    continue
                for form in (f"<ID{n}>", f"< ID{n} >", f"[ID{n}]", f"[ ID{n} ]", f"[ID {n}]", f"<ID{n}]", f"< ID{n}]",
                             f"<ID{n} ]", f"<id{n}>", f"< id{n} >", f"[id{n}]", f"[ id{n} ]", f"[id {n}]", f"<id{n}]",
                             f"< id{n}]", f"<id{n} ]"):
                    found[form] = m
                text = text.replace(m, f"<ID{n}>")
                n += 1
        return self.re.sub(r"\s+", " ", text).replace(">/", ">").replace("]/", "]"), found

    def pre(self, sent):
        for pat, rep in self.punc:
            sent = self.re.sub(pat, rep, sent)
        sent = "".join(str(unicodedata.digit(c)) if unicodedata.category(c) == "Nd" else c for c in sent.strip())
        sent, found = self._placeholders(sent)
        joined = " ".join(self.tok.trivial_tokenize(self.norm.normalize(sent.strip()), "ml"))
        xlated = self.xl.transliterate(joined, "ml", "hi").replace(" ् ", "्").strip()
        return f"mal_Mlym eng_Latn {xlated}", found

    def post(self, sent, found):
        for k, v in found.items():
            sent = sent.replace(k, v)
        return self.detok.detokenize(sent.split(" "))


class Translator:
    def __init__(self, name=BEST, device=None, beams=5, max_len=256, path=None):
        self.name, (self.kind, repo) = name, SYSTEMS[name]
        repo = path or repo
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.beams, self.max_len = beams, max_len
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        remote = self.kind == "indictrans"
        extra = {"src_lang": "mal_Mlym"} if self.kind == "nllb" else {}
        self.tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=remote, **extra)
        kw = {"attn_implementation": "eager"} if remote else {}
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(repo, trust_remote_code=remote, torch_dtype=dtype, **kw)
        self.model = self.model.to(self.device).eval()
        self.proc = IndicProcessor() if remote else None

    def _gen_kwargs(self):
        kw = {"num_beams": self.beams, "max_length": self.max_len, "do_sample": False}
        if self.kind == "indictrans":
            kw.update(use_cache=True, min_length=0, num_return_sequences=1)
        if self.kind == "nllb":
            kw["forced_bos_token_id"] = self.tok.convert_tokens_to_ids("eng_Latn")
        return kw

    def _chunk(self, chunk):
        if self.kind == "indictrans":
            pairs = [self.proc.pre(s) for s in chunk]
            text = [p[0] for p in pairs]
        else:
            text = [f"<2en> {s}" for s in chunk] if self.kind == "madlad" else list(chunk)
        x = self.tok(text, truncation=True, padding="longest", max_length=self.max_len, return_tensors="pt",
                     return_attention_mask=True).to(self.device)
        with torch.no_grad():
            y = self.model.generate(**x, **self._gen_kwargs())
        if self.kind == "indictrans":
            with self.tok.as_target_tokenizer():
                out = self.tok.batch_decode(y.detach().cpu().tolist(), skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True)
            return [self.proc.post(s, p[1]).strip() for s, p in zip(out, pairs)]
        return [s.strip() for s in self.tok.batch_decode(y, skip_special_tokens=True)]

    def translate(self, texts, batch=16):
        texts = [str(t) if str(t).strip() else "." for t in texts]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths batch together: less padding
        done = run_with_backoff(self._chunk, [texts[i] for i in order], batch)[0]
        out = [""] * len(texts)
        for i, s in zip(order, done):
            out[i] = s
        return out
