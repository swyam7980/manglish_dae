"""CER/WER (primary) and BLEU/chrF (secondary) between reconstructions and clean targets."""
import jiwer
import sacrebleu


def score(preds, refs):
    """All four metrics on a corpus. CER/WER are percentages (lower is better); BLEU/chrF are 0-100 (higher is better).
    An empty prediction is scored as an empty string (jiwer counts it as all-deletions)."""
    assert len(preds) == len(refs) and refs, "preds and refs must be the same non-zero length"
    return {
        "CER": round(jiwer.cer(refs, preds) * 100, 3),
        "WER": round(jiwer.wer(refs, preds) * 100, 3),
        "BLEU": round(sacrebleu.corpus_bleu(preds, [refs]).score, 3),
        "chrF": round(sacrebleu.corpus_chrf(preds, [refs]).score, 3),
    }


def sentence_cer(pred, ref):
    """Per-sentence CER in percent, used to find the worst predictions."""
    return jiwer.cer(ref, pred) * 100 if pred else 100.0
