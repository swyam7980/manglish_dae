"""CER/WER (primary) and BLEU/chrF (secondary) between reconstructions and clean targets."""
import jiwer
import sacrebleu


def score(preds, refs):
    preds = [p if p.strip() else "-" for p in preds]
    return {
        "CER": round(jiwer.cer(refs, preds) * 100, 3),
        "WER": round(jiwer.wer(refs, preds) * 100, 3),
        "BLEU": round(sacrebleu.corpus_bleu(preds, [refs]).score, 3),
        "chrF": round(sacrebleu.corpus_chrf(preds, [refs]).score, 3),
    }
