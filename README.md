# Manglish denoising adaptation: mBART-50 vs mT5 vs ByT5

Compares three pretrained multilingual encoder-decoder models on reconstructing clean Manglish (Malayalam in
Latin script) from synthetically corrupted Manglish, using LoRA with an identical budget for each model.
Claim is about denoising only; there is no translation claim. Methodology follows
`Manglish_DAE_Comparison_Research_Proposal.docx`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build first (Windows/NVIDIA)
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Put the SeLLaI file in `data/raw/` (see `data/README.md`).

## Run, phase by phase (inside `src/`)

| Phase | Command | What to check |
|---|---|---|
| B data | `python data_prep.py` | class counts, filter funnel, train/val sizes, sample pairs |
| C baseline | `python evaluate.py --baseline` | CER clearly above 0 and in line with the noise rates |
| D lengths | `python check_lengths.py --size base` | essentially 0% over 128 for every tokenizer |
| E smoke | `python smoke_test.py --size base` | trainable < 5%, loss falls, non-empty output, peak GPU memory |
| F train | `python train.py --model mbart` then `mt5`, `byt5` (`--size small\|base`, `--grad-ckpt` if VRAM is tight) | loss falls smoothly, never NaN |
| G eval | `python evaluate.py --model mbart` (same for the others) | every model beats the baseline on CER and WER |
| H report | `python compare.py` and `python analyze.py` | `results/summary.{csv,md}`, `comparison.png`, `analysis_*.md` |

Or everything in order: `python run_all.py --size base [--smoke]`.

## Settings (all in `src/config.py`, shared by every model)

90/10 split, seed 42; max length 128 (sentences capped at 120 chars so ByT5's bytes are not truncated);
3 epochs, batch 2 x accumulation 8, LR 2e-4; LoRA r=16, alpha=32, dropout 0.05 on the attention projections;
greedy decoding. Noise: repetition p=0.10, duplication/deletion p=0.04, seeded per sentence (train and val use different seeds).

Choices the proposal does not fix, state these in the report:

- Noise probabilities 0.10 / 0.04, mBART uses `en_XX` for source and target, mT5 gets the prefix
  `reconstruct clean Manglish: `, sentences are lower-cased.
- **Precision deviation:** mBART trains in FP16 as proposed, but mT5 and ByT5 use bf16 (fp32 if the GPU has no bf16),
  because T5-family models overflow in fp16.
- mT5 and ByT5 must use the same size tier (`--size`).

## Tests

`pytest tests` runs unit tests (noise, metrics, data prep) and an offline end-to-end wiring test on tiny randomly
initialised models (no downloads). They verify the plumbing, not model quality; real results need the real weights.

## Limits of the study

Synthetic noise only, one seed, one split, no translation claim. If the models are within seed-to-seed variation,
report that instead of declaring a winner.
