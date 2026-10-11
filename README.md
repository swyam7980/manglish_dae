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

If a smoke test looks wrong (loss flat or huge, garbage output), run `python diagnose.py --model mt5` (or `mbart`, `byt5`).
It prints the plain pretrained model's loss on noisy -> clean pairs in the configured precision and in fp32, with a few
generations. `set MDAE_FORCE_FP32=1` (Windows; `export` elsewhere) forces fp32 for training and evaluation. Any run that
uses it is a precision deviation to report.

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

## More data: three sources, one combined test set

Beyond the original Manglish class (2,934 train / 325 validation), `extra_data.py` builds a larger training file and a
combined multi-source test file from three sources:

| Source | Licence | Notes |
|---|---|---|
| SeLLaI Manglish class | CC BY 4.0 | original data, validation set unchanged |
| SeLLaI Mixed-Code class | CC BY 4.0 | Latin-script code-mixed captions, same file |
| Dakshina Malayalam romanised sentences | CC BY-SA 4.0 | `dakshina_dataset_v1.0.tar` from Google, about 2 GB; formal Wikipedia text; the loader reads only the Malayalam romanised files straight from the tar |

```bash
cd src
python extra_data.py --tag all --sellai-label Mixed-Code --dakshina ../data/raw/dakshina_dataset_v1.0.tar
```

This writes `data/processed/train_plus_all.jsonl` (18,548 sentences) and `test_all.jsonl` (2,059 sentences, 10% of each
extra source plus the original 325 validation sentences) and a `.stats.json` with the sanitiser's counts. No test
sentence is in any training file, and duplicates across sources (exact or ignoring spaces and punctuation) are removed.

**Sanitiser** (`sanitize.py`): repairs Unicode forms, HTML entities, control and zero-width characters, curly quotes,
emoji, URLs and mentions; drops non-Latin script, truncated fragments, elongated letters, digit- or symbol-heavy
lines, gibberish tokens, repeated words, single words and over-long lines (these are split at clause boundaries
first). Every dropped line is counted by reason, with examples, in the stats file. The original Manglish class is
audited but not re-filtered, so the 325-sentence validation set and all earlier results stay exactly comparable.

Train on it with `python train.py --model byt5 --train-file train_plus_all.jsonl --auto-batch --resume --tag-suffix _plusall`.

## Memory fail-safes

More data does not raise GPU memory (the batch does), but the tools protect a 6 GB card anyway; the effective batch
stays at 16 in every case:

- `--auto-batch` runs one forward and backward pass on the longest training examples and picks the largest micro-batch
  that fits the free VRAM with headroom (`memory.py`). The probe prints its predicted peaks; they matched real peaks.
- An out-of-memory error during training halves the micro-batch, rebuilds the model and retries (up to four times).
- `--resume` writes a checkpoint every 200 steps and continues from the latest one after a crash or power cut.
- Generation (`generation.py`) halves its batch on out-of-memory too.

## Extra seeds, deviations and the follow-up studies

- `train.py --seed N` changes only the training seed; the split and the validation noise stay fixed. Pair it with
  `--tag-suffix _sN` so each run gets its own folder. `src/run_seeds.sh` runs seeds 43 and 44 for all three backbones;
  `run_more.py --phases seeds` adds seeds 45 and 46 for mBART-50 and ByT5.
- `train.py --lr X` overrides the learning rate. Any run that uses it is a deviation and must be reported as one
  (the mT5 runs at 1e-3 are labelled that way; the specified 2e-4 runs are kept alongside them).
- `eval_combined.py` scores every trained seed on the combined test set, overall and per source.
- `robustness.py` scores every adapter on noise it was not trained on (2 and 3 times the rates, vowel dropping,
  phonetic spelling variants) without retraining.
- `decode_study.py` compares greedy and 4-beam decoding, with and without a length guard that replaces outputs under
  60% of the input length with a rule-based cleaning, and scores the rule-based cleaner (collapse letter runs) alone.
- `compare.py` ranks the seed-42 runs only; `seeds.py` aggregates every seed (mean and SD) and runs paired bootstraps on
  the CER difference between models, on the validation set and on the combined test set.
- `run_more.py` runs all of the above as one resumable queue in phases (`seeds data train combined studies report`);
  finished steps are skipped and a failing step does not stop the rest:

```bash
cd src
python run_more.py --phases seeds
python run_more.py --phases data train combined studies report --dakshina ../data/raw/dakshina_dataset_v1.0.tar
```

- `profile_project.py` records the dataset sizes and sentence lengths and the time and memory of every training run
  (`results/data/dataset_profile.json`, `results/run_times.json`); pass the console logs of `run_more.py` with
  `--queue-log` to include the step times of the longer evaluation and study steps.
- `collect_results.py` copies the metrics, predictions and summaries into `results/` (committed, 4 MB), and
  `build_report.py` turns them into the static page `report/index.html` (prose lives in `report/copy.json`). Open the
  file in a browser; it has no server and no dependencies beyond Google Fonts. Adapters (`outputs/`) and datasets
  (`data/`) are not committed.

## Publishing the results page (GitHub Pages)

The page is one self-contained file, `report/index.html` (its only outside request is Google Fonts). To serve it:

```bash
python src/publish_pages.py     # builds the local gh-pages branch: index.html and .nojekyll only
git push origin gh-pages
```

Then on GitHub open Settings, Pages, choose "Deploy from a branch", pick `gh-pages` and the `/ (root)` folder, and
save. The site appears at `https://<user>.github.io/<repository>/`. Run the two commands again after rebuilding the
page; an unchanged page adds no commit. The branch is built with git plumbing, so your working folder and current
branch are not touched.

## Tests

`pytest tests` runs unit tests (noise, metrics, data prep) and an offline end-to-end wiring test on tiny randomly
initialised models (no downloads). They verify the plumbing, not model quality; real results need the real weights.

## Limits of the study

Synthetic noise only (plus two unseen synthetic types in the robustness test), one split per protocol, no translation
claim. The small-data models ran five seeds; the models trained on the combined data ran once, so their ranking
rests on a large test set but not on seed variation. Two of the three datasets are caption-style; Dakshina is formal
Wikipedia text. If models are within seed-to-seed variation, report that instead of declaring a winner.
