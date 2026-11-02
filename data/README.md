# Data

Dataset: **SeLLaI Mal-Eng** (Mendeley Data, DOI [10.17632/5p4zbpy8wz.1](https://doi.org/10.17632/5p4zbpy8wz.1)),
licence **CC BY 4.0**. Cite it as given on the Mendeley page when you publish results.

1. Open the DOI page and press **Download All**.
2. Put the data file (`.csv`, `.xlsx` or `.tsv`) in `data/raw/`. Keep the licence/citation text next to it.
3. Run `python src/data_prep.py`. It prints the class counts, so check the exact class spelling
   (expected `Manglish`; pass `--label` if it differs), the number kept after every filter, and sample pairs.

`data/raw/` and `data/processed/` are git-ignored. Do not commit the dataset.

## English task (`src/en_data.py`)

Malayalam-English pairs, all openly licensed. BPCC and IN22 are gated on Hugging Face: open each dataset page while
logged in, request access (granted automatically), then download with your own login (`hf auth login`).

| source | licence | put it in |
|---|---|---|
| [ai4bharat/BPCC](https://huggingface.co/datasets/ai4bharat/BPCC): `{bpcc-seed-latest,ilci,massive,wiki,daily}/mal_Mlym.tsv` | CC0 (seed, ILCI, Massive), CC BY 4.0 (Wiki, Daily) | `data/raw/bpcc/<subset>/mal_Mlym.tsv` |
| [PMIndia v1](https://data.statmt.org/pmindia/v1/parallel/) `pmindia.v1.ml-en.tsv` | CC BY 4.0 | `data/raw/pmindia/` |
| OPUS [Wikimedia v20230407](https://object.pouta.csc.fi/OPUS-wikimedia/v20230407/moses/en-ml.txt.zip) and [Tatoeba v2023-04-12](https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-ml.txt.zip) | CC BY-SA 4.0, CC BY 2.0 FR | unzip to `data/raw/opus/{wikimedia,Tatoeba}/` |
| [ai4bharat/IN22-Conv](https://huggingface.co/datasets/ai4bharat/IN22-Conv) (test only; its `IN22_benchmark.tar.gz` holds Conv and Gen) | CC BY 4.0 | `data/raw/IN22-Conv/` |

```bash
for s in bpcc-seed-latest ilci massive wiki daily; do
  hf download ai4bharat/BPCC $s/mal_Mlym.tsv --repo-type dataset --local-dir data/raw/bpcc
done
hf download ai4bharat/IN22-Conv --repo-type dataset --local-dir data/raw/IN22-Conv
```
