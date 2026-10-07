# Data

Dataset: **SeLLaI Mal-Eng** (Mendeley Data, DOI [10.17632/5p4zbpy8wz.1](https://doi.org/10.17632/5p4zbpy8wz.1)),
licence **CC BY 4.0**. Cite it as given on the Mendeley page when you publish results.

1. Open the DOI page and press **Download All**.
2. Put the data file (`.csv`, `.xlsx` or `.tsv`) in `data/raw/`. Keep the licence/citation text next to it.
3. Run `python src/data_prep.py`. It prints the class counts, so check the exact class spelling
   (expected `Manglish`; pass `--label` if it differs), the number kept after every filter, and sample pairs.

`data/raw/` and `data/processed/` are git-ignored. Do not commit the dataset.
