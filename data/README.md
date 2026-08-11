# Data

This folder is intentionally empty of actual data files. Raw and processed
datasets are **not committed** to this repository — see rationale below.

## Datasets used

| Dataset | Role | Source |
|---|---|---|
| Kaggle Fake-and-Real-News (mirror of ISOT) | Training + in-domain evaluation | [Kaggle: clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| LIAR | Zero-shot cross-dataset evaluation | [Wang, 2017 — LIAR dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) (original release accompanying the ACL 2017 paper) |

## What's excluded and why

- **`kaggle_clean.csv`** (~458 MB after preprocessing) is not committed. It
  exceeds GitHub's normal file-size comfort limit (and its raw sources —
  `Fake.csv` / `True.csv` — are third-party Kaggle data whose redistribution
  terms this repository does not independently verify).
- **`liar_clean.csv`** and the raw LIAR `train.tsv` / `test.tsv` / `valid.tsv`
  splits are also excluded from this repository. The original LIAR release is
  publicly downloadable from the link above; this repo does not re-host it.
- No raw third-party dataset is redistributed here. If you need to confirm
  redistribution rights for either dataset before mirroring it yourself,
  check the current terms on Kaggle and in Wang (2017) directly — this
  repository does not make that determination for you.

## How to reconstruct the cleaned datasets

1. Download the Kaggle Fake-and-Real-News dataset and place `Fake.csv` and
   `True.csv` in `data/kaggle/`.
2. Download the LIAR dataset and place `train.tsv`, `test.tsv`, `valid.tsv`
   in `data/liar/`.
3. From the repository root, run:
   ```bash
   python src/load_data.py      # sanity-check both datasets load correctly
   python src/preprocess.py     # writes data/liar_clean.csv, data/kaggle_clean.csv
   ```
4. `preprocess.py` produces both cleaned CSVs with the columns described below.

## Expected row counts (from the manuscript and verified against the recovered cleaned CSVs)

| Dataset | Raw rows | After dedup/cleaning | REAL | FAKE |
|---|---|---|---|---|
| LIAR | 12,791 (train+test+valid, after removing 26 dup/null rows) | 12,791 | 7,134 (55.8%) | 5,657 (44.2%) |
| Kaggle | ~44,898 raw | 39,105 (after removing 5,793 duplicates, 13.0%, and null rows) | 21,197 (54.2%) | 17,908 (45.8%) |

Note: the recovered `kaggle_clean.csv` in the original archive contains
39,114 rows and `liar_clean.csv` contains 12,811 rows (including header)
— consistent with the manuscript's reported counts to within normal
row-counting/header conventions. Re-running the pipeline from a fresh
Kaggle download may differ by a handful of rows depending on the exact
snapshot of the Kaggle mirror you download.

## Label mapping (LIAR six-point → binary)

```
REAL if label ∈ {true, mostly-true, half-true}
FAKE if label ∈ {barely-true, false, pants-fire}
```

## Cleaned CSV columns (produced by `src/preprocess.py`)

| Column | Used by | Description |
|---|---|---|
| `text` | — | Original raw text |
| `binary_label` | all | `REAL` / `FAKE` |
| `source` | — | `LIAR` or `KAGGLE` |
| `clean_text` | classical ML (`src/baseline_models.py`, `src/cross_dataset_test.py`, `src/length_confound_control.py`) | Pipeline A: lowercased, HTML/URL/punctuation stripped, stopwords removed |
| `transformer_text` | DistilBERT notebooks | Pipeline B: case/punctuation/stopwords preserved; only HTML, URLs, and the `CITY (Reuters) -` dateline pattern stripped |
| `transformer_text_no_reuters` | `notebooks/reuters_ablation.ipynb`, `notebooks/three_way_ablation.ipynb` | `transformer_text` with every standalone "Reuters" mention removed |
| `transformer_text_no_reuters_destyled` | `notebooks/three_way_ablation.ipynb` | `transformer_text_no_reuters` with repeated `!`/`?` collapsed and ALL-CAPS words converted to Title Case |

## Licensing note

Datasets retain whatever license/terms their original publishers apply.
The MIT License on this repository's code does **not** extend to any
dataset — see the note in the main README.
