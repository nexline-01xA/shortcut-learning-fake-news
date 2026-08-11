# Reproducibility Guide

## Environment

- Python 3.9–3.12 should work. Verified against Python 3.12.3 with
  pandas 3.0.2, scikit-learn 1.8.0, numpy 2.4.4 during preparation of this
  repository (see `requirements.txt` for minimum versions — the original
  notebooks/scripts do not pin exact historical versions).
- The classical pipeline (`src/*.py`, minus `transformer_models.py`) needs
  **no GPU** and runs on a laptop.
- The transformer notebooks (`notebooks/*.ipynb`) need a **GPU runtime**
  (the manuscript used a single T4 on Google Colab). `src/transformer_models.py`
  is a CLI equivalent of `notebooks/distilbert_training.ipynb` for running
  outside Colab, but still needs a GPU to finish in reasonable time.

## Step-by-step order

1. **Obtain datasets** — see `data/README.md`. Place raw files in
   `data/kaggle/` (`Fake.csv`, `True.csv`) and `data/liar/`
   (`train.tsv`, `test.tsv`, `valid.tsv`).
2. **Sanity-check loading**: `python src/load_data.py`
3. **Preprocess**: `python src/preprocess.py`
   → writes `data/liar_clean.csv`, `data/kaggle_clean.csv`.
4. **Baseline models** (classical, no GPU): `python src/baseline_models.py`
   → reproduces manuscript Table II, writes `results_baselines.csv`.
5. **Transformer fine-tuning** (GPU required): run
   `notebooks/distilbert_training.ipynb` in Colab, or
   `python src/transformer_models.py --dataset kaggle --model distilbert`
   and `--dataset liar --model distilbert` locally with a GPU.
   → reproduces manuscript Table III.
6. **LIME explainability** (GPU required, retrains DistilBERT on Kaggle
   internally): `notebooks/lime_explainability.ipynb`
   → reproduces manuscript Table IV.
7. **Frequency verification** (no GPU): `python src/token_frequency_check.py`
   → reproduces manuscript Table V exactly.
8. **Reuters ablation** (GPU required): `notebooks/reuters_ablation.ipynb`
   → reproduces the DistilBERT half of manuscript Table VI.
9. **Three-way ablation** (GPU required): `notebooks/three_way_ablation.ipynb`
   → also reproduces the DistilBERT half of Table VI plus the de-styled
   condition; see Section VI of the manuscript for the anomalous-result
   note this notebook's history relates to.
10. **Multi-seed variance check** (no GPU, classical pipeline):
    `python src/multiseed_variance.py`
    → reproduces manuscript Table VII exactly.
11. **Structural artifact analysis** (no GPU): `python src/structural_artifact_check.py`
    → reproduces manuscript Table VIII exactly.
12. **Cross-dataset generalization** (no GPU, classical pipeline):
    `python src/cross_dataset_test.py`
    → reproduces manuscript Table IX / the 98.91% → 48.48% headline result
    exactly.
13. **Length-confounder control** (no GPU, classical pipeline):
    `python src/length_confound_control.py`
    → reproduces manuscript Table X exactly.

## What was actually re-verified while preparing this repository

Steps 2, 4, 7, 10 (logic same as 11), 11, 12, and 13 were re-run end-to-end
against the recovered cleaned CSVs during repository preparation, and every
number matched the manuscript's reported tables exactly (Table II, V, VIII,
IX, X). This confirms the classical-pipeline (TF-IDF + Logistic
Regression / Naive Bayes / SVM) results — including both headline findings,
the cross-dataset collapse and the length-confounder control — are backed
by real, reproducible computation, not invented figures.

## What could not be independently re-verified here

This experiment is documented in the manuscript but the recovered archive
does not contain everything required for exact reproduction in this
environment:

- **DistilBERT training, LIME explainability, and the DistilBERT-side
  ablation results** (Tables III, IV, and the DistilBERT rows of Table VI)
  require a GPU and Hugging Face model downloads, neither of which is
  available in the environment used to prepare this repository. The
  notebooks are included as-is and are runnable on Colab; their code was
  read and is internally consistent with the manuscript's methodology, but
  the specific numbers (99.97% Kaggle accuracy, 62.44% LIAR accuracy, the
  LIME token weights) were not re-executed here.
- The manuscript's own Limitations section (Section VIII) already flags
  that DistilBERT's cross-dataset generalization and the length-confounder
  control were only measured with the classical pipeline — this is a
  disclosed scope boundary of the original study, not a gap introduced by
  this repository.

## Output file locations

| Script/notebook | Output |
|---|---|
| `src/preprocess.py` | `data/liar_clean.csv`, `data/kaggle_clean.csv` |
| `src/baseline_models.py` | `results_baselines.csv` |
| `src/token_frequency_check.py` | `token_frequency_by_class.csv` |
| `src/structural_artifact_check.py` | `structural_artifact_summary.csv` |
| `src/multiseed_variance.py` | printed summary only (no file) |
| `src/cross_dataset_test.py` | printed summary only (no file) |
| `src/length_confound_control.py` | printed summary only (no file) |
| `notebooks/distilbert_training.ipynb` | `results_full_comparison.csv`, `saved_model_*/` (Colab download) |
| `notebooks/lime_explainability.ipynb` | `lime_explanations.csv`, `lime_top_tokens_summary.csv`, `lime_html_examples/` (Colab download) |
| `notebooks/reuters_ablation.ipynb` | `reuters_ablation_results.csv` (Colab download) |
| `notebooks/three_way_ablation.ipynb` | `three_way_ablation_results.csv` (Colab download) |
