# Methodology

Full detail is in the manuscript (`paper/manuscript.pdf`); this document is a
map from methodology sections to the code that implements them.

## Two preprocessing pipelines (`src/preprocess.py`)

- **Pipeline A** (`clean_text`) — for classical ML: lowercase, strip
  HTML/URLs/the Reuters dateline, strip punctuation, drop single-character
  tokens, remove stopwords (scikit-learn's English list).
- **Pipeline B** (`clean_text_minimal`) — for transformers: preserves case,
  punctuation, and stopwords (these carry signal for contextual models);
  strips only HTML, URLs, and the `CITY (Reuters) -` dateline pattern.
- Two further Pipeline-B variants remove the "Reuters" token entirely
  (`clean_text_minimal_no_reuters`) or additionally neutralize stylistic
  artifacts — collapsed `!`/`?` and Title-Cased ALL-CAPS words
  (`clean_text_no_reuters_destyled`) — used by the ablation notebooks.

## Baseline models (`src/baseline_models.py`)

TF-IDF (10,000 features) + Logistic Regression / Multinomial Naive Bayes /
Linear SVM, trained separately per dataset with an 80/20 stratified split.
Reproduces manuscript Table II exactly when run against the cleaned data.

## Transformer fine-tuning (`src/transformer_models.py`, `notebooks/distilbert_training.ipynb`)

DistilBERT (`distilbert-base-uncased`), 3 epochs per dataset, Hugging Face
`Trainer`. Sequence length capped at 64 tokens (LIAR) / 256 tokens (Kaggle)
to match each dataset's typical length. Requires a GPU — run in
`notebooks/distilbert_training.ipynb` (Colab) or via
`python src/transformer_models.py --dataset kaggle --model distilbert`
on a machine with a GPU.

## Explainability (LIME) (`notebooks/lime_explainability.ipynb`)

LIME applied to the fine-tuned Kaggle DistilBERT model on 10 REAL + 10 FAKE
test predictions, 300 local perturbation samples per example. Aggregates
per-token importance to surface globally influential tokens per class
(reproduces manuscript Table IV).

## Frequency verification (`src/token_frequency_check.py`)

Cross-checks whether LIME's local explanations reflect genuine corpus-level
patterns by directly counting token occurrence rates by class across the
full 39,105-article Kaggle corpus. Reproduces manuscript Table V exactly.

## Ablation design (`notebooks/reuters_ablation.ipynb`, `notebooks/three_way_ablation.ipynb`, `src/multiseed_variance.py`)

Two causal interventions, tested by retraining on modified text and
comparing to a same-hyperparameter baseline:
1. Remove every standalone "Reuters" mention (not just the dateline prefix).
2. Additionally neutralize stylistic artifacts (repeated `!`/`?`,
   ALL-CAPS → Title Case).

Tested on **both** pipelines: DistilBERT (15K-article subsample, single run)
and TF-IDF + Logistic Regression (full dataset). The TF-IDF version was
additionally re-run across 3 random seeds (`src/multiseed_variance.py`,
seeds 42/100/999) to confirm the direction of the Reuters-removal effect is
stable rather than single-split noise. Reproduces manuscript Tables VI–VII
exactly against the recovered cleaned data.

## Structural artifact analysis (`src/structural_artifact_check.py`)

Corpus-level surface-style statistics (exclamation/question mark density,
capitalization ratio, sentence length) computed independently of the
token-level ablation, to check for a stylistic (not just lexical) shortcut.
Reproduces manuscript Table VIII exactly.

## Cross-dataset generalization (`src/cross_dataset_test.py`)

TF-IDF + Logistic Regression trained entirely on Kaggle, evaluated zero-shot
on LIAR with no retraining. **This is the classical pipeline only** —
DistilBERT's cross-dataset generalization was not separately measured; see
the Limitations section of the manuscript and `docs/REPRODUCIBILITY.md`.
Reproduces manuscript Table IX / the 98.91% → 48.48% headline result
exactly.

## Length-confounder control (`src/length_confound_control.py`)

Truncates held-out Kaggle test articles to their first 18 words (LIAR's
average length) while keeping them in-domain, to separate a pure
length effect from the domain-shift effect above. Also classical-pipeline
only. Reproduces manuscript Table X exactly (4.7-point drop from
truncation alone vs. 50.4-point drop under actual domain shift).

## A note on the anomalous 100% result

An initial three-way ablation run twice produced identical 100.00% scores
across all three conditions — implausible for independently trained models.
The manuscript's Section VI documents the root cause (an interaction between
malformed-row skipping and an unconditional sample size that produced a
poorly stratified split) and how it was caught and corrected. This is
preserved here as a methodological contribution, not smoothed over.
