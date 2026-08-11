"""
Phase 1, Step 2: Preprocessing pipeline for Fake News Detection project.

- Dataset cleaning: drop_duplicates, drop_nulls
- Text cleaning: clean_text() -- URLs, punctuation, HTML tags, lowercase, whitespace, stopwords
- Token analysis: most frequent words, vocab size, avg tokens/article

Run per-dataset (LIAR is short claims, Kaggle is full articles -- keep them
separate through preprocessing since they're different NLP tasks; combine
only at the comparison stage).
"""

import re
import string
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Kaggle-specific leakage: real articles are scraped with a
# "CITY (Reuters) -" dateline prefix that fake articles never have.
# A model can learn to key off "reuters" alone instead of real content.
DATELINE_PATTERN = re.compile(r"^[A-Z][A-Za-z\s]*\(Reuters\)\s*-\s*", re.IGNORECASE)
SOURCE_TAG_WORDS = {"reuters"}


def clean_dataset(df: pd.DataFrame, text_col: str = "text", label_col: str = "binary_label") -> pd.DataFrame:
    """Drop duplicate texts and null rows. Returns a fresh, reindexed DataFrame."""
    before = len(df)
    df = df.dropna(subset=[text_col, label_col])
    df = df.drop_duplicates(subset=[text_col])
    df = df.reset_index(drop=True)
    after = len(df)
    print(f"  Dropped {before - after} rows (duplicates/nulls). {before} -> {after}")
    return df


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Pipeline A (remove_stopwords=True): for classical ML (LR, NB, SVM).
    lowercase -> strip HTML -> strip URLs -> strip dateline leakage ->
    strip punctuation -> collapse whitespace -> drop 1-char tokens ->
    drop source-tag words -> remove stopwords

    Pipeline B (remove_stopwords=False): NOT what this function returns for
    transformers -- use clean_text_minimal() instead. This flag only controls
    stopword removal within the classical-ML cleaning; it still lowercases
    and strips punctuation, which you don't want for BERT/DistilBERT.
    """
    text = DATELINE_PATTERN.sub(" ", str(text))  # strip before lowercasing (needs original case)
    text = text.lower()
    text = HTML_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1]  # drop single-char fragments (s, t, etc.)
    tokens = [t for t in tokens if t not in SOURCE_TAG_WORDS]  # drop source leakage words

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    text = " ".join(tokens)
    return text


REUTERS_WORD_PATTERN = re.compile(r"\bReuters\b", re.IGNORECASE)
MULTI_EXCLAIM_PATTERN = re.compile(r"!+")
MULTI_QUESTION_PATTERN = re.compile(r"\?+")


def destyle_text(text: str) -> str:
    """
    Neutralizes stylistic artifacts identified by the structural check:
    - collapses repeated '!' to a single '.', repeated '?' to a single '?'
    - converts ALL-CAPS words (shouting/emphasis) to Title Case, preserving
      normal sentence-initial and proper-noun capitalization elsewhere
    Does NOT touch word choice, sentence structure, or content -- only the
    punctuation/capitalization channel the structural check flagged.
    """
    text = MULTI_EXCLAIM_PATTERN.sub(".", str(text))
    text = MULTI_QUESTION_PATTERN.sub("?", text)

    words = text.split()
    fixed = []
    for w in words:
        core = w.strip(string.punctuation)
        if len(core) > 1 and core.isupper():
            fixed.append(w.capitalize() if w[0].isalpha() else w)
        else:
            fixed.append(w)
    return " ".join(fixed)


def clean_text_minimal(text: str) -> str:
    """
    Pipeline B: for transformers (BERT, DistilBERT, RoBERTa).
    Keeps case, punctuation, and stopwords -- the tokenizer and attention
    mechanism use these as signal. Only strips things that are pure noise
    or leakage: HTML, URLs, the Reuters dateline, and extra whitespace.
    """
    text = DATELINE_PATTERN.sub(" ", str(text))
    text = HTML_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def clean_text_minimal_no_reuters(text: str) -> str:
    """
    Same as clean_text_minimal, but additionally strips every standalone
    mention of 'Reuters' anywhere in the text -- not just the dateline
    prefix. Confirmed via frequency analysis: 'Reuters' appears in 83% of
    REAL articles vs 1.2% of FAKE (71x disproportion) -- the dateline strip
    alone only removed the prefix pattern, missing in-body mentions.
    Used for the leakage-ablation experiment: retrain and compare accuracy
    against the version that still contains 'Reuters'.
    """
    text = clean_text_minimal(text)
    text = REUTERS_WORD_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def token_analysis(df: pd.DataFrame, text_col: str = "clean_text", label: str = "") -> None:
    """Print vocab size, most frequent words, avg tokens/article."""
    all_tokens = []
    for t in df[text_col]:
        all_tokens.extend(t.split())

    vocab = set(all_tokens)
    counter = Counter(all_tokens)
    avg_tokens = len(all_tokens) / len(df) if len(df) else 0

    print(f"\n--- Token analysis: {label} ---")
    print(f"Vocabulary size = {len(vocab)}")
    print(f"Average tokens/article (post-cleaning) = {avg_tokens:.1f}")
    print("Top 15 most frequent words:")
    for word, freq in counter.most_common(15):
        print(f"  {word}: {freq}")


def clean_text_no_reuters_destyled(text: str) -> str:
    """Both interventions combined: Reuters removed AND style artifacts neutralized."""
    text = clean_text_minimal_no_reuters(text)
    return destyle_text(text)


def preprocess(df: pd.DataFrame, label: str, text_col: str = "text", label_col: str = "binary_label") -> pd.DataFrame:
    print(f"\n=== Preprocessing: {label} ===")
    df = clean_dataset(df, text_col, label_col)
    df["clean_text"] = df[text_col].apply(clean_text)          # Pipeline A: classical ML
    df["transformer_text"] = df[text_col].apply(clean_text_minimal)  # Pipeline B: original
    df["transformer_text_no_reuters"] = df[text_col].apply(clean_text_minimal_no_reuters)  # Reuters removed
    df["transformer_text_no_reuters_destyled"] = df[text_col].apply(clean_text_no_reuters_destyled)  # + style removed
    token_analysis(df, "clean_text", label)
    return df


if __name__ == "__main__":
    from load_data import load_liar, load_kaggle_fake_news

    liar_df = pd.concat([load_liar("train"), load_liar("test"), load_liar("valid")], ignore_index=True)
    liar_clean = preprocess(liar_df, "LIAR")

    kaggle_df = load_kaggle_fake_news()
    kaggle_clean = preprocess(kaggle_df, "Kaggle")

    liar_clean.to_csv("data/liar_clean.csv", index=False)
    kaggle_clean.to_csv("data/kaggle_clean.csv", index=False)
    print("\nSaved: data/liar_clean.csv, data/kaggle_clean.csv")
