"""
Structural/style artifact check.
Tests whether REAL vs FAKE differ systematically in surface-level writing
style (not word choice) -- sentence length, punctuation density, capitalization
patterns. These are harder-to-remove artifacts than single tokens, and would
explain why Reuters-ablation alone didn't move accuracy.
"""

import re
import pandas as pd

df = pd.read_csv("data/kaggle_clean.csv")
df = df.dropna(subset=["transformer_text", "binary_label"])

def stats_for_text(t):
    t = str(t)
    words = t.split()
    n_words = len(words) or 1
    sentences = re.split(r'[.!?]+', t)
    sentences = [s for s in sentences if s.strip()]
    n_sentences = len(sentences) or 1

    return pd.Series({
        "avg_sentence_len_words": n_words / n_sentences,
        "avg_word_len_chars": sum(len(w) for w in words) / n_words,
        "exclamation_marks": t.count("!"),
        "question_marks": t.count("?"),
        "quote_marks": t.count('"') + t.count("'"),
        "all_caps_words": sum(1 for w in words if w.isupper() and len(w) > 1),
        "capital_ratio": sum(1 for c in t if c.isupper()) / max(len(t), 1),
        "digit_count": sum(1 for c in t if c.isdigit()),
        "comma_count": t.count(","),
    })

print("Computing per-article structural stats (this takes a minute on 39K rows)...")
stats = df["transformer_text"].apply(stats_for_text)
stats["binary_label"] = df["binary_label"].values

summary = stats.groupby("binary_label").mean()
summary_pct_diff = ((summary.loc["FAKE"] - summary.loc["REAL"]) / summary.loc["REAL"] * 100).round(1)

print("\n=== Mean values by class ===")
print(summary.round(3).T)

print("\n=== FAKE vs REAL: percent difference ===")
print(summary_pct_diff.sort_values(key=abs, ascending=False))

summary.T.to_csv("structural_artifact_summary.csv")
print("\nSaved to structural_artifact_summary.csv")
