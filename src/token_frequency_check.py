"""
Word-frequency-by-class check.
Verifies whether LIME's top tokens actually occur disproportionately
in REAL vs FAKE articles, rather than trusting LIME's local weights alone.
"""

import pandas as pd
import re

df = pd.read_csv("data/kaggle_clean.csv")
df = df.dropna(subset=["transformer_text", "binary_label"])

real_texts = df[df["binary_label"] == "REAL"]["transformer_text"]
fake_texts = df[df["binary_label"] == "FAKE"]["transformer_text"]

n_real = len(real_texts)
n_fake = len(fake_texts)
print(f"REAL articles: {n_real} | FAKE articles: {n_fake}")

TOKENS = ["Indonesia", "Reuters", "parliament", "committee", "Senate", "Yomiuri", "Dr"]

def count_articles_containing(texts, token):
    # word-boundary match, case-insensitive, count ARTICLES containing it (not raw occurrences)
    pattern = re.compile(r"\b" + re.escape(token) + r"\b", re.IGNORECASE)
    return texts.apply(lambda t: bool(pattern.search(str(t)))).sum()

rows = []
for token in TOKENS:
    real_count = count_articles_containing(real_texts, token)
    fake_count = count_articles_containing(fake_texts, token)
    real_rate = real_count / n_real
    fake_rate = fake_count / n_fake
    ratio = (real_rate / fake_rate) if fake_rate > 0 else float("inf")

    rows.append({
        "token": token,
        "real_count": real_count,
        "fake_count": fake_count,
        "real_rate": round(real_rate, 4),
        "fake_rate": round(fake_rate, 4),
        "real_to_fake_ratio": round(ratio, 2) if ratio != float("inf") else "inf (never in FAKE)",
    })

result_df = pd.DataFrame(rows)
result_df.to_csv("token_frequency_by_class.csv", index=False)
print("\n" + result_df.to_string(index=False))
print("\nSaved to token_frequency_by_class.csv")
