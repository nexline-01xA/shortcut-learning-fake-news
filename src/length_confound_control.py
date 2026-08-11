"""
Length-confounder control test.
The Kaggle-to-LIAR cross-dataset collapse could be explained by article
length mismatch (417 words vs 18 words) rather than genuine domain/shortcut
mismatch. This test isolates length from domain: take the SAME Kaggle-trained
model and test it on Kaggle articles artificially truncated to ~18 words
(matching LIAR's average length). If accuracy stays high on truncated
IN-DOMAIN text, length alone does not explain the collapse -- domain shift does.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

kaggle = pd.read_csv("data/kaggle_clean.csv").dropna(subset=["clean_text", "binary_label"])
liar = pd.read_csv("data/liar_clean.csv").dropna(subset=["clean_text", "binary_label"])

print(f"Training on Kaggle: {len(kaggle)} articles\n")

vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(kaggle["clean_text"])
y_train = kaggle["binary_label"]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Held-out Kaggle test split (same as the original cross-dataset test used)
kaggle_test = kaggle.sample(frac=0.2, random_state=42)

# Condition 1: full-length Kaggle test articles (original in-domain result)
X_full = vectorizer.transform(kaggle_test["clean_text"])
acc_full = accuracy_score(kaggle_test["binary_label"], model.predict(X_full))

# Condition 2: SAME Kaggle test articles, truncated to first 18 words
# (matching LIAR's average length) -- still Kaggle domain, just short
truncated_text = kaggle_test["clean_text"].apply(lambda t: " ".join(str(t).split()[:18]))
X_trunc = vectorizer.transform(truncated_text)
acc_trunc = accuracy_score(kaggle_test["binary_label"], model.predict(X_trunc))
f1_trunc = f1_score(kaggle_test["binary_label"], model.predict(X_trunc), pos_label="FAKE")

# Condition 3: actual cross-domain result for reference (LIAR)
X_liar = vectorizer.transform(liar["clean_text"])
acc_liar = accuracy_score(liar["binary_label"], model.predict(X_liar))

print(f"Condition 1 -- Kaggle test, full length (417 words avg):     {acc_full:.4f}")
print(f"Condition 2 -- Kaggle test, TRUNCATED to 18 words:            {acc_trunc:.4f}  (f1={f1_trunc:.4f})")
print(f"Condition 3 -- LIAR test, native short claims (cross-domain): {acc_liar:.4f}")
print(f"\nDrop from length truncation ALONE (same domain): {(acc_full-acc_trunc)*100:.1f} points")
print(f"Drop from cross-domain shift (Kaggle model on LIAR):  {(acc_full-acc_liar)*100:.1f} points")

if acc_trunc > 0.85:
    print("\n=> Truncation alone causes minimal drop. Length is NOT the primary cause")
    print("   of the cross-dataset collapse. Supports the domain/shortcut-learning explanation.")
else:
    print("\n=> Truncation alone causes a substantial drop. Length may be a confounding")
    print("   factor in the cross-dataset result -- the shortcut-learning claim needs qualification.")
