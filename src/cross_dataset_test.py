"""
Cross-dataset generalization test.
Trains TF-IDF + Logistic Regression on Kaggle (full articles), tests on LIAR
(short claims) with zero retraining -- directly tests whether the model
generalizes across domains or only within the distribution it was trained on.
Runs locally, no GPU needed, no Colab risk.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

kaggle = pd.read_csv("data/kaggle_clean.csv").dropna(subset=["clean_text", "binary_label"])
liar = pd.read_csv("data/liar_clean.csv").dropna(subset=["clean_text", "binary_label"])

print(f"Training on Kaggle: {len(kaggle)} articles")
print(f"Testing on LIAR (zero-shot, no retraining): {len(liar)} statements\n")

vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(kaggle["clean_text"])
y_train = kaggle["binary_label"]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# In-domain sanity check (Kaggle self-test, for comparison)
X_kaggle_test = vectorizer.transform(kaggle["clean_text"].sample(frac=0.2, random_state=42))
y_kaggle_test = kaggle["binary_label"].sample(frac=0.2, random_state=42)
in_domain_acc = accuracy_score(y_kaggle_test, model.predict(X_kaggle_test))

# Cross-domain test: LIAR, completely unseen domain
X_liar = vectorizer.transform(liar["clean_text"])
y_liar = liar["binary_label"]
preds = model.predict(X_liar)

cross_acc = accuracy_score(y_liar, preds)
cross_f1 = f1_score(y_liar, preds, pos_label="FAKE")
cm = confusion_matrix(y_liar, preds, labels=["FAKE", "REAL"])

print(f"In-domain (Kaggle self-test, for reference): {in_domain_acc:.4f}")
print(f"Cross-domain (trained Kaggle, tested LIAR):  accuracy={cross_acc:.4f}  f1={cross_f1:.4f}")
print(f"Confusion matrix [FAKE/REAL]:\n{cm}")
print(f"\nAccuracy drop from in-domain to cross-domain: {(in_domain_acc-cross_acc)*100:.1f} percentage points")
print("\n" + classification_report(y_liar, preds))
