"""
Multi-seed variance check for the Reuters ablation (TF-IDF pipeline).
Addresses the single-run-variance critique: reports mean +/- std across
3 random seeds instead of a single point estimate.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("data/kaggle_clean.csv").dropna(subset=[
    "transformer_text", "transformer_text_no_reuters", "binary_label"])

SEEDS = [42, 100, 999]
results = {"original": {"acc": [], "f1": []}, "no_reuters": {"acc": [], "f1": []}}

for seed in SEEDS:
    for col, label in [("transformer_text", "original"), ("transformer_text_no_reuters", "no_reuters")]:
        X_train, X_test, y_train, y_test = train_test_split(
            df[col], df["binary_label"], test_size=0.2, random_state=seed, stratify=df["binary_label"]
        )
        vec = TfidfVectorizer(max_features=10000)
        X_train_v = vec.fit_transform(X_train)
        X_test_v = vec.transform(X_test)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_v, y_train)
        preds = model.predict(X_test_v)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, pos_label="FAKE")
        results[label]["acc"].append(acc)
        results[label]["f1"].append(f1)
        print(f"seed={seed}  {label}: acc={acc:.4f}  f1={f1:.4f}")

print("\n=== Summary (mean +/- std across 3 seeds) ===")
for label in ["original", "no_reuters"]:
    acc_arr = np.array(results[label]["acc"])
    f1_arr = np.array(results[label]["f1"])
    print(f"{label}: accuracy = {acc_arr.mean():.4f} +/- {acc_arr.std():.4f}   f1 = {f1_arr.mean():.4f} +/- {f1_arr.std():.4f}")

drop = np.array(results["original"]["acc"]) - np.array(results["no_reuters"]["acc"])
print(f"\nReuters-removal accuracy drop per seed: {[f'{d*100:.2f}pts' for d in drop]}")
print(f"Mean drop: {drop.mean()*100:.2f} points, std: {drop.std()*100:.2f} points")
if drop.min() > 0:
    print("=> Drop is positive across ALL seeds -- direction of effect is stable, not noise.")
