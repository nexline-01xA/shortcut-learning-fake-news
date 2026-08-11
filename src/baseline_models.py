"""
Phase 1, Step 3: Baseline models.

TF-IDF -> train -> evaluate, for:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Linear SVM

Run separately per dataset (LIAR = short claims, Kaggle = full articles --
different tasks, don't combine). Uses Pipeline A (clean_text, stopwords removed).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
}


def run_baselines(df: pd.DataFrame, label: str, text_col: str = "clean_text", target_col: str = "binary_label"):
    print(f"\n{'=' * 60}\nDataset: {label}\n{'=' * 60}")

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[target_col], test_size=0.2, random_state=42, stratify=df[target_col]
    )

    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    results = []
    for name, model in MODELS.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, pos_label="FAKE")
        rec = recall_score(y_test, preds, pos_label="FAKE")
        f1 = f1_score(y_test, preds, pos_label="FAKE")
        cm = confusion_matrix(y_test, preds, labels=["FAKE", "REAL"])

        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}  (FAKE as positive class)")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 score:  {f1:.4f}")
        print(f"Confusion matrix [rows=true, cols=pred, order=FAKE/REAL]:\n{cm}")

        results.append({
            "dataset": label, "model": name, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    liar_df = pd.read_csv("data/liar_clean.csv")
    kaggle_df = pd.read_csv("data/kaggle_clean.csv")

    # LinearSVC/TfidfVectorizer can choke on NaN from empty strings after cleaning
    liar_df["clean_text"] = liar_df["clean_text"].fillna("")
    kaggle_df["clean_text"] = kaggle_df["clean_text"].fillna("")

    liar_results = run_baselines(liar_df, "LIAR")
    kaggle_results = run_baselines(kaggle_df, "Kaggle")

    all_results = pd.concat([liar_results, kaggle_results], ignore_index=True)
    all_results.to_csv("results_baselines.csv", index=False)
    print(f"\n{'=' * 60}\nSummary table saved to results_baselines.csv\n{'=' * 60}")
    print(all_results.to_string(index=False))
