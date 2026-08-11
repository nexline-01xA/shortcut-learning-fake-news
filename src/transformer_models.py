"""
Phase 1, Step 4: Transformer models (BERT, DistilBERT).

Run this on Colab or Kaggle Notebooks with a GPU runtime -- it will not run
in this sandbox (no GPU, no huggingface.co access here).

Usage:
    python transformer_models.py --dataset liar --model bert
    python transformer_models.py --dataset kaggle --model distilbert

Uses Pipeline B (transformer_text column -- raw case/punctuation preserved,
only leakage/HTML/URL stripped). Do NOT use the stopword-removed clean_text
column here -- that hurts transformer performance, per the plan.

Install first:
    pip install transformers datasets torch scikit-learn pandas --break-system-packages
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)

MODEL_NAMES = {
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
}

LABEL_MAP = {"FAKE": 0, "REAL": 1}
LABEL_NAMES = ["FAKE", "REAL"]

# Kaggle articles average 220 tokens post-cleaning; LIAR claims average ~10-18.
# Cap sequence length per dataset to save compute without truncating most examples.
MAX_LENGTH = {"liar": 64, "kaggle": 256}


def load_data(dataset: str):
    path = f"data/{dataset}_clean.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["transformer_text", "binary_label"])
    df["label"] = df["binary_label"].map(LABEL_MAP)
    return df[["transformer_text", "label"]].rename(columns={"transformer_text": "text"})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, pos_label=0),  # FAKE = 0
        "recall": recall_score(labels, preds, pos_label=0),
        "f1": f1_score(labels, preds, pos_label=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["liar", "kaggle"], required=True)
    parser.add_argument("--model", choices=["bert", "distilbert"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    model_name = MODEL_NAMES[args.model]
    max_length = MAX_LENGTH[args.dataset]

    print(f"Loading {args.dataset} dataset...")
    df = load_data(args.dataset)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    print(f"Loading tokenizer + model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True)).map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=f"./results_{args.dataset}_{args.model}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print(f"\nFine-tuning {model_name} on {args.dataset}...")
    trainer.train()

    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    # Full confusion matrix for the report
    preds = trainer.predict(test_ds)
    pred_labels = np.argmax(preds.predictions, axis=-1)
    cm = confusion_matrix(test_ds["label"], pred_labels, labels=[0, 1])
    print(f"\nConfusion matrix [rows=true, cols=pred, order=FAKE/REAL]:\n{cm}")

    model.save_pretrained(f"./saved_model_{args.dataset}_{args.model}")
    tokenizer.save_pretrained(f"./saved_model_{args.dataset}_{args.model}")
    print(f"\nModel saved to ./saved_model_{args.dataset}_{args.model}")

    # Append to running results file
    result_row = pd.DataFrame([{
        "dataset": args.dataset, "model": model_name,
        "accuracy": metrics["eval_accuracy"], "precision": metrics["eval_precision"],
        "recall": metrics["eval_recall"], "f1": metrics["eval_f1"],
    }])
    try:
        existing = pd.read_csv("results_transformers.csv")
        result_row = pd.concat([existing, result_row], ignore_index=True)
    except FileNotFoundError:
        pass
    result_row.to_csv("results_transformers.csv", index=False)
    print("Appended results to results_transformers.csv")


if __name__ == "__main__":
    main()
