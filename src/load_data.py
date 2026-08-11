"""
Phase 1: Dataset loading for Fake News Detection project.
Loads and standardizes Kaggle Fake News + LIAR datasets into one schema.
"""

import pandas as pd

LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title",
    "state", "party", "barely_true_ct", "false_ct", "half_true_ct",
    "mostly_true_ct", "pants_fire_ct", "context"
]

# LIAR has 6 fine-grained labels. We collapse them to binary (real/fake)
# for compatibility with the Kaggle dataset's binary labels.
LIAR_LABEL_MAP = {
    "true": "REAL",
    "mostly-true": "REAL",
    "half-true": "REAL",
    "barely-true": "FAKE",
    "false": "FAKE",
    "pants-fire": "FAKE",
}


def load_liar(split="train"):
    """Load one split ('train', 'test', 'valid') of the LIAR dataset."""
    path = f"data/liar/{split}.tsv"
    df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS)
    df["binary_label"] = df["label"].map(LIAR_LABEL_MAP)
    df["text"] = df["statement"]
    df["source"] = "LIAR"
    return df[["text", "binary_label", "source"]]


def load_kaggle_fake_news(fake_path="data/kaggle/Fake.csv", true_path="data/kaggle/True.csv"):
    """
    Load the Kaggle Fake News dataset (Fake.csv + True.csv).
    Download manually first — see instructions printed below if files are missing.
    """
    import os
    if not (os.path.exists(fake_path) and os.path.exists(true_path)):
        raise FileNotFoundError(
            f"Missing {fake_path} or {true_path}.\n"
            "Download steps:\n"
            "1. Go to https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            "2. Click Download (you'll need a free Kaggle account)\n"
            "3. Unzip it, then put Fake.csv and True.csv into fake-news-detection/data/kaggle/"
        )

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["binary_label"] = "FAKE"
    true_df["binary_label"] = "REAL"

    fake_df["text"] = fake_df["title"] + " " + fake_df["text"]
    true_df["text"] = true_df["title"] + " " + true_df["text"]

    fake_df["source"] = "KAGGLE"
    true_df["source"] = "KAGGLE"

    combined = pd.concat([fake_df, true_df], ignore_index=True)
    return combined[["text", "binary_label", "source"]]


if __name__ == "__main__":
    print("Loading LIAR dataset...")
    liar_train = load_liar("train")
    liar_test = load_liar("test")
    liar_valid = load_liar("valid")
    print(f"  LIAR train: {len(liar_train)} rows")
    print(f"  LIAR test:  {len(liar_test)} rows")
    print(f"  LIAR valid: {len(liar_valid)} rows")
    print(liar_train.head(3), "\n")

    print("Loading Kaggle dataset...")
    try:
        kaggle_df = load_kaggle_fake_news()
        print(f"  Kaggle: {len(kaggle_df)} rows")
        print(kaggle_df.head(3))
    except FileNotFoundError as e:
        print(e)
