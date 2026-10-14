"""Download and load the Polish companies bankruptcy dataset (UCI #365).

The paper uses the "3rd year" file: 64 financial ratios per company, label =
bankrupt (1) or not (0) three years later. 10,503 rows, ~4.7% bankrupt.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split

DATA_URL = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 42


def download_data():
    """Download the UCI zip once and extract the .arff files into data/."""
    if (DATA_DIR / "3year.arff").exists():
        return
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Downloading dataset from {DATA_URL} ...")
    with urllib.request.urlopen(DATA_URL) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(DATA_DIR)


def load_year(year=3):
    """Return features X (DataFrame, NaN for missing) and labels y (0/1 array)."""
    download_data()
    raw, _ = arff.loadarff(DATA_DIR / f"{year}year.arff")
    df = pd.DataFrame(raw)
    y = df.pop("class").astype(str).astype(int).to_numpy()
    X = df.astype(float)  # '?' in the file is already read as NaN
    return X, y


def get_split(year=3):
    """80/20 random train/test split, same as the paper (Section 3.3)."""
    X, y = load_year(year)
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


if __name__ == "__main__":
    X, y = load_year(3)
    X_train, X_test, y_train, y_test = get_split(3)
    print(f"All data : {len(y):5d} rows, {y.sum():4d} bankrupt ({y.mean():.1%})")
    print(f"Train    : {len(y_train):5d} rows, {y_train.sum():4d} bankrupt ({y_train.mean():.1%})")
    print(f"Test     : {len(y_test):5d} rows, {y_test.sum():4d} bankrupt ({y_test.mean():.1%})")
    print(f"Features : {X.shape[1]}")
    print(f"Rows with at least one missing value: {X.isna().any(axis=1).mean():.1%}")
    worst = X.isna().mean().sort_values(ascending=False).head(3)
    print("Most-missing features:", ", ".join(f"{k} ({v:.0%})" for k, v in worst.items()))
