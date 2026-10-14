"""Replicate Table 1 of the paper as closely as the paper describes it.

Paper setup (Sections 3-5):
  * 3rd-year data, random 80/20 split
  * mean imputation, NO feature scaling
  * L2 regularisation, learning rate 0.001
  * 5-fold cross validation on the training set
  * metrics: accuracy, log loss, train accuracy, fit time

Note: the paper's log loss values (e.g. 1.824752 for SVM) only come out if
log loss is computed on hard 0/1 predictions instead of probabilities, so we
do the same here to reproduce the table. improved.py uses real probabilities.
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from data import RANDOM_STATE, get_split

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def paper_models():
    """The six models from Section 4 with the settings the paper reports."""
    models = {
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000),  # L2 by default
        # gamma="auto" was the scikit-learn default in 2019; it gives the paper's ~1.0 train accuracy
        "SVM": SVC(C=1.0, kernel="rbf", gamma="auto"),
        "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(5, 5), activation="logistic",
                                       solver="sgd", alpha=1e-4, learning_rate_init=0.001,
                                       max_iter=200, random_state=RANDOM_STATE),
        "NaiveBayes": BernoulliNB(),
        # gblinear is a linear booster, so the paper's max_depth=3 has no effect and is left out
        "ExtremeGradientBoosting": XGBClassifier(booster="gblinear", learning_rate=0.001,
                                                 reg_lambda=1.0, random_state=RANDOM_STATE),
        "LightGradientBoostingMachine": LGBMClassifier(learning_rate=0.001, reg_lambda=1.0,
                                                       random_state=RANDOM_STATE, verbose=-1),
    }
    # Every model gets the same preprocessing: mean imputation only (Section 3.4 / 5.3)
    return {name: make_pipeline(SimpleImputer(strategy="mean"), m) for name, m in models.items()}


def hard_log_loss(y_true, y_pred):
    """Log loss on 0/1 predictions, clipped like older scikit-learn (eps=1e-15)."""
    return log_loss(y_true, np.clip(y_pred.astype(float), 1e-15, 1 - 1e-15), labels=[0, 1])


def main():
    X_train, X_test, y_train, y_test = get_split()
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for name, model in paper_models().items():
        print(f"Training {name} ...")
        cv_acc = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        pred_test = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred_test, labels=[0, 1]).ravel()
        rows.append({
            "Classifier": name,
            "Test Accuracy": accuracy_score(y_test, pred_test),
            "Test Log Loss": hard_log_loss(y_test, pred_test),
            "Train Accuracy": accuracy_score(y_train, model.predict(X_train)),
            "Train Time (s)": train_time,
            "CV Accuracy (5-fold)": cv_acc.mean(),
            # Extra columns the paper did not report, but which expose the problem:
            "Test Recall (bankrupt)": recall_score(y_test, pred_test, zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "replication.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Replication of paper Table 1 ===")
    print(table.round(4).to_string(index=False))
    print(f"\nAlways predicting 'not bankrupt' would score {1 - y_test.mean():.4f} test accuracy.")
    print(f"Saved to {RESULTS_DIR / 'replication.csv'}")


if __name__ == "__main__":
    main()
