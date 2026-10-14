"""An improved (but still simple) version of the paper's experiment.

What we change compared to replicate.py, and why:
  1. Class imbalance: only ~5% of companies go bankrupt, so a model that always
     says "not bankrupt" gets ~95% accuracy. We give the rare class more weight
     (class_weight="balanced" / scale_pos_weight).
  2. Scaling: the 64 ratios are heavily skewed with extreme outliers. A
     QuantileTransformer maps every feature to a normal shape, which is what
     LR, SVM, the MLP and Gaussian Naive Bayes need.
  3. Median imputation instead of mean (the mean is dragged around by outliers).
     XGBoost/LightGBM handle missing values natively, so they get raw data.
  4. Proper tree boosting (the paper used a linear booster) + a random forest.
  5. Better metrics: ROC-AUC, PR-AUC, recall, precision, F1 and balanced accuracy,
     with stratified 5-fold CV on ROC-AUC.
"""
import time
import warnings
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC
from xgboost import XGBClassifier

from data import RANDOM_STATE, get_split

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def improved_models(neg_pos_ratio):
    """neg_pos_ratio = (# not bankrupt) / (# bankrupt) in the training set."""
    def scaled(model):
        return make_pipeline(SimpleImputer(strategy="median"),
                             QuantileTransformer(output_distribution="normal",
                                                 random_state=RANDOM_STATE),
                             model)

    return {
        "LogisticRegression": scaled(LogisticRegression(class_weight="balanced", max_iter=1000)),
        "SVM": scaled(SVC(kernel="rbf", class_weight="balanced")),
        # MLPClassifier has no class_weight option, so it relies on scaling only.
        # No early_stopping: on 95/5 data the validation accuracy stops improving
        # almost immediately and the network never learns the rare class.
        "NeuralNetwork": scaled(MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3,
                                              max_iter=500, random_state=RANDOM_STATE)),
        "NaiveBayes": scaled(GaussianNB()),
        "RandomForest": make_pipeline(SimpleImputer(strategy="median"),
                                      RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                             class_weight="balanced_subsample",
                                                             n_jobs=-1, random_state=RANDOM_STATE)),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=neg_pos_ratio, random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   class_weight="balanced", random_state=RANDOM_STATE,
                                   verbose=-1),
    }


def bankrupt_score(model, X):
    """Higher = more likely bankrupt. SVC has no predict_proba, so use its margin.
    ROC-AUC and PR-AUC only depend on the ranking, so either kind of score works."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main():
    X_train, X_test, y_train, y_test = get_split()
    RESULTS_DIR.mkdir(exist_ok=True)
    neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows, test_scores = [], {"y_true": y_test}
    for name, model in improved_models(neg_pos_ratio).items():
        print(f"Training {name} ...")
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        score = bankrupt_score(model, X_test)
        pred = model.predict(X_test)
        test_scores[name] = score
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append({
            "Classifier": name,
            "CV ROC-AUC": cv_auc.mean(),
            "CV ROC-AUC std": cv_auc.std(),
            "Test ROC-AUC": roc_auc_score(y_test, score),
            "Test PR-AUC": average_precision_score(y_test, score),
            "Test Recall": recall_score(y_test, pred, zero_division=0),
            "Test Precision": precision_score(y_test, pred, zero_division=0),
            "Test F1": f1_score(y_test, pred, zero_division=0),
            "Test Balanced Acc": balanced_accuracy_score(y_test, pred),
            "Test Accuracy": accuracy_score(y_test, pred),
            "Train Time (s)": train_time,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })

    table = pd.DataFrame(rows).sort_values("Test ROC-AUC", ascending=False)
    table.to_csv(RESULTS_DIR / "improved.csv", index=False)
    pd.DataFrame(test_scores).to_csv(RESULTS_DIR / "improved_test_scores.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n=== Improved models (sorted by test ROC-AUC) ===")
    print(table.round(4).to_string(index=False))
    print(f"\nTest set: {len(y_test)} companies, {y_test.sum()} bankrupt. "
          f"Recall = share of bankrupt companies the model catches.")
    print(f"Saved to {RESULTS_DIR / 'improved.csv'}")


if __name__ == "__main__":
    main()
