"""Make the report figures. Run replicate.py and improved.py first.

  fig1_class_balance.png       - paper Fig 1/2: how imbalanced the data is
  fig2_svm_learning_curve.png  - paper Fig 3a: learning curve / scalability for SVM
  fig3_accuracy_vs_recall.png  - paper models: high accuracy, (almost) zero recall
  fig4_roc_pr_curves.png       - improved models: ROC and precision-recall curves
  fig5_confusion_matrices.png  - paper SVM vs improved XGBoost on the test set

Use `python plots.py --fast` to skip fig2, which takes ~10 minutes.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import learning_curve

from data import get_split
from replicate import paper_models

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"

# Colours: fixed categorical order, text in ink colours (never the series colour)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "lines.linewidth": 2, "legend.frameon": False, "font.size": 10,
})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / name}")


def fig_class_balance(y_train, y_test):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    splits = {"Train": y_train, "Test": y_test}
    x = np.arange(len(splits))
    for i, (label, cls) in enumerate([("Not bankrupt", 0), ("Bankrupt", 1)]):
        counts = [(y == cls).sum() for y in splits.values()]
        bars = ax.bar(x + (i - 0.5) * 0.38, counts, width=0.36, color=SERIES[i], label=label)
        for bar, y in zip(bars, splits.values()):
            ax.annotate(f"{bar.get_height():,}\n({bar.get_height() / len(y):.1%})",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8, color=INK_2,
                        xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, list(splits))
    ax.set_ylabel("Companies")
    ax.set_ylim(0, max(len(y_train), 1) * 1.15)
    ax.set_title("Only ~5% of companies go bankrupt")
    ax.legend(loc="upper right")
    save(fig, "fig1_class_balance.png")


def fig_svm_learning_curve(X_train, y_train):
    """Same three panels as the paper's Figure 3.a (scikit-learn 'plot_learning_curve')."""
    print("Computing SVM learning curve (takes a few minutes) ...")
    sizes, train_sc, val_sc, fit_t, _ = learning_curve(
        paper_models()["SVM"], X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), scoring="accuracy", return_times=True)
    baseline = 1 - y_train.mean()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    for scores, label, c in [(train_sc, "Training score", SERIES[0]),
                             (val_sc, "Cross-validation score", SERIES[1])]:
        m, s = scores.mean(1), scores.std(1)
        ax.fill_between(sizes, m - s, m + s, color=c, alpha=0.15, linewidth=0)
        ax.plot(sizes, m, "o-", color=c, label=label, markersize=5)
    ax.axhline(baseline, color=MUTED, linestyle="--", linewidth=1.2)
    ax.annotate("always predict 'not bankrupt'", (sizes[-1], baseline), color=INK_2, fontsize=8,
                ha="right", xytext=(0, 5), textcoords="offset points")
    ax.set(title="Learning curve (SVM)", xlabel="Training examples", ylabel="Accuracy")
    ax.legend(loc="center right")

    ax = axes[1]
    ax.plot(sizes, fit_t.mean(1), "o-", color=SERIES[0], markersize=5)
    ax.fill_between(sizes, fit_t.mean(1) - fit_t.std(1), fit_t.mean(1) + fit_t.std(1),
                    color=SERIES[0], alpha=0.15, linewidth=0)
    ax.set(title="Scalability", xlabel="Training examples", ylabel="Fit time (s)")

    ax = axes[2]
    order = np.argsort(fit_t.mean(1))
    ax.plot(fit_t.mean(1)[order], val_sc.mean(1)[order], "o-", color=SERIES[1], markersize=5)
    ax.set(title="Performance", xlabel="Fit time (s)", ylabel="CV accuracy")
    save(fig, "fig2_svm_learning_curve.png")


def fig_accuracy_vs_recall(rep):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    y = np.arange(len(rep))
    for i, (col, label) in enumerate([("Test Accuracy", "Test accuracy"),
                                      ("Test Recall (bankrupt)", "Recall on bankrupt companies")]):
        bars = ax.barh(y + (i - 0.5) * 0.38, rep[col], height=0.36, color=SERIES[i], label=label)
        for bar in bars:
            ax.annotate(f"{bar.get_width():.2f}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        va="center", fontsize=8, color=INK_2, xytext=(3, 0), textcoords="offset points")
    ax.set_yticks(y, rep["Classifier"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.grid(axis="y", visible=False)
    ax.set_title("Paper setup: ~95% accuracy, ~0 bankruptcies caught", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save(fig, "fig3_accuracy_vs_recall.png")


def fig_roc_pr(scores):
    y_true = scores.pop("y_true").to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, (name, s) in enumerate(scores.items()):
        fpr, tpr, _ = roc_curve(y_true, s)
        prec, rec, _ = precision_recall_curve(y_true, s)
        ax1.plot(fpr, tpr, color=SERIES[i], label=f"{name} (AUC {roc_auc_score(y_true, s):.3f})")
        ax2.plot(rec, prec, color=SERIES[i], label=f"{name} (AP {average_precision_score(y_true, s):.3f})")
    ax1.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2, label="Random guess")
    ax2.axhline(y_true.mean(), color=MUTED, linestyle="--", linewidth=1.2, label="Random guess")
    ax1.set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate (recall)")
    ax2.set(title="Precision-recall curve (test set)", xlabel="Recall", ylabel="Precision")
    ax1.legend(loc="lower right", fontsize=8)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    save(fig, "fig4_roc_pr_curves.png")


def fig_confusion_matrices(rep, imp):
    panels = [("Paper SVM", rep.set_index("Classifier").loc["SVM"]),
              ("Improved XGBoost", imp.set_index("Classifier").loc["XGBoost"])]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (title, r) in zip(axes, panels):
        cm = np.array([[r.TN, r.FP], [r.FN, r.TP]], dtype=int)
        row_share = cm / cm.sum(axis=1, keepdims=True)  # colour = share of each true class
        ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}\n({row_share[i, j]:.0%})", ha="center", va="center",
                    color="white" if row_share[i, j] > 0.5 else INK, fontsize=11)
        ax.set_xticks([0, 1], ["Not bankrupt", "Bankrupt"])
        ax.set_yticks([0, 1], ["Not bankrupt", "Bankrupt"])
        ax.set(title=title, xlabel="Predicted", ylabel="Actual")
        ax.grid(False)
    save(fig, "fig5_confusion_matrices.png")


def main(fast=False):
    X_train, X_test, y_train, y_test = get_split()
    rep = pd.read_csv(RESULTS_DIR / "replication.csv")
    imp = pd.read_csv(RESULTS_DIR / "improved.csv")
    scores = pd.read_csv(RESULTS_DIR / "improved_test_scores.csv")

    fig_class_balance(y_train, y_test)
    fig_accuracy_vs_recall(rep)
    fig_roc_pr(scores)
    fig_confusion_matrices(rep, imp)
    if not fast:
        fig_svm_learning_curve(X_train, y_train)


if __name__ == "__main__":
    main(fast="--fast" in sys.argv)
