# Corporate Bankruptcy Prediction: Replication and Improvement

This project replicates the CS229 (Fall 2019) project
[*Corporate Bankruptcy Prediction*](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26646282.pdf)
(N. A. Patil). It then fixes the main weakness in the paper's evaluation and improves the models.

**Dataset:** [Polish Companies Bankruptcy](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data) from the UCI ML Repository.
We use the **3rd-year** file, like the paper:
- 10,503 companies, each described by 64 financial ratios.
- The label says whether the company went bankrupt within the following 3 years.
- Only **4.7%** of the companies went bankrupt.

## How to run

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cd src
python data.py        # downloads the data and prints a summary (~1 s)
python replicate.py   # paper's Table 1 -> results/replication.csv (~3 min)
python improved.py    # improved models -> results/improved.csv (~2 min)
python plots.py       # figures -> results/figures/ (~10 min, mostly the SVM learning curve)
                      # add --fast to skip the learning curve
```

## Project structure

| File | What it does |
|---|---|
| `src/data.py` | Downloads the UCI data and loads `3year.arff` (`?` becomes NaN). Makes the 80/20 random train/test split. |
| `src/replicate.py` | Runs the paper's six models with the paper's settings: mean imputation, no scaling, L2 regularisation, lr = 0.001, 5-fold CV. |
| `src/improved.py` | Our improved pipeline: class weighting, quantile scaling, tree boosting and imbalance-aware metrics. |
| `src/plots.py` | All figures in `results/figures/`. |

## 1. Replication of the paper's Table 1

Paper numbers vs. our run (test set = 20% hold-out, 2,101 companies):

| Classifier | Paper test acc | Our test acc | Paper test log loss | Our test log loss | Paper train acc | Our train acc |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9467 | 0.9481 | 1.84 | 1.79 | 0.952 | 0.954 |
| SVM | 0.9472 | 0.9491 | 1.82 | 1.76 | 1.000 | 0.997 |
| Neural Network | 0.9472 | 0.9491 | 1.82 | 1.76 | 0.954 | 0.954 |
| Naive Bayes (Bernoulli) | 0.7558 | 0.7630 | 8.43 | 8.19 | 0.765 | 0.766 |
| XGBoost (gblinear) | 0.5264 | 0.9491 | 16.36 | 1.76 | 1.000 | 0.954 |
| LightGBM | 0.7087 | 0.9491 | 10.06 | 1.76 | 1.000 | 0.954 |
| *Always predict "not bankrupt"* | *0.9472* | *0.9491* | | | | |

**Logistic Regression, SVM, the Neural Network and Naive Bayes replicate closely.** The remaining
differences come from the random split: the paper's test set had 111 bankrupt companies, ours has 107.

Things we found while replicating:

1. **The ~95% accuracy is the majority-class baseline.** On the paper's test set, the SVM confusion
   matrix `[[1990, 0], [111, 0]]` shows it predicts "not bankrupt" for every company. 1990/2101 =
   0.9472, which is exactly the SVM's reported accuracy. Our run is the same. LR, the SVM, the NN and
   both boosting models catch **0 or 1 of the 107 bankrupt companies** (`fig3`, `fig5`). The SVM
   learning curve (`fig2`) shows the same thing: train accuracy is ~1.0, but CV accuracy never rises
   above the "always predict not bankrupt" line, however much data it gets.
2. **The log loss was computed on hard 0/1 predictions, not probabilities.** The paper's numbers
   only come out that way (error rate × 34.54). It is really a rescaled error rate, not a log loss.
3. **The SVM's train accuracy of 1.0 needs `gamma="auto"`.** That was scikit-learn's default in
   2019. Without feature scaling, the RBF kernel memorises the training set.
4. **We could not reproduce the boosting results (0.53 and 0.71).** Those scores are far *below*
   the trivial 0.947 baseline. Every reasonable XGBoost/LightGBM configuration we tried scores at
   least 0.949. Default LightGBM scores 0.973, the best accuracy of any model. The low numbers were
   most likely a bug in the original code; the paper itself says boosting "needs further inspection".

## 2. Improved approach

We kept the same data, split and model families, with these changes:

| Problem in the paper | Our fix |
|---|---|
| Only 4.7% positives, so models learn to always say "no" | `class_weight="balanced"` (LR, SVM, RF, LightGBM) and `scale_pos_weight` (XGBoost) |
| Accuracy hides the failure | ROC-AUC, PR-AUC, recall, precision, F1 and balanced accuracy; stratified 5-fold CV on ROC-AUC |
| Features are very skewed with huge outliers, and scaling was skipped | `QuantileTransformer(output_distribution="normal")` for LR, SVM, MLP and Gaussian NB |
| Mean imputation is pulled around by outliers | Median imputation. XGBoost and LightGBM handle NaN natively. |
| "XGBoost" used a *linear* booster | Real gradient-boosted trees, plus a Random Forest |

Results on the same test set (107 bankrupt out of 2,101), sorted by test ROC-AUC:

| Classifier | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Recall | Precision | F1 | Bankrupt caught |
|---|---|---|---|---|---|---|---|
| **LightGBM** | 0.915 | **0.937** | **0.729** | 0.56 | 0.83 | **0.67** | 60 / 107 |
| **XGBoost** | 0.913 | 0.933 | 0.702 | 0.67 | 0.58 | 0.62 | 72 / 107 |
| Neural Network (MLP 32-16) | 0.832 | 0.891 | 0.632 | 0.52 | 0.76 | 0.62 | 56 / 107 |
| Random Forest | 0.853 | 0.891 | 0.489 | 0.07 | 0.78 | 0.12 | 7 / 107 |
| SVM (RBF) | 0.796 | 0.826 | 0.284 | 0.61 | 0.21 | 0.31 | 65 / 107 |
| Logistic Regression | 0.789 | 0.816 | 0.242 | 0.72 | 0.13 | 0.22 | 77 / 107 |
| Naive Bayes (Gaussian) | 0.715 | 0.752 | 0.131 | 0.68 | 0.11 | 0.20 | 73 / 107 |

A random guess scores ROC-AUC 0.5 and PR-AUC 0.05. The paper's SVM caught 1 of 107.

**Conclusions**
- **Gradient-boosted trees are the best models on this data.** That is the opposite of the paper's
  conclusion, and it agrees with the dataset's original study (Zięba et al., 2016). Trees ignore the
  scale of the features, are robust to outliers and handle missing values natively.
- **Logistic Regression, the SVM and Naive Bayes reach high recall only at very low precision.** They flag many healthy companies to catch the bankrupt ones.
- **The Random Forest ranks companies well (AUC 0.89) but its 0.5 threshold is too strict.** Recall
  is low. Lowering the threshold would trade precision for recall; the PR curve in `fig4` shows
  that trade-off.
- **On imbalanced problems, always report a metric that the "always predict the majority class"
  baseline cannot win.** ROC-AUC, PR-AUC, recall and F1 all qualify.

All hyperparameters in `improved.py` were set once, a priori, and **not** tuned on the test set.

## Figures

| | |
|---|---|
| ![](results/figures/fig1_class_balance.png) | ![](results/figures/fig3_accuracy_vs_recall.png) |
| ![](results/figures/fig5_confusion_matrices.png) | ![](results/figures/fig2_svm_learning_curve.png) |

![](results/figures/fig4_roc_pr_curves.png)

## References

1. N. A. Patil, *Corporate Bankruptcy Prediction*, CS229 project, Stanford, Fall 2019.
2. M. Zięba, S. Tomczak, J. Tomczak, *Ensemble boosted trees with synthetic features generation in
   application to bankruptcy prediction*, Expert Systems with Applications, 2016.
3. UCI Machine Learning Repository: Polish Companies Bankruptcy Data (dataset #365).
