# Corporate Bankruptcy Prediction: Replication and Improvement

This project replicates the CS229 (Fall 2019) project
[*Corporate Bankruptcy Prediction*](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26646282.pdf)
(N. A. Patil). It compares Logistic Regression, SVM, a Neural Network, Naive Bayes, XGBoost and
LightGBM on the [Polish Companies Bankruptcy](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)
dataset (3rd-year file: 10,503 companies, 64 financial ratios). It then tries to improve on the results.

## Setup

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Plan

- [ ] `src/data.py`: download and load the data, 80/20 train/test split
- [ ] `src/replicate.py`: reproduce the paper's Table 1
- [ ] `src/improved.py`: handle class imbalance, scaling, better metrics
- [ ] `src/plots.py`: figures
- [ ] Results and write-up
