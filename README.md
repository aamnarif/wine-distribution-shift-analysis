# 🍷 Wine Dataset — Distribution Shift Robustness Analysis

A data analytics assignment (DSA8040) investigating how covariate shift affects three classifiers — Logistic Regression, Gaussian Naïve Bayes, and Random Forest — on the UCI Wine dataset. The analysis spans EDA, model inductive bias theory, failure mode diagnosis, and mitigation strategies.

---

## Overview

The UCI Wine dataset contains 13 chemical measurements from wine samples of three grape cultivars (classes 0, 1, 2). The task is to classify wines by cultivar under two conditions:

- **Clean test set** — drawn from the same distribution as training
- **Shifted test set A** — five features are systematically shifted by a constant ±1.50σ offset, simulating a sensor calibration artefact

| Split | Samples |
|---|---|
| Training set | 142 |
| Clean test set | 36 |
| Shifted test set A | 36 |

---

## Key Results

### Model Accuracy — Clean vs Shifted Test Set

| Model | Clean Accuracy | Shifted Accuracy | Drop |
|---|---|---|---|
| Random Forest | **1.000** | 0.944 | −0.056 |
| Logistic Regression | 0.972 | 0.972 | 0.000* |
| Gaussian Naïve Bayes | 0.972 | 0.861 | −0.111 |

> *Logistic Regression's zero drop is misleading — per-class F1 reveals a Class 0 drop of −0.043 offset by a Class 2 gain of +0.053. Overall accuracy masks asymmetric error redistribution.

### Shifted Features

| Feature | Raw Offset | Shift (σ units) | OOD? |
|---|---|---|---|
| `ash` | +0.40 | +1.50σ | No |
| `alcalinity_of_ash` | +5.05 | +1.50σ | Yes |
| `flavanoids` | −1.42 | −1.50σ | Yes |
| `proanthocyanins` | +0.87 | +1.50σ | Yes |
| `proline` | −450.65 | −1.50σ | Yes |

---

## Project Structure
wine-distribution-shift-analysis/
- DSA8040_Data_Analytics_AI_aamna_arif.ipynb 
- X_train.csv               (Training features)
- y_train.csv               (Training labels)
- X_test_clean.csv          (Clean test features)
- X_test_shifted_A.csv      (Shifted test features)
- y_test.csv                (Test labels)

> **Note:** Place all CSV files in the working directory before running the notebook.

---

## Sections Breakdown

### Section 1 — Exploratory Data Analysis
- Dataset overview, class distribution, and missing value checks
- Distribution shift quantification: per-feature mean offsets in σ units
- KDE plots and histograms comparing train / clean test / shifted test
- Out-of-distribution (OOD) range analysis — 4 features exceed training bounds
- Per-class feature analysis showing non-uniform shift impact across cultivars
- PCA projection visualising cluster migration under shift
- Feature correlation structure analysis (shift is translation-invariant → correlations preserved)

### Section 2 — Model Assumptions and Inductive Bias
- Theoretical analysis of how each model's inductive bias responds to covariate shift
- Logistic Regression: weight-based sensitivity to absolute feature values
- Gaussian Naïve Bayes: likelihood miscalibration from independence assumption violations
- Random Forest: localised threshold-based degradation; robustness reserve from unshifted features

### Section 3 — Model Implementation and Classification Metrics
- 5-fold stratified cross-validation hyperparameter tuning (theoretically motivated grids)
- Overall accuracy comparison across clean and shifted test sets
- Per-class confusion matrices and F1 scores for all three models on both test sets
- Random Forest feature importances: 62.6% of signal in unshifted features

### Section 4 — Failure Modes and Robustness
- Detailed failure mode analysis for each model under shift
- Random Forest: localised Class 0 → Class 1 errors from `proline` threshold crossings
- Logistic Regression: error redistribution masking real degradation
- Gaussian Naïve Bayes: systematic likelihood miscalibration across all classes
- Five proposed mitigation strategies (drift detection, test-time correction, feature regularisation)

### Section 5 — Synthesis and Judgement
- Cross-model comparison and ranking of robustness
- Key takeaway: robustness is a design decision, not a coincidence

### Section 6 — Inductive Bias Stress Test
- Targeted adversarial shift designed to exploit the Gaussian Naïve Bayes' core assumption
- Demonstrates that robustness depends on whether a shift directly violates model assumptions

---

## Setup

### Requirements

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

| Package | Role |
|---|---|
| `scikit-learn` | Logistic Regression, GNB, Random Forest, GridSearchCV, metrics |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting (KDE, histograms, confusion matrices, PCA) |
| `seaborn` | Heatmaps and KDE overlays |

### Run

```bash
jupyter notebook DSA8040_Data_Analytics_AI_aamna_arif_40503414.ipynb
```

---

## References

- [Breiman (2001) — Random Forests](https://doi.org/10.1023/A:1010933404324)
- [Pedregosa et al. (2011) — scikit-learn](https://www.jmlr.org/papers/v12/pedregosa11a.html)
- [Sugiyama et al. (2007) — Covariate Shift Adaptation](https://www.jmlr.org/papers/v8/sugiyama07a.html)
- [Guo et al. (2017) — Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)
- [Klaise et al. (2022) — Alibi Detect](https://github.com/SeldonIO/alibi-detect)
- [Jolliffe (2002) — Principal Component Analysis](https://link.springer.com/book/10.1007/b98835)
- [Hastie, Tibshirani & Friedman (2009) — Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)

---

## License

This project is licensed under the [MIT License](LICENSE).
