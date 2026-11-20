# Comparative Machine Learning Modeling

One repository, two problems, the same question: **how do different machine learning models behave on small, real-world tabular data?**

The two projects keep the question fixed and change the problem type. One predicts a category, the other predicts a number. Everything else — the notebook structure, the way data is cleaned, the leakage-safe pipelines, the cross validation and the held out test set — is deliberately held constant so the comparison means something.

| Project | Task | Target | Data | Best model in this run |
| ------- | ---- | ------ | ---- | ---------------------- |
| [Liver Patient Records Classification](liver_patient_records_classification/README.md) | Binary classification | Liver disease, yes or no | Indian Liver Patient Dataset, 583 rows, 10 features | Random Forest, F1 ~0.79 |
| [Red Wine Quality Regression](red_wine_quality_regression/README.md) | Regression | Quality score, 3 to 8 | Red Wine Quality (Kaggle / UCI), 1,599 rows, 11 features | Random Forest, CV RMSE 0.658 / R² 0.342 |


## Repository Layout

```text
liver_patient_records_classification/
  data/raw/            untouched source CSV
  data/processed/      cleaned CSV
  data/outputs/        metrics and comparison CSVs
  notebooks/           01 cleaning → 02 EDA → 03 models → 04 comparison
  assets/images/       generated charts
  src/

red_wine_quality_regression/
  data/raw/            untouched source CSV
  data/processed/      cleaned CSV
  data/outputs/        metrics, cleaning summary and comparison CSVs
  notebooks/           01 cleaning → 02 EDA → 03 models → 04 comparison
  assets/images/       generated charts
  src/
```

## What They Share

- **Four notebooks in the same order:** clean, explore, train, compare.
- **Raw data is never overwritten.** Cleaning writes a separate processed CSV.
- **Preprocessing lives inside the `Pipeline`.** Imputation, scaling and encoding are fitted on training folds only.
- **The same split and seed:** 20% held out, `random_state=42`.
- **The same cross validation style:** 5 folds, shuffled, seeded.
- **The same five model families:** linear model, decision tree, k-nearest neighbours, kernel model and random forest.
- **A final comparison notebook** that looks at uncertainty and when to pick each model, not just the top score.

## Where They Differ

| Aspect | Liver classification | Wine regression |
| ------ | -------------------- | --------------- |
| Target | Binary label `Dataset` (1 or 2) | Continuous `quality` score (3-8) |
| Rows used | 570 after removing 13 duplicates | 1,359 after removing 240 duplicates |
| Missing values | 4 gaps in Albumin/Globulin Ratio | None |
| Categorical input | `Gender`, one-hot encoded | None, all features are numeric |
| Imbalance | About 71:29, handled with SMOTE | Not applicable |
| Scaling | StandardScaler | StandardScaler |
| Cross validation | `StratifiedKFold` | `KFold` |
| Models | Logistic Regression, Decision Tree, KNN, SVC, Random Forest | Linear Regression, Decision Tree, KNN, SVR, Random Forest |
| Extra reference model | None | Mean-prediction `DummyRegressor` |
| Metrics | Accuracy, F1, ROC-AUC | RMSE, MAE, R² |
| Model diagnostics | Confusion matrix, ROC curve | Predicted vs actual, residuals |
| Main data challenge | Imbalance and missing values | Duplicates, skew and a narrow target range |

The imbalance is the clearest split. SMOTE cannot be used for a numeric target, so the wine project relies on a shared pipeline and a baseline model instead. The liver project needs class balancing and an encoded categorical feature, and it is scored with classification metrics.

## Results at a Glance

### Liver — F1 (higher is better)

| Model | F1 |
| ----- | --: |
| Random Forest | ~0.79 |
| Decision Tree | ~0.76 |
| SVM RBF | ~0.74 |
| Logistic Regression | ~0.71 |
| KNN | ~0.68 |

### Wine — CV RMSE and R² (lower RMSE / higher R² is better)

| Model | CV RMSE | CV R² |
| ----- | ------: | ----: |
| Random Forest | 0.658 | 0.342 |
| Linear Regression | 0.668 | 0.321 |
| KNN (k=5) | 0.713 | 0.228 |
| SVR (RBF) | 0.723 | 0.206 |
| Decision Tree | 0.760 | 0.121 |
| Mean baseline | 0.818 | -0.011 |

In both studies Random Forest came out on top, but neither result is a clean sweep. On the liver data the Decision Tree is close and far easier to explain. On the wine data Linear Regression is within the fold-to-fold spread of Random Forest. Both READMEs make that point explicitly: the best model depends on whether accuracy, interpretability, simplicity or training cost matters most.

## What Each Project Teaches

**Liver classification** is the place to study:

- Working with missing values and a categorical feature
- Imbalanced classes and why SMOTE belongs inside a pipeline
- Class-balanced model fitting
- Classification metrics: F1, ROC-AUC and confusion matrices
- Interpretability of linear coefficients and a single tree

**Wine regression** is the place to study:

- Duplicate detection and its effect on honest evaluation
- Skewed, differently scaled measurements
- A discrete target treated as continuous, and what that assumes
- Regression metrics: RMSE, MAE and R²
- Residual analysis and a mean-baseline sanity check
- Correlated feature pairs and train-fitted preprocessing

Together they show that the modelling skeleton transfers across problem types, while the preprocessing and the evaluation have to change with the target.

## How to Run

For either project, from inside its folder:

### Option 1: UV Recommended

```bash
uv sync
uv run jupyter lab
```

### Option 2: PIP

```bash
pip install -r requirements.txt
jupyter lab
```

Open notebooks `01` through `04` in order. Each project's README explains what every notebook does and records the results from its own run.
