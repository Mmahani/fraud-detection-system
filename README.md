# Fraud Detection System — IEEE-CIS

End-to-end machine learning project for credit-card fraud detection using the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset.

The project covers data integration, exploratory analysis, leakage-safe
preprocessing, baseline modeling, model comparison, validation, threshold
optimization, SHAP explainability, and model packaging.

> **Current status:** Phase 11 model artifacts are available and inference works.
> Engineering hardening (API, automated tests, CI/CD, final documentation) is in
> progress. See [Known limitations](#known-limitations).

---

## Table of contents

- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [Final model](#final-model)
- [Results](#results)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Notebook workflow](#notebook-workflow)
- [Reproducibility notes](#reproducibility-notes)
- [Known limitations](#known-limitations)
- [Data and license](#data-and-license)

---

## Dataset

The IEEE-CIS dataset contains anonymized e-commerce transactions. The raw files
are not committed to this repository because of their size.

Place the following files in `data/raw/`:

```text
data/raw/
├── train_transaction.csv   (~651 MB)
└── train_identity.csv      (~25 MB)
```

After integration:

| Item | Value |
|---|---:|
| Rows | 590,540 |
| Raw features after merge | 434 |
| Fraud rate | 3.4989% |
| Training rows (80%) | 472,432 |
| Test rows (20%) | 118,108 |
| Transformed features | 495 |

The current split is a stratified random split (`random_state=42`). A
time-based validation split is planned before final delivery.

---

## Repository structure

```text
fraud-detection-system/
├── data/
│   ├── raw/                 # not tracked by Git
│   └── processed/           # not tracked by Git
├── models/
│   ├── best_tuned_model.pkl
│   ├── preprocessor.pkl
│   ├── threshold_config.json
│   ├── model_metadata.json
│   └── explainability_config.json
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_data_integration.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_preprocessing.ipynb
│   ├── 05_baseline_model.ipynb
│   ├── 06_model_comparison.ipynb
│   ├── 07_model_validation_audit.ipynb
│   ├── 08_imbalance_and_tuning.ipynb
│   ├── 08_model_improvement.ipynb      # earlier experiment, not the final path
│   ├── 09_threshold_optimization.ipynb
│   ├── 10_shap_explainability.ipynb
│   └── 11_final_model_packaging.ipynb
├── results/                 # metrics, reports and plots
├── src/
│   └── prediction.py        # inference and explanation utilities
├── .gitignore
├── README.md
└── requirements.txt
```

Only the final inference artifacts are intended to be tracked in Git:

- `models/best_tuned_model.pkl`
- `models/preprocessor.pkl`
- `models/threshold_config.json`
- `models/model_metadata.json`
- `models/explainability_config.json`

`models/random_forest.pkl` (157 MB) and the processed matrices are intentionally
ignored.

---

## Final model

| Property | Value |
|---|---|
| Algorithm | `RandomForestClassifier` |
| Estimators | 20 |
| `max_depth` | `None` |
| `min_samples_leaf` | 2 |
| `min_samples_split` | 5 |
| `max_features` | 0.5 |
| `class_weight` | `balanced` |
| Raw input features | 434 |
| Transformed features | 495 |
| Classification threshold | **0.35** |

The threshold was selected from out-of-fold (OOF) training probabilities using a
simulated cost of `COST_FP=1` and `COST_FN=5`, with F1 as a tie-breaker. These
costs are illustrative, not verified business costs.

Risk levels are presentation bands, not the classification threshold:

- Low: `probability < 0.30`
- Medium: `0.30 <= probability < 0.70`
- High: `probability >= 0.70`

---

## Results

Retrospective test-set metrics at the frozen threshold **0.35**:

| Metric | Value |
|---|---:|
| Accuracy | 0.972161 |
| Balanced accuracy | 0.852896 |
| Precision | 0.582119 |
| Recall | 0.724655 |
| F1 | 0.645613 |
| ROC-AUC | 0.937292 |
| PR-AUC (Average Precision) | 0.730518 |
| TN / FP / FN / TP | 111,825 / 2,150 / 1,138 / 2,995 |

At threshold 0.50, for comparison:

| Metric | Value |
|---|---:|
| Precision | 0.757558 |
| Recall | 0.636584 |
| F1 | 0.691822 |

> **Metric definition:** all PR-AUC values in this repository should be read as
> **Average Precision** (`sklearn.metrics.average_precision_score`). Earlier
> notebooks used trapezoidal PR-AUC (`auc(recall, precision)`), which produces
> slightly different numbers. Do not compare PR-AUC across phases unless the
> implementation is identical.

The test set has been evaluated in several phases. The values above are therefore
retrospective and should not be treated as a pristine final holdout.

---

## Installation

Tested with Python 3.12.6 on Windows.

```powershell
cd "C:\Users\HP\Desktop\AI Projct\fraud-detection-system"

py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The model artifacts were serialized with **scikit-learn 1.9.0**. Installing the
pinned version avoids the `InconsistentVersionWarning` seen under 1.8.0.

---

## Quick start

### Batch prediction from processed data

```python
import pandas as pd

from src.prediction import predict_batch

df = pd.read_csv("data/processed/merged_data.csv.gz", nrows=5)
result = predict_batch(df)
print(result.to_string(index=False))
```

### Single transaction

```python
import pandas as pd

from src.prediction import predict_transaction

df = pd.read_csv("data/processed/merged_data.csv.gz", nrows=1)
result = predict_transaction(df)
print(result)
```

### Explanation (requires `shap`)

```python
import pandas as pd

from src.prediction import explain_transaction

df = pd.read_csv("data/processed/merged_data.csv.gz", nrows=1)
result = explain_transaction(df, top_n=3)
print(result)
```

### Loading the preprocessor

Load the preprocessor through the project helper, not directly with `joblib`:

```python
from src.prediction import load_preprocessor

preprocessor = load_preprocessor()
```

Direct `joblib.load("models/preprocessor.pkl")` fails because the artifact
references compatibility symbols defined in `src.prediction`.

---

## Notebook workflow

Run the notebooks from the `notebooks/` directory, in this order:

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_data_inspection.ipynb` | Raw data profile |
| 02 | `02_data_integration.ipynb` | Merge transaction and identity data |
| 03 | `03_eda.ipynb` | Exploratory analysis |
| 04 | `04_preprocessing.ipynb` | Leakage-safe preprocessing and split |
| 05 | `05_baseline_model.ipynb` | Baseline model |
| 06 | `06_model_comparison.ipynb` | Model comparison |
| 07 | `07_model_validation_audit.ipynb` | Validation and leakage audit |
| 08 | `08_imbalance_and_tuning.ipynb` | Final tuning path |
| 09 | `09_threshold_optimization.ipynb` | OOF threshold selection |
| 10 | `10_shap_explainability.ipynb` | SHAP artifacts |
| 11 | `11_final_model_packaging.ipynb` | Packaging and inference checks |

`08_model_improvement.ipynb` is an earlier experiment and is not the authoritative
tuning path. Its `models/improved_model.pkl` artifact has poor test performance
and is not used for inference.

---

## Reproducibility notes

- Raw and processed data are excluded from Git. Regenerate processed data with
  notebooks 01–04.
- The final model is tracked in Git; large comparison models are not.
- The preprocessor must be loaded through `src.prediction`.
- Random seeds are fixed at `42` where applicable.
- The model was trained with scikit-learn 1.9.0.
- The current test metrics are not a clean final holdout and must be reported as
  retrospective.

---

## Known limitations

- The notebooks now resolve the project root from the current working directory,
  but they have not been re-executed end-to-end after that change.
- Phase 10 CSV artifacts referenced by `models/explainability_config.json` are
  missing from `results/`.
- The model shows a large train/test gap (train F1 0.9569 vs test F1 0.6913), so
  overfitting remains a concern.
- The current split is random and stratified, not time-based; `TransactionDT` is
  among the most important SHAP features, so temporal leakage is possible.
- The test set has been reused across phases and is not a pristine holdout.
- The probabilities are not calibrated; risk bands are heuristic.
- There is no API, automated test suite, CI pipeline, or model card yet.
- `models/xgboost.pkl` requires the optional `xgboost` package.

---

## Data and license

The dataset is provided by Vesta Corporation for the IEEE-CIS Fraud Detection
competition on Kaggle. Refer to the competition rules and dataset license before
redistributing the data.

The source code in this repository does not yet declare a license. Add an explicit
license before public distribution.
