# Notebooks

The notebooks document the research path in execution order. They are committed
with outputs cleared; run them locally after obtaining the IEEE-CIS data.

## Order

| # | Notebook | Purpose | Key outputs |
|---|---|---|---|
| 01 | [`01_data_inspection.ipynb`](01_data_inspection.ipynb) | Profile the raw transaction and identity files | Row counts, dtypes, missingness |
| 02 | [`02_data_integration.ipynb`](02_data_integration.ipynb) | Merge transaction and identity tables | `data/processed/merged_data.csv.gz` |
| 03 | [`03_eda.ipynb`](03_eda.ipynb) | Exploratory analysis | Fraud rate, distributions, time patterns |
| 04 | [`04_preprocessing.ipynb`](04_preprocessing.ipynb) | Leakage-safe preprocessing and split | `X_train`, `X_test`, `y_train`, `y_test`, `preprocessor.pkl` |
| 05 | [`05_baseline_model.ipynb`](05_baseline_model.ipynb) | SGD baseline | Baseline metrics |
| 06 | [`06_model_comparison.ipynb`](06_model_comparison.ipynb) | Compare SGD, Random Forest, Gradient Boosting, XGBoost | `results/model_comparison.csv` |
| 07 | [`07_model_validation_audit.ipynb`](07_model_validation_audit.ipynb) | Validation and leakage audit | Audit findings |
| 08 | [`08_imbalance_and_tuning.ipynb`](08_imbalance_and_tuning.ipynb) | Final tuning path | `models/best_tuned_model.pkl`, Phase 8 summaries |
| 09 | [`09_threshold_optimization.ipynb`](09_threshold_optimization.ipynb) | OOF threshold selection | Frozen threshold `0.35` |
| 10 | [`10_shap_explainability.ipynb`](10_shap_explainability.ipynb) | Global and local SHAP | SHAP CSVs and plots |
| 11 | [`11_final_model_packaging.ipynb`](11_final_model_packaging.ipynb) | Packaging and inference checks | Final model artifacts |

`08_model_improvement.ipynb` is an **earlier experiment**, not the authoritative
tuning path. Its `models/improved_model.pkl` artifact has poor test performance
and is not used for inference.

## Prerequisites

1. Obtain the IEEE-CIS data from Kaggle after accepting the competition rules.
2. Place `train_transaction.csv` and `train_identity.csv` in `data/raw/`.
3. Install the full requirements:

```powershell
python -m pip install -r requirements.txt
```

4. Start Jupyter from the project root:

```powershell
python -m jupyter lab
```

Run notebooks 01–04 before the modeling notebooks. `data/` is ignored by Git.

## Reproducibility notes

- Seeds are fixed at `42` where applicable.
- The model was trained with `scikit-learn 1.9.0`.
- The test split is a retrospective holdout reused across phases; see
  [`docs/RESULTS.md`](../docs/RESULTS.md).
- Some Phase 6 notebook outputs were refreshed from saved test probabilities
  rather than a full retrain because the 8 GB environment could not complete it.
  See [`results/phase8_evaluation_audit.md`](../results/phase8_evaluation_audit.md).

## Outputs

Generated outputs under `data/processed/` and large model files are ignored by
Git. Tracked result artifacts live in [`results/`](../results/README.md) and are
inventoried in [`artifacts/manifest.json`](../artifacts/manifest.json).
