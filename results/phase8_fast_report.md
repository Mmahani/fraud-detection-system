# Phase 8 — Fast Imbalance Handling and Hyperparameter Tuning Report

## Execution status

The optimized Phase 8 script was executed successfully in the agent's environment using the real processed IEEE-CIS Fraud Detection project files. The user's Windows machine was not used for the computation. The test set was loaded but was used only in the final evaluation block.

The speed optimization used **3-fold StratifiedKFold** for the class-weight experiments and RandomizedSearchCV, with `n_iter=3`. Random Forest used `n_jobs=-1`, while the outer search used `n_jobs=1` to avoid excessive nested memory consumption. The search cross-validation was therefore reduced to 3-fold, which shortens execution time but makes the search estimate somewhat less stable.

After the search, the tuned model was evaluated with a separate **5-fold training-only** cross-validation (`final_validation_cv: 5`, `final_validation_executed: true`). This final validation uses `X_train` and `y_train` only; the test set is not involved.

## Data verification

| Item | Result |
|---|---:|
| Training matrix | `(472432, 495)` |
| Test matrix | `(118108, 495)` |
| Fraud count in training | `16530` |
| Fraud percentage | `3.498916246147594%` |
| Imbalance ratio | `27.580278281911674` legitimate per fraud |
| Test used during tuning | `No` |

## Class-weight experiment — 3-fold CV on training data

| Strategy | PR-AUC | F1 | Recall | Precision | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| None | 0.633806 | 0.564524 | 0.406473 | 0.923852 | 0.893788 |
| Balanced | 0.644903 | 0.617321 | 0.486691 | 0.843908 | 0.912446 |

The selected strategy was **`Balanced`**, based primarily on PR-AUC with F1 considered as a secondary criterion.

## RandomizedSearchCV

The search used 3 folds and 3 randomly selected parameter combinations, for 9 total fits. The actual search duration was **2785.52 seconds**.

| Parameter | Selected value |
|---|---|
| `n_estimators` | `20` |
| `max_depth` | `None` |
| `min_samples_leaf` | `2` |
| `min_samples_split` | `5` |
| `max_features` | `0.5` |
| `class_weight` | `balanced` |
| Best 3-fold CV PR-AUC | `0.6847187000753766` |

## Final five-fold training-only validation

The tuned configuration was re-validated with 5-fold `StratifiedKFold` on the training set only. These values, not the test metrics, drive the readiness decision.

| Metric | Mean | Std (ddof=1) |
|---|---:|---:|
| PR-AUC | 0.704887 | 0.008989 |
| F1 | 0.666683 | 0.007884 |
| Recall | 0.608227 | 0.010642 |
| Precision | 0.737667 | 0.007506 |
| ROC-AUC | 0.931818 | 0.003393 |

## Final test evaluation

The tuned model was retrained on the complete training set and then evaluated once on the untouched test set.

| Metric | Untuned Random Forest | Tuned Random Forest | Difference |
|---|---:|---:|---:|
| Accuracy | 0.980958 | 0.980154 | -0.000804 |
| Balanced Accuracy | 0.767213 | 0.813782 | +0.046569 |
| Precision | 0.868256 | 0.758601 | -0.109656 |
| Recall | 0.537382 | 0.634890 | +0.097508 |
| F1 | 0.663877 | 0.691254 | +0.027377 |
| ROC-AUC | 0.940513 | 0.937292 | -0.003220 |
| PR-AUC | 0.722463 | 0.730518 | +0.008055 |
| False positives | 337 | 835 | +498 |
| False negatives | 1912 | 1509 | -403 |
| True positives | 2221 | 2624 | +403 |
| True negatives | 113638 | 113140 | -498 |

## Decision

The readiness rule is based only on the 5-fold training-only validation (positive mean PR-AUC and F1, and a PR-AUC standard deviation below the mean), never on the test set:

> **READY FOR THRESHOLD OPTIMIZATION**

For reference, the tuned model's test PR-AUC improved from **0.722463** to **0.730518** (absolute increase **0.008055**) and F1 from **0.663877** to **0.691254**. It also increased recall and reduced false negatives, but produced more false positives and lower precision. This trade-off is expected from class balancing and should be addressed in Phase 9 through threshold optimization. These test values are reported after the model was frozen and did not influence the readiness decision.

## Saved artifacts

The following files were created in `/home/ubuntu/fraud_local`:

- `models/best_tuned_model.pkl`
- `results/best_model_parameters.json`
- `results/hyperparameter_search_results.csv`
- `results/phase8_tuned_vs_untuned_test.csv`
- `results/final_model_comparison_phase8.csv`
- `results/phase8_fast_summary.json`

**Phase 8 complete.** Later phases cover model validation, threshold optimization, and packaging.
