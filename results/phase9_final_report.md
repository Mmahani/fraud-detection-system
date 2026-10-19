# Phase 9 — Threshold Optimization Report

## Model

The model is the Phase 8 tuned Random Forest loaded from `models/best_tuned_model.pkl`. Threshold selection used 5-fold out-of-fold probabilities from `X_train` and `y_train` only. The test set was used only after the threshold was frozen.

## OOF threshold selection

The OOF baseline at threshold 0.50 was: **Precision=0.736508, Recall=0.610103, F1=0.667373, FP=3608, FN=6445**.

The best F1 candidate was threshold **0.50** with Precision=0.736508, Recall=0.610103, and F1=0.667373.

The lowest simulated-cost candidate, under the initial scenario COST_FP=1 and COST_FN=5, was threshold **0.35** with Precision=0.566226, Recall=0.701633, F1=0.626699, FP=8885, FN=4932, and simulated cost=33545. These costs are illustrative and are not claimed to be real business costs.

## Final selected threshold

`FINAL_THRESHOLD = 0.35`

The threshold was selected from OOF training predictions using the simulated cost rule, with F1 as a tie-break. It was frozen before test probabilities were generated. The expected benefit is fewer false negatives under the assumed higher FN cost. The trade-off is potentially more false positives and operational workload.

## Final test performance

| Metric | Threshold 0.50 | Final threshold |
|---|---:|---:|
| Accuracy | 0.980154 | 0.972161 |
| Balanced Accuracy | 0.814598 | 0.852896 |
| Precision | 0.757558 | 0.582119 |
| Recall | 0.636584 | 0.724655 |
| F1 | 0.691822 | 0.645613 |
| ROC-AUC | 0.937292 | 0.937292 |
| PR-AUC | 0.730518 | 0.730518 |
| FP | 842 | 2150 |
| FN | 1502 | 1138 |
| TP | 2631 | 2995 |
| TN | 113133 | 111825 |

PR-AUC is threshold-independent, whereas F1, precision, recall, and confusion-matrix counts depend on the selected threshold.

## Risk levels

Low risk is probability < 0.30, medium risk is 0.30 to below 0.70, and high risk is probability >= 0.70. These are presentation boundaries, not the classification threshold. The prediction flow is: fraud probability → risk level → compare probability with the frozen classification threshold → Fraud or Legitimate.

## Decision

**THRESHOLD SUCCESSFULLY SELECTED.** The threshold is frozen at 0.35. Phase 10 must not begin until explicit confirmation.
