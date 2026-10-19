# Phase 8 Evaluation and Performance Audit

## Conclusion

The saved Phase 8 test metrics are numerically consistent with the current evaluation function. Recomputing predictions from `best_tuned_model.pkl` on the saved `X_test` and `y_test` produced a maximum absolute difference of `0.0000000491` from the saved summary; the difference is due to floating-point or model-version serialization precision and is practically negligible.

## Metric verification

The function `metrics(model, X, y)` correctly obtains hard predictions with `model.predict(X)`, class-1 probabilities with `model.predict_proba(X)[:, 1]`, and computes accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and the confusion-matrix counts. PR-AUC is calculated with `average_precision_score`, which matches the project primary metric.

The tuned test values were recomputed as follows: Accuracy `0.9801537576`, Balanced Accuracy `0.8137818712`, Precision `0.7586007517`, Recall `0.6348899105`, F1 `0.6912539515`, ROC-AUC `0.9372922418`, PR-AUC `0.7305181402`, TN `113140`, FP `835`, FN `1509`, and TP `2624`. The saved values match to within `4.91e-08`.

## Performance comparison

Compared with the untuned Random Forest, the tuned model increased PR-AUC from `0.7224630422` to `0.7305181893` and F1 from `0.6638768495` to `0.6912539515`. Recall increased from `0.5373820469` to `0.6348899105`, while Precision decreased from `0.8682564504` to `0.7586007517`. False positives increased from `337` to `835`, and false negatives decreased from `1912` to `1509`. ROC-AUC decreased from `0.9405127387` to `0.9372922450`. These values are consistent with the saved comparison table.

## Important methodological findings

The test set is evaluated after the final tuned model is fitted, and the search scoring uses `average_precision`. The code therefore computes the reported test metrics correctly and does not use test data inside `RandomizedSearchCV`.

An earlier draft declared `cv_final = StratifiedKFold(n_splits=5, ...)` and recorded `final_validation_cv: 5` without executing it. The current notebook source does pass `cv_final` to `cross_validate` on `X_train`/`y_train`, and `results/phase8_fast_summary.json` now records the executed results in `cv_summary_5_fold_training_only` and `cv_std_5_fold_training_only`, with `final_validation_executed: true`. The label is therefore no longer a documentation mismatch.

A second issue was that an earlier draft used the final test PR-AUC and F1 for the `READY FOR THRESHOLD OPTIMIZATION` decision. The current notebook source derives `improved` from the 5-fold training-only validation (`cv_final_summary`, `cv_final_std`) and records `readiness_decision_based_on_test: false`. The test set is therefore reserved for the final report after the model is frozen.

The train/test F1 gap is calculated from a training-set metric and a test-set metric. This is useful as a diagnostic, but it is not an unbiased generalization estimate because the tuned model was fitted on all training rows before its training F1 was measured. The large observed gap should be described as an overfitting diagnostic, not as cross-validated performance.

## Resolution status

The numerical results are valid and consistent. The two methodological findings above have been addressed in the current artifacts: the five-fold training-only validation is executed and saved, and the readiness decision is based on training cross-validation rather than test metrics. `results/phase8_fast_report.md` and `results/phase8_fast_summary.json` were updated accordingly.
