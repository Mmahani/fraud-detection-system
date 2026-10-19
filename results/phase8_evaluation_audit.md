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

However, the code declares `cv_final = StratifiedKFold(n_splits=5, ...)` and records `final_validation_cv: 5`, but it never passes `cv_final` to `cross_validate` or otherwise executes a five-fold final validation. Therefore, the label `final_validation_cv: 5` is a documentation mismatch and must not be described as an executed validation result.

A second issue is that the variable `improved` uses the final test PR-AUC and F1 to decide `READY FOR THRESHOLD OPTIMIZATION`. This does not alter the metric calculations, but it uses the test set to make a model-readiness decision. Under a strict leakage-control protocol, that readiness decision should be based on training cross-validation results; the test set should be reserved for a final report after the model is frozen.

The train/test F1 gap is calculated from a training-set metric and a test-set metric. This is useful as a diagnostic, but it is not an unbiased generalization estimate because the tuned model was fitted on all training rows before its training F1 was measured. The large observed gap should be described as an overfitting diagnostic, not as cross-validated performance.

## Recommended wording correction

The numerical results are valid and consistent, but the Phase 8 conclusion should be changed from a test-driven `READY FOR THRESHOLD OPTIMIZATION` rule to a training-CV-based readiness rule. The reported five-fold final validation should either be executed and saved or removed from the metadata. No test metrics should be used to select, reject, or approve the model before the final locked evaluation.
