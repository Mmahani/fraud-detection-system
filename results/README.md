# Results

Metrics, reports, and plots produced by the research pipeline. All numbers are
retrospective on the reserved IEEE-CIS test split; read
[`docs/RESULTS.md`](../docs/RESULTS.md) for the interpretation caveats.

## Figures

| Figure | File | What it shows |
|---|---|---|
| Canonical model comparison | [`../docs/assets/model_comparison_pr_auc.svg`](../docs/assets/model_comparison_pr_auc.svg) | Average Precision from the tracked result files (current reference) |
| Historical model comparison | [`phase8_plots/model_pr_auc_comparison.png`](phase8_plots/model_pr_auc_comparison.png) | Archived Phase 8 plot using the older trapezoidal PR-AUC; audit only, not the canonical values |
| Time-bucket PR-AUC | [`phase12_plots/time_bucket_pr_auc.png`](phase12_plots/time_bucket_pr_auc.png) | Temporal stability diagnostic on the reserved test split |
| SHAP global importance | [`phase10_plots/shap_global_bar.png`](phase10_plots/shap_global_bar.png) | Mean absolute SHAP value by feature |
| SHAP beeswarm | [`phase10_plots/shap_beeswarm.png`](phase10_plots/shap_beeswarm.png) | Direction and spread of SHAP values |
| OOF threshold metrics | [`phase9_plots/oof_threshold_metrics.png`](phase9_plots/oof_threshold_metrics.png) | Precision, recall, and F1 versus threshold on OOF data |
| OOF precision-recall | [`phase9_plots/oof_precision_recall_curve.png`](phase9_plots/oof_precision_recall_curve.png) | OOF precision-recall curve |
| OOF error counts | [`phase9_plots/oof_error_counts.png`](phase9_plots/oof_error_counts.png) | False positives and false negatives versus threshold |

## Metric and evaluation files

| File | Contents |
|---|---|
| [`final_threshold_test_results.csv`](final_threshold_test_results.csv) | Frozen-threshold test metrics at 0.35 |
| [`final_model_comparison_phase8.csv`](final_model_comparison_phase8.csv) | Tuned versus untuned Phase 8 comparison |
| [`model_comparison.csv`](model_comparison.csv) | Phase 6 model comparison with canonical PR-AUC |
| [`phase8_tuned_vs_untuned_test.csv`](phase8_tuned_vs_untuned_test.csv) | Test-set deltas between tuned and untuned models |
| [`threshold_analysis.csv`](threshold_analysis.csv) | Threshold sweep |
| [`threshold_constraints_oof.csv`](threshold_constraints_oof.csv) | Candidate thresholds under OOF constraints |
| [`threshold_sensitivity_test.csv`](threshold_sensitivity_test.csv) | Test-set sensitivity around the frozen threshold |
| [`threshold_050_vs_final_test.csv`](threshold_050_vs_final_test.csv) | Threshold 0.50 versus 0.35 |
| [`best_model_parameters.json`](best_model_parameters.json) | Selected hyperparameters |
| [`hyperparameter_search_results.csv`](hyperparameter_search_results.csv) | Randomized search results |
| [`improvement_comparison.csv`](improvement_comparison.csv) | Earlier improvement experiment |
| [`test_probabilities_metadata.json`](test_probabilities_metadata.json) | Shapes, versions, and PR-AUC values for the local probability arrays |

## Reports

| File | Contents |
|---|---|
| [`phase8_fast_report.md`](phase8_fast_report.md) | Phase 8 tuning report |
| [`phase8_fast_summary.json`](phase8_fast_summary.json) | Phase 8 metrics and training-only CV |
| [`phase8_evaluation_audit.md`](phase8_evaluation_audit.md) | Metric consistency and methodology audit |
| [`phase9_final_report.md`](phase9_final_report.md) | Threshold selection and final operating point |
| [`phase10_final_report.md`](phase10_final_report.md) | SHAP configuration and verification |
| [`phase12_summary.json`](phase12_summary.json) | Time-bucket summary |
| [`phase12_time_validation.md`](phase12_time_validation.md) | Time-bucket diagnostic report |
| [`phase12_time_validation.csv`](phase12_time_validation.csv) | Per-bucket metrics and fraud rates |
| [`metrics_definition_audit.md`](metrics_definition_audit.md) | PR-AUC definition audit |
| [`metrics_definition_audit.csv`](metrics_definition_audit.csv) | Numerical audit table |

## Explainability files

| File | Contents |
|---|---|
| [`shap_global_importance.csv`](shap_global_importance.csv) | Full global ranking |
| [`shap_global_importance_top20.csv`](shap_global_importance_top20.csv) | Top 20 features |
| [`shap_fraud_direction.csv`](shap_fraud_direction.csv) | Directional contributions |
| [`shap_local_explanations.csv`](shap_local_explanations.csv) | Local examples, including error cases |

## Tracked versus local-only

The `.npy` probability arrays are **ignored by Git** because of their size:

- `results/oof_proba.npy`
- `results/test_proba_{sgd,rf,gb,xgb}.npy`

Their shapes and library versions are recorded in
[`test_probabilities_metadata.json`](test_probabilities_metadata.json). An
additional local scratch file, `test_proba_rf.npy.tmp.npy`, is ignored as well.
The tracked CSV and Markdown files let the reported metrics be checked without
the arrays.

## Data handling

- No raw or processed IEEE-CIS rows are stored here.
- [`final_prediction_examples.csv`](final_prediction_examples.csv) uses anonymized
  labels (`example_01` … `example_10`) in place of dataset transaction
  identifiers.
- SHAP local examples use synthetic row labels and model-derived values.
- Aggregate metrics and plots must not be used to reconstruct the dataset.

See [`DATA_LICENSE.md`](../DATA_LICENSE.md) for the data and model rights notice.
