# PR-AUC Definition Audit

## Canonical definition

The project-wide PR-AUC is **Average Precision** from
`sklearn.metrics.average_precision_score`, exposed through
`src.fraud.metrics.pr_auc`.

The historical Phase 5/6 notebooks used the trapezoidal implementation
`auc(recall, precision)`. That implementation is retained only as
`src.fraud.metrics.pr_auc_trapezoidal` for audit purposes and must not be used
for model selection or delivery claims.

## Recomputed comparison

| Dataset | Model | PR-AUC (Average Precision) | PR-AUC (trapezoidal) | Difference |
|---|---|---:|---:|---:|
| test | Baseline SGD | 0.286753 | 0.399750 | +0.112997 |
| test | Random Forest (untuned) | 0.722463 | 0.731103 | +0.008640 |
| test | Random Forest (tuned) | 0.730519 | 0.732144 | +0.001625 |
| test | Gradient Boosting | 0.343137 | 0.369340 | +0.026203 |
| test | XGBoost (untuned) | n/a | n/a | n/a |
| train_oof | Random Forest (tuned) | 0.704879 | 0.706801 | +0.001923 |

> Predictions are generated with a single worker (`n_jobs=1`) so that
> floating-point summation order and Average Precision tie ordering are
> deterministic. XGBoost is skipped when the optional `xgboost` package is not
> installed.

## Random Forest comparison

| Comparison | Average Precision | Trapezoidal |
|---|---:|---:|
| Untuned Random Forest | 0.722463 | 0.731103 |
| Tuned Random Forest | 0.730519 | 0.732144 |
| Tuned minus untuned | +0.008056 | +0.001041 |

> The saved Phase 8 summary records `0.7305181893` for the tuned model; the recomputation above gives `0.7305186802`. The difference of `4.91e-07` is attributable to model serialization/runtime version and does not change the conclusion.

The Phase 6 Random Forest value `0.731103` is the trapezoidal result
(`0.731103` after recomputation). Its Average Precision is
`0.722463`, which matches the Phase 8 untuned value `0.722463` to
rounding.

The tuned model's Average Precision is `0.730519`. The canonical
improvement over the untuned Random Forest is therefore
**+0.008056**. Under the historical trapezoidal definition,
the improvement is **+0.001041**. Both comparisons are
internally consistent; mixing the two definitions is not.

## Decision

- Use `src.fraud.metrics.pr_auc` (Average Precision) everywhere.
- Label archived Phase 5/6 values explicitly as trapezoidal if they are quoted.
- Do not compare a Phase 6 trapezoidal value directly with a Phase 8/9
  Average Precision value.
- Notebooks 05 and 06 now call the canonical function; their stored outputs
  predate this change and should be refreshed when the notebooks are rerun.

## Source

Machine-readable table: `results/metrics_definition_audit.csv`.
