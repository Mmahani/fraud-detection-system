# Phase 12 — Time-Based Validation Diagnostic

## What this measures

The original preprocessing split is random and stratified, and `TransactionDT` is one of the
strongest features. This diagnostic sorts the reserved test set by `TransactionDT` and scores
the frozen models on five equal-count time buckets, from earliest to latest.

This is a **temporal-stability check on unseen rows**, not a strict forward-in-time holdout:
because the split is random, the test period overlaps the training period, so the models have
already seen transactions from every bucket. A full forward-in-time retrain would require
holding the ~1.2 GB training matrix and is deferred under the current 8 GB memory budget.
No frozen model was retrained or modified.

## PR-AUC by time bucket

| Bucket | Gradient Boosting (Phase 6) | Random Forest (Phase 6) | SGD (Phase 5) | Tuned Random Forest (final) | XGBoost (Phase 6) |
|---|---|---|---|---|---|
| 1 | 0.259426 | 0.644481 | 0.241088 | 0.654729 | 0.490881 |
| 2 | 0.401364 | 0.756164 | 0.353199 | 0.764482 | 0.632733 |
| 3 | 0.347916 | 0.721503 | 0.285028 | 0.741715 | 0.575761 |
| 4 | 0.357011 | 0.733202 | 0.328289 | 0.737265 | 0.588657 |
| 5 | 0.345932 | 0.725507 | 0.227007 | 0.723480 | 0.544135 |

## First vs last bucket

| Model | Earliest PR-AUC | Latest PR-AUC | Latest − Earliest |
|---|---:|---:|---:|
| SGD (Phase 5) | 0.241088 | 0.227007 | -0.014080 |
| Random Forest (Phase 6) | 0.644481 | 0.725507 | +0.081026 |
| Gradient Boosting (Phase 6) | 0.259426 | 0.345932 | +0.086506 |
| XGBoost (Phase 6) | 0.490881 | 0.544135 | +0.053255 |
| Tuned Random Forest (final) | 0.654729 | 0.723480 | +0.068751 |

## Interpretation

- The later buckets (3–5) are stable for the tree models: the tuned Random Forest
  stays in the `0.723–0.742` PR-AUC range.
- The earliest bucket is weaker for every model, but it also has the lowest fraud
  rate (`2.41%` versus `3.40–3.99%` in the other buckets). PR-AUC is
  prevalence-sensitive, so part of that gap is compositional rather than temporal
  degradation.
- This diagnostic cannot establish forward-in-time generalization: because the split
  is random, every time bucket overlaps the training period. The main leakage risk
  remains unquantified until the training matrix can be rebuilt with a strict time
  cutoff.

## Full bucket table

| Bucket | Model | Rows | Fraud Count | Fraud Rate (%) | Scaled Time Min | Scaled Time Max | PR-AUC | ROC-AUC | F1 | Recall | Precision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | SGD (Phase 5) | 23622 | 569 | 2.408771 | 0.018737 | 0.499687 | 0.241088 | 0.853277 | 0.282909 | 0.557118 | 0.189593 |
| 1 | Random Forest (Phase 6) | 23622 | 569 | 2.408771 | 0.018737 | 0.499687 | 0.644481 | 0.924819 | 0.629803 | 0.590510 | 0.674699 |
| 1 | Gradient Boosting (Phase 6) | 23622 | 569 | 2.408771 | 0.018737 | 0.499687 | 0.259426 | 0.789816 | 0.244118 | 0.145870 | 0.747748 |
| 1 | XGBoost (Phase 6) | 23622 | 569 | 2.408771 | 0.018737 | 0.499687 | 0.490881 | 0.910549 | 0.177462 | 0.833040 | 0.099309 |
| 1 | Tuned Random Forest (final) | 23622 | 569 | 2.408771 | 0.018737 | 0.499687 | 0.654729 | 0.920645 | 0.599209 | 0.666081 | 0.544540 |
| 2 | SGD (Phase 5) | 23622 | 943 | 3.992041 | 0.499708 | 1.210747 | 0.353199 | 0.881432 | 0.383650 | 0.632025 | 0.275416 |
| 2 | Random Forest (Phase 6) | 23622 | 943 | 3.992041 | 0.499708 | 1.210747 | 0.756164 | 0.944401 | 0.712542 | 0.680806 | 0.747381 |
| 2 | Gradient Boosting (Phase 6) | 23622 | 943 | 3.992041 | 0.499708 | 1.210747 | 0.401364 | 0.811241 | 0.301853 | 0.181336 | 0.900000 |
| 2 | XGBoost (Phase 6) | 23622 | 943 | 3.992041 | 0.499708 | 1.210747 | 0.632733 | 0.915656 | 0.223905 | 0.878049 | 0.128312 |
| 2 | Tuned Random Forest (final) | 23622 | 943 | 3.992041 | 0.499708 | 1.210747 | 0.764482 | 0.943912 | 0.674929 | 0.750795 | 0.612987 |
| 3 | SGD (Phase 5) | 23622 | 881 | 3.729574 | 1.210752 | 1.889361 | 0.285028 | 0.848134 | 0.336156 | 0.585698 | 0.235724 |
| 3 | Random Forest (Phase 6) | 23622 | 881 | 3.729574 | 1.210752 | 1.889361 | 0.721503 | 0.937476 | 0.684674 | 0.661748 | 0.709246 |
| 3 | Gradient Boosting (Phase 6) | 23622 | 881 | 3.729574 | 1.210752 | 1.889361 | 0.347916 | 0.781279 | 0.277778 | 0.164586 | 0.889571 |
| 3 | XGBoost (Phase 6) | 23622 | 881 | 3.729574 | 1.210752 | 1.889361 | 0.575761 | 0.902691 | 0.205198 | 0.855846 | 0.116574 |
| 3 | Tuned Random Forest (final) | 23622 | 881 | 3.729574 | 1.210752 | 1.889361 | 0.741715 | 0.939811 | 0.640316 | 0.735528 | 0.566929 |
| 4 | SGD (Phase 5) | 23621 | 937 | 3.966809 | 1.889362 | 2.649768 | 0.328289 | 0.865062 | 0.344395 | 0.614728 | 0.239203 |
| 4 | Random Forest (Phase 6) | 23621 | 937 | 3.966809 | 1.889362 | 2.649768 | 0.733202 | 0.944968 | 0.690889 | 0.659552 | 0.725352 |
| 4 | Gradient Boosting (Phase 6) | 23621 | 937 | 3.966809 | 1.889362 | 2.649768 | 0.357011 | 0.780391 | 0.270852 | 0.161153 | 0.848315 |
| 4 | XGBoost (Phase 6) | 23621 | 937 | 3.966809 | 1.889362 | 2.649768 | 0.588657 | 0.907849 | 0.223915 | 0.878335 | 0.128313 |
| 4 | Tuned Random Forest (final) | 23621 | 937 | 3.966809 | 1.889362 | 2.649768 | 0.737265 | 0.937223 | 0.648700 | 0.732124 | 0.582343 |
| 5 | SGD (Phase 5) | 23621 | 803 | 3.399517 | 2.649828 | 3.424895 | 0.227007 | 0.848740 | 0.276844 | 0.605230 | 0.179468 |
| 5 | Random Forest (Phase 6) | 23621 | 803 | 3.399517 | 2.649828 | 3.424895 | 0.725507 | 0.944285 | 0.694517 | 0.662516 | 0.729767 |
| 5 | Gradient Boosting (Phase 6) | 23621 | 803 | 3.399517 | 2.649828 | 3.424895 | 0.345932 | 0.786263 | 0.303772 | 0.185554 | 0.837079 |
| 5 | XGBoost (Phase 6) | 23621 | 803 | 3.399517 | 2.649828 | 3.424895 | 0.544135 | 0.907516 | 0.199142 | 0.866750 | 0.112494 |
| 5 | Tuned Random Forest (final) | 23621 | 803 | 3.399517 | 2.649828 | 3.424895 | 0.723480 | 0.937696 | 0.646396 | 0.714819 | 0.589928 |

## Artifacts

- `results/phase12_time_validation.csv`
- `results/phase12_summary.json`
- `results/phase12_plots/time_bucket_pr_auc.png`
