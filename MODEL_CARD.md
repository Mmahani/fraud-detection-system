# Model Card — IEEE-CIS Fraud Detection

## Model details

| Item | Value |
|---|---|
| Model name | IEEE-CIS Fraud Detection — Phase 8 tuned Random Forest |
| Model card version | 1.0 (Phase 15 delivery snapshot) |
| Algorithm | `RandomForestClassifier` |
| Estimators | 20 |
| `max_depth` | `None` |
| `min_samples_leaf` | 2 |
| `min_samples_split` | 5 |
| `max_features` | 0.5 |
| `class_weight` | `balanced` |
| `random_state` | 42 |
| Raw input features | 434 |
| Transformed features | 495 |
| Classification threshold | **0.35** |
| Training environment | Python 3.12, `scikit-learn==1.9.0` |
| Explainability | SHAP `TreeExplainer` (`shap==0.54.1`) |
| Inference entry points | `src/prediction.py`, FastAPI service in `src/api/main.py` |

The model is a standalone estimator. Preprocessing is stored separately in
`models/preprocessor.pkl` and must be loaded through `src.prediction` because the
artifact references compatibility symbols. The preprocessing pipeline applies
median imputation and `StandardScaler(with_mean=False)` to numeric columns,
constant `"Missing"` imputation plus one-hot encoding to low-cardinality
categorical columns, and frequency encoding to high/medium-cardinality
categorical columns. Two flags, `DeviceInfo_is_missing` and
`id_33_is_missing`, are derived from their raw fields at inference time.

## Intended use

The model is intended for **research, education, and offline decision support**:
ranking transactions by fraud probability, comparing models, and demonstrating an
end-to-end fraud workflow. It can be used through the batch helpers or the local
REST API.

**Primary users:** data scientists, ML engineers, and reviewers working with the
IEEE-CIS dataset or a closely similar transaction schema.

**Out-of-scope uses:**

- Fully autonomous blocking or approval of transactions without human review.
- Production decisions without calibration, drift monitoring, and a cost review.
- Other datasets, geographies, payment channels, or feature schemas.
- Adversarial fraud patterns deliberately optimized against this model.
- Fairness-sensitive decisions affecting individuals without a dedicated audit.
- Any use that treats the risk bands as calibrated probabilities.

## Factors

The model consumes transaction, card, address, email-domain, device, identity,
counting, and time-delta features from the IEEE-CIS data. The strongest global
SHAP associations in a 500-row training sample were `V258`, `C1`, `C13`, `C14`,
`TransactionAmt`, and `TransactionDT`.

Relevant variability factors include:

- **Time:** `TransactionDT` is among the top features. The original split is
  random, so temporal leakage is possible.
- **Fraud prevalence:** PR-AUC is prevalence-sensitive. The earliest Phase 12
  time bucket has a lower fraud rate and a lower PR-AUC than the middle buckets.
- **Class imbalance:** fraud is about 3.5% of transactions (27.6 legitimate per
  fraud), which is why average precision and recall are more informative than
  accuracy.
- **Threshold:** the frozen threshold is 0.35; 0.50 produces higher precision and
  lower recall.
- **Missingness:** many identity fields are missing. The model relies on explicit
  missing-value handling and the two derived missingness flags.

## Metrics

### Frozen-threshold test metrics (threshold 0.35)

Retrospective evaluation on the 118,108-row reserved test split:

| Metric | Value |
|---|---:|
| Accuracy | 0.972161 |
| Balanced accuracy | 0.852896 |
| Precision | 0.582119 |
| Recall | 0.724655 |
| F1 | 0.645613 |
| ROC-AUC | 0.937292 |
| PR-AUC (Average Precision) | **0.730518** |
| TN / FP / FN / TP | 111,825 / 2,150 / 1,138 / 2,995 |

### Threshold 0.50, for comparison

| Metric | Value |
|---|---:|
| Precision | 0.757558 |
| Recall | 0.636584 |
| F1 | 0.691822 |
| TN / FP / FN / TP | 113,133 / 842 / 1,502 / 2,631 |

### Five-fold training-only validation

The tuned configuration was validated with 5-fold `StratifiedKFold` on the
training set only; the test set was not used.

| Metric | Mean | Std (ddof=1) |
|---|---:|---:|
| PR-AUC | 0.704887 | 0.008989 |
| F1 | 0.666683 | 0.007884 |
| Recall | 0.608227 | 0.010642 |
| Precision | 0.737667 | 0.007506 |
| ROC-AUC | 0.931818 | 0.003393 |

### Phase 6 model comparison (canonical Average Precision)

Each Phase 6 model is scored at its own default prediction threshold; the tuned
final model is reported separately at the frozen 0.35 threshold.

| Model | PR-AUC | F1 | ROC-AUC |
|---|---:|---:|---:|
| Random Forest (untuned) | 0.722463 | 0.663877 | 0.940513 |
| XGBoost | 0.573457 | 0.306512 | 0.909753 |
| Gradient Boosting | 0.343137 | 0.221847 | 0.789104 |
| SGD baseline | 0.286753 | 0.328110 | 0.860654 |

The tuned final model reaches PR-AUC **0.730518** at threshold 0.35; see the
frozen-threshold table above.

### Time-bucket diagnostic

Phase 12 sorts the reserved test set by `TransactionDT` into five equal-count
buckets. The tuned model's PR-AUC by bucket is:

| Bucket | PR-AUC | Fraud rate |
|---|---:|---:|
| 1 (earliest) | 0.654729 | 2.41% |
| 2 | 0.764482 | 3.99% |
| 3 | 0.741715 | 3.73% |
| 4 | 0.737265 | 3.97% |
| 5 (latest) | 0.723480 | 3.40% |

Later buckets are stable. The earliest bucket is weaker, but it also has the
lowest fraud rate; PR-AUC is prevalence-sensitive, so part of that gap is
compositional. This is a stability check on unseen rows, **not** a strict
forward-in-time holdout, because the random split means every time bucket
overlaps the training period.

### Training fit and overfitting diagnostic

| Metric | Training | Test |
|---|---:|---:|
| Accuracy | 0.996850 | 0.972161 |
| Balanced accuracy | 0.998368 | 0.852896 |
| Precision | 0.917416 | 0.582119 |
| Recall | 1.000000 | 0.724655 |
| F1 | 0.956929 | 0.645613 |
| ROC-AUC | 0.999998 | 0.937292 |
| PR-AUC | 0.999923 | 0.730518 |

The train/test F1 gap of about **0.266** is large. It is an overfitting
diagnostic, not an unbiased generalization estimate, because the training
metrics are measured after fitting on all training rows.

## Evaluation data

- **Source:** IEEE-CIS Fraud Detection competition data (Vesta Corporation,
  Kaggle). The data is anonymized.
- **Rows:** 590,540 total; 472,432 training and 118,108 test.
- **Fraud:** 16,530 in training (3.4989%) and 4,133 in test (3.4993%).
- **Split:** random stratified 80/20 with `random_state=42`, performed before
  fitting any preprocessing statistics.
- **Reuse:** the test split has been evaluated in several phases. It is a
  retrospective holdout, not a pristine final holdout.
- **Time coverage:** the split is not time-based. `TransactionDT` is present as a
  feature and is among the top SHAP associations, so temporal leakage is a known
  risk. Phase 12 quantifies stability but cannot replace a strict
  forward-in-time retrain.
- **Excluded data:** raw and processed data are not committed to Git. See the
  README for regeneration steps.

## Ethical considerations

- **False positives** create customer friction and may block legitimate
  transactions. At threshold 0.35, the test set has 2,150 false positives.
- **False negatives** leave fraud losses unaddressed. At threshold 0.35, the test
  set has 1,138 false negatives.
- **Cost assumptions are illustrative.** The threshold used `COST_FP=1` and
  `COST_FN=5`, with F1 as a tie-breaker. These are not verified business costs.
- **No fairness audit has been performed.** The dataset is anonymized and does
  not expose protected attributes, but card type, email domain, address, device,
  and other features may act as proxies. Performance across groups is unknown.
- **Human oversight is required** for high-impact decisions. The model should
  flag or rank cases, not make unreviewable decisions.
- **Explainability is associative, not causal.** SHAP values describe how the
  model used features; they do not prove that a feature causes fraud.

## Caveats and recommendations

1. **Do not treat the reported test metrics as a clean production estimate.**
   The split is random, the test set has been reused, and temporal leakage is
   possible.
2. **Run a strict forward-in-time retrain** when sufficient memory is available:
   fit preprocessing and the model on earlier transactions only, then evaluate on
   a later period.
3. **Calibrate the probabilities** before using them as risk scores. The current
   outputs are uncalibrated; the risk bands are presentation heuristics.
4. **Re-select the threshold** with verified business costs and a review of the
   precision/recall trade-off.
5. **Monitor drift and performance** in production, especially fraud prevalence,
   feature distributions, and false-positive/false-negative rates.
6. **Audit fairness and proxy effects** before deployment in a regulated or
   customer-facing setting.
7. **Secure the API.** The local service has no authentication, rate limiting,
   or TLS; deploy it behind a gateway with logging and abuse controls.
8. **Keep a human in the loop** for blocked or high-risk transactions, with an
   appeal or review path.

## Artifacts and reproducibility

| Artifact | Path |
|---|---|
| Final model | `models/best_tuned_model.pkl` |
| Preprocessor | `models/preprocessor.pkl` |
| Threshold and risk bands | `models/threshold_config.json` |
| Model metadata | `models/model_metadata.json` |
| Frozen-threshold metrics | `results/final_threshold_test_results.csv` |
| Phase 8 CV and test summary | `results/phase8_fast_summary.json` |
| Phase 8 evaluation audit | `results/phase8_evaluation_audit.md` |
| Canonical PR-AUC audit | `results/metrics_definition_audit.md` |
| Time-bucket diagnostic | `results/phase12_time_validation.md` |
| SHAP report | `results/phase10_final_report.md` |
| Artifact manifest | `artifacts/manifest.json` |
| API | `src/api/main.py` |
| Example payload | `examples/sample_transaction.json` |
| Tests | `tests/` |

Reproduce the artifact manifest and run the checks:

```powershell
python -m src.fraud.artifacts build
python -m src.fraud.artifacts validate
python -m ruff check src scripts tests
python -m unittest discover -s tests -v
```

Start the API:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

## References

- IEEE-CIS Fraud Detection competition: <https://www.kaggle.com/c/ieee-fraud-detection>
- scikit-learn: <https://scikit-learn.org>
- SHAP: <https://shap.readthedocs.io>
- FastAPI: <https://fastapi.tiangolo.com>
