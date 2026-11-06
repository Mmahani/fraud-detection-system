# Release Notes

## v1.0.1 — Public research release (prepared)

| Item | Value |
|---|---|
| Release | `v1.0.1` — prepared public research release |
| Base | `v1.0.0` snapshot `3a7c4cb` plus release-preparation changes (uncommitted) |
| Date | 2026-11-05 |
| Status | **Public research release candidate** — reviewed, tested, and documented. Not approved for autonomous production use. |
| Code license | **MIT** — Copyright (c) 2026 Mohammed M. A. Mahani |
| Data | IEEE-CIS data is **not included**; see [DATA_LICENSE.md](DATA_LICENSE.md) |
| Model | Phase 8 tuned `RandomForestClassifier`, 20 trees, threshold `0.35` |
| Primary metric | Average Precision (PR-AUC) |

The `v1.0.0` tag remains untouched. Because the repository is still private and
the publication status of `v1.0.0` could not be verified, this release is
prepared as `v1.0.1` rather than moving an existing tag. No tag has been created
or pushed yet.

### What changed for the public research release

- Added the MIT code license and copyright notice.
- Added the data/model rights notice for the IEEE-CIS dataset.
- Added a security policy and private vulnerability-reporting guidance.
- Removed local paths from the README and cleared machine-specific notebook
  outputs.
- Replaced `examples/sample_transaction.json` with a fully synthetic 433-field
  payload.
- Updated the JSON-null regression test to build its payload from the processed
  data instead of assuming the example file is the first data row.
- Added `.gitattributes` and normalized all text files to LF so artifact
  hashes validate on any platform.
- Removed the last absolute local path from `results/phase8_fast_report.md`
  and refreshed the affected manifest hashes.
- Updated the FastAPI application version to `1.0.1`.

### Data and model rights

The IEEE-CIS dataset is not distributed with this repository. The competition
rules permit non-commercial use only and prohibit redistribution to any party not
participating in the competition. The trained model artifacts are provided for
research and education only. The MIT License covers the source code and
documentation, not the dataset. See [DATA_LICENSE.md](DATA_LICENSE.md).

### Security

The FastAPI service is a local research service with no authentication, rate
limiting, or TLS. Do not expose it directly to the internet. See
[SECURITY.md](SECURITY.md).

### v1.0.1 release checklist

- [x] MIT `LICENSE` and copyright notice
- [x] `DATA_LICENSE.md` data/model rights notice
- [x] `SECURITY.md` vulnerability-reporting policy
- [x] README status, license, data, and security sections updated
- [x] Model card license section updated
- [x] Local paths and machine-specific notebook outputs removed
- [x] Synthetic example payload
- [x] LF line endings (`.gitattributes`) and cross-platform manifest validation
- [ ] Commit, tag, and publish (owner approval required)
- [ ] Verify remote CI after push
- [ ] Change repository visibility to public (owner action)

### Clean-clone verification

Verified on 2026-11-06 from a fresh clone of `v1.0.0` (`3a7c4cb`) with the
release-preparation changes applied:

| Check | Result |
|---|---|
| Fresh clone + `pip install -r requirements.txt` | Success (Python 3.12.6) |
| Unit tests | 21 tests: 20 passed, 1 skipped (needs `data/processed`) |
| `ruff check src scripts tests` | All checks passed |
| `pip check` | No broken requirements |
| Artifact manifest | 40/40 present, 0 errors; 6 optional local data files absent as expected |
| API | `/`, `/health`, `/model/info`, `/model/features`, `/predict`, `/predict/batch`, `/explain`, `/docs` all OK |
| Notebooks | 12/12 parse; 0 stored outputs |
| Local paths, secrets, attribution | None found |

## v1.0.0 — Delivery snapshot

| Item | Value |
|---|---|
| Release | `v1.0.0` delivery snapshot |
| Date | 2026-10-20 |
| Base code revision | `f8afd5d` (plus this documentation commit) |
| Status | **Delivery candidate** — reviewed, tested, and documented. Not approved for autonomous production use. |
| Model | Phase 8 tuned `RandomForestClassifier`, 20 trees, threshold `0.35` |
| Primary metric | Average Precision (PR-AUC) |

### Scope

This snapshot covers the full project path:

- Data inspection, integration, EDA, leakage-safe preprocessing, and split.
- Baseline, model comparison, validation audit, imbalance handling, and tuning.
- Threshold optimization, SHAP explainability, and final model packaging.
- Phase 12 time-based validation diagnostic.
- Phase 13 FastAPI inference service.
- Phase 14 GitHub Actions CI.
- Phase 15 model card and delivery documentation.

### Verification snapshot

Run on 2026-10-20 from the project root:

| Check | Result |
|---|---|
| Unit tests | **21/21 passed** |
| Lint (`ruff check src scripts tests`) | **All checks passed** |
| Artifact manifest | **40/40 present**, 0 errors, 0 warnings |
| Notebook parse | **12/12**, 0 stored errors |
| `pip check` | **No broken requirements** |
| API smoke test | `/health` returned `status=ok`; `/predict` returned a valid response |
| Final consistency audit | **0 failures, 0 warnings** |

The GitHub Actions workflow was validated locally (YAML parse and the same lint/test
commands). Its remote run status could not be queried from the local environment
because the repository is private and the `gh` CLI is unavailable.

### Headline metrics

#### Final model at the frozen 0.35 threshold

| Metric | Value |
|---|---:|
| PR-AUC (Average Precision) | **0.730518** |
| ROC-AUC | 0.937292 |
| Precision | 0.582119 |
| Recall | 0.724655 |
| F1 | 0.645613 |
| Accuracy | 0.972161 |
| TN / FP / FN / TP | 111,825 / 2,150 / 1,138 / 2,995 |

#### Five-fold training-only validation

| Metric | Mean | Std (ddof=1) |
|---|---:|---:|
| PR-AUC | 0.704887 | 0.008989 |
| F1 | 0.666683 | 0.007884 |
| Recall | 0.608227 | 0.010642 |
| Precision | 0.737667 | 0.007506 |
| ROC-AUC | 0.931818 | 0.003393 |

#### Phase 6 comparison (Average Precision)

| Model | PR-AUC |
|---|---:|
| Random Forest (untuned) | 0.722463 |
| XGBoost | 0.573457 |
| Gradient Boosting | 0.343137 |
| SGD baseline | 0.286753 |

#### Phase 12 time buckets (tuned model)

| Bucket | PR-AUC | Fraud rate |
|---|---:|---:|
| 1 (earliest) | 0.654729 | 2.41% |
| 2 | 0.764482 | 3.99% |
| 3 | 0.741715 | 3.73% |
| 4 | 0.737265 | 3.97% |
| 5 (latest) | 0.723480 | 3.40% |

Later buckets are stable. The earliest bucket is weaker but also has the lowest
fraud rate, and PR-AUC is prevalence-sensitive. This is a stability check on the
random-split holdout, not a strict forward-in-time validation.

### Deliverables

#### Models and configuration

- `models/best_tuned_model.pkl`
- `models/preprocessor.pkl`
- `models/threshold_config.json`
- `models/model_metadata.json`
- `models/explainability_config.json`

#### Evaluation and explainability

- `results/final_threshold_test_results.csv`
- `results/phase8_fast_summary.json`
- `results/phase8_fast_report.md`
- `results/phase8_evaluation_audit.md`
- `results/metrics_definition_audit.csv` / `.md`
- `results/phase12_time_validation.csv` / `.md` / `results/phase12_summary.json`
- `results/model_comparison.csv`
- `results/phase10_final_report.md`
- `artifacts/manifest.json`

#### Service and examples

- `src/prediction.py`
- `src/api/main.py`, `src/api/schemas.py`
- `examples/sample_transaction.json`

#### Documentation, tests, and CI

- `README.md`
- `MODEL_CARD.md`
- `LICENSE` (MIT)
- `DATA_LICENSE.md` (IEEE-CIS data/model rights notice)
- `SECURITY.md` (vulnerability-reporting policy)
- `tests/` (21 tests)
- `.github/workflows/ci.yml`
- `requirements.txt`, `requirements-ci.txt`

### How to verify the handoff

```powershell
python -m pip install -r requirements.txt
python -m src.fraud.artifacts validate
python -m ruff check src scripts tests
python -m unittest discover -s tests -v
```

Start the API:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

The raw and processed data, the out-of-fold probabilities, and the saved test
probability arrays are not committed. Tests that require processed data skip
automatically. See the README for data regeneration.

### Handoff checklist

- [x] Frozen model, preprocessor, threshold, and metadata
- [x] Reproducible metrics and canonical PR-AUC audit
- [x] All notebooks parse without stored errors
- [x] Artifact manifest validated (40/40)
- [x] Unit tests and lint passing (21/21)
- [x] FastAPI service with health, info, single, batch, and explain endpoints
- [x] Example payload
- [x] Model card with intended use, metrics, and limitations
- [x] CI workflow
- [ ] Strict forward-in-time retrain (blocked by the current 8 GB memory budget)
- [ ] Probability calibration and threshold review against verified business costs
- [ ] Fairness and proxy-variable audit
- [ ] Drift and performance monitoring
- [ ] API authentication, rate limiting, and TLS
- [ ] Deployment pipeline / container image
- [ ] Explicit code license before public distribution (completed in `v1.0.1`; see above)

### Known limitations

- The split is random and stratified, not time-based; `TransactionDT` is a strong
  feature, so temporal leakage remains possible.
- The test set has been evaluated across multiple phases and is not a pristine
  final holdout.
- The model overfits relative to the test set at the default 0.5 threshold
  (train F1 `0.9569` vs test F1 `0.6913`).
- Probabilities are uncalibrated; risk bands are presentation heuristics.
- The API has no authentication, rate limiting, or TLS and should run behind a
  gateway.
- Raw/processed data and `.npy` probability arrays are intentionally excluded
  from Git.
- The code is MIT-licensed, but the IEEE-CIS dataset remains subject to the
  Kaggle competition rules, including non-commercial use and no redistribution.
  The trained model artifacts are provided for research and education only.

### Recommended next steps

1. Run a strict forward-in-time retrain and compare it with the random-split
   metrics.
2. Calibrate probabilities and re-select the threshold using verified costs.
3. Audit performance across relevant segments and proxy variables.
4. Add drift monitoring and human review for high-risk decisions.
5. Add deployment packaging and API security before any production rollout.
