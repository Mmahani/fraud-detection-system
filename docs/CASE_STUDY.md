# Case Study — Explainable Fraud Detection on IEEE-CIS

A research case study for a portfolio or technical interview. Every metric below
is taken from a tracked artifact in this repository; none is estimated or
invented.

## At a glance

| Item | Detail |
|---|---|
| Role | Project author and machine learning engineer |
| Scope | End-to-end research build: data, modeling, evaluation, explainability, API, tests, documentation |
| Dataset | IEEE-CIS Fraud Detection (Kaggle / Vesta Corporation), 590,540 rows |
| Imbalance | 3.4989% fraud |
| Final model | Tuned `RandomForestClassifier`, 20 trees, threshold `0.35` |
| Primary metric | Average Precision (PR-AUC) |
| Frozen test PR-AUC | **0.730518** |
| Training-only 5-fold PR-AUC | **0.704887 ± 0.008989** |
| Stack | Python, pandas, scikit-learn, XGBoost, SHAP, FastAPI, unittest, ruff, GitHub Actions |
| Status | Research prototype — **not** approved for production |

## The problem

Fraud detection is a rare-event ranking problem. In this dataset only 3.5% of
transactions are fraudulent, so a model can look highly accurate while still
missing most fraud. The project therefore optimizes and reports **Average
Precision (PR-AUC)** rather than accuracy, and it makes the operating threshold
an explicit, documented decision.

The practical questions were:

1. Which model ranks fraud well under severe class imbalance?
2. How can preprocessing avoid leaking test information?
3. Where should the decision threshold sit if false negatives are assumed to cost
   more than false positives?
4. Can individual decisions be explained to a reviewer?
5. Can the whole artifact set be verified and reproduced?

## My role

I was the author and owner of the repository. I built and documented the
end-to-end research pipeline:

- integrating the transaction and identity tables;
- designing leakage-safe preprocessing and the train/test split;
- running the baseline and model comparison;
- tuning the final Random Forest and selecting the threshold from out-of-fold
  training probabilities;
- producing and validating SHAP explanations;
- packaging the frozen artifacts and building the FastAPI service;
- writing the tests, artifact manifest, CI configuration, and documentation.

The work was a research build, not a production deployment. No business impact,
revenue saving, or live-traffic result is claimed.

## Constraints

| Constraint | Effect on the design |
|---|---|
| 8 GB memory environment | Limited model size and prevented a full Phase 6 retrain |
| IEEE-CIS rules | Raw and processed data cannot be redistributed |
| No production data | All evaluation is offline and retrospective |
| Rare fraud class | PR-AUC, balanced class weights, and recall-aware thresholding |
| Windows development | Explicit LF normalization for cross-platform artifact hashes |
| Reproducibility requirement | Pinned scikit-learn, fixed seeds, hash manifest, tests |

## What I built

1. **Data integration** — merged 590,540 transactions with identity fields into
   434 columns (432 model inputs plus the identifier and label).
2. **Leakage-safe preprocessing** — numeric, low-cardinality, and
   high-cardinality branches fitted on training rows only; 495 transformed
   features.
3. **Model comparison** — SGD baseline, Random Forest, Gradient Boosting, and
   XGBoost under one evaluation function.
4. **Tuning and validation** — 20-tree balanced Random Forest, validated with
   5-fold `StratifiedKFold` on training data only.
5. **Threshold selection** — out-of-fold probabilities, illustrative
   `COST_FP=1` / `COST_FN=5`, frozen at **0.35** before test scoring.
6. **Explainability** — `shap.TreeExplainer` on 500 training transactions,
   aligned to the class-1 output, with an additivity check.
7. **Serving** — a local FastAPI service for `/predict`, `/predict/batch`, and
   `/explain`.
8. **Verification** — 21 unit tests (one data-dependent skip on a clean clone),
   `ruff`, GitHub Actions, and a 40-entry SHA-256 artifact manifest.

## Key technical decisions

| Decision | Alternative | Why I chose it | Trade-off |
|---|---|---|---|
| Average Precision as primary metric | Accuracy or ROC-AUC | Directly reflects ranking quality under 3.5% prevalence | Less familiar to non-specialists |
| `class_weight="balanced"` | SMOTE or random oversampling | Simpler, no synthetic rows, keeps the pipeline reproducible | Does not create new minority examples |
| 20-tree Random Forest | Larger forest or boosted model | Fit the 8 GB environment and kept inference practical | Lower capacity than a larger ensemble |
| OOF threshold selection | Tune on the test set | Prevents test leakage into the operating point | OOF threshold may not transfer perfectly |
| Threshold 0.35 | Default 0.50 | Illustrative cost ratio favours recall | More false positives (2,150 vs 842) |
| SHAP `TreeExplainer` | Model-agnostic Kernel SHAP | Exact and fast for tree ensembles | Specific to tree models |
| Local FastAPI service | Notebook-only inference | Makes the model usable and testable through a stable contract | Adds an operational surface |
| Hash manifest | No artifact inventory | Detects accidental artifact drift | Requires hash refresh after intentional changes |
| Research/education notice | Broad model license | Respects the dataset terms | Limits commercial reuse |

## Challenges and how they were handled

### 1. Severe class imbalance

Accuracy is misleading at a 3.5% fraud rate. I used PR-AUC as the primary metric,
balanced class weights, and a recall-aware threshold. The result is a deliberate
precision/recall trade-off rather than a single "best" model.

### 2. Leakage and time

The split is random and stratified, and `TransactionDT` is among the strongest
SHAP features. Preprocessing was fitted on training rows only, and Phase 12 adds
a five-bucket temporal stability diagnostic. That diagnostic is explicitly
labelled as **not** a strict forward-in-time holdout.

### 3. Memory limits

A full Phase 6 retrain exceeded the available 8 GB. I kept the saved test
probability arrays and refreshed the metrics from them, while documenting that
training times and feature importances remain from the historical run.

### 4. Metric ambiguity

Some early outputs used the trapezoidal PR-AUC variant while others used Average
Precision. I added a metric-definition audit, made `average_precision_score` the
canonical implementation, and documented the historical difference.

### 5. Explainability correctness

SHAP values for a binary tree model can be returned in different shapes. I
normalized them to the class-1 output and verified additivity with a mean
absolute error of `0.00001165` before interpreting the signs.

### 6. Cross-platform reproducibility

Artifact hashes initially differed between a Windows working tree (CRLF) and a
clean clone (LF). I added `.gitattributes`, normalized the text files to LF, and
recomputed the manifest hashes. A fresh clone now validates on any platform.

### 7. Honest communication

The test split has been reused across phases, the model is not calibrated, and
the train/test F1 gap is large. These caveats are stated in the README, model
card, and results documentation instead of being hidden.

## Results

### Frozen operating point (threshold 0.35)

| Metric | Value |
|---|---:|
| PR-AUC (Average Precision) | **0.730518** |
| ROC-AUC | 0.937292 |
| Precision | 0.582119 |
| Recall | 0.724655 |
| F1 | 0.645613 |
| TN / FP / FN / TP | 111,825 / 2,150 / 1,138 / 2,995 |

### Training-only validation

Five-fold cross-validation on the training partition: PR-AUC **0.704887 ±
0.008989**, F1 **0.666683 ± 0.007884**, ROC-AUC **0.931818 ± 0.003393**.

### Model comparison (test PR-AUC)

| Model | PR-AUC |
|---|---:|
| Random Forest (untuned) | 0.722463 |
| XGBoost | 0.573457 |
| Gradient Boosting | 0.343137 |
| SGD baseline | 0.286753 |

### Tuning effect at threshold 0.50

| Metric | Untuned | Tuned |
|---|---:|---:|
| PR-AUC | 0.722463 | 0.730518 |
| Recall | 0.537382 | 0.634890 |
| Precision | 0.868256 | 0.758601 |
| F1 | 0.663877 | 0.691254 |
| FP | 337 | 835 |
| FN | 1,912 | 1,509 |

Source: [`results/phase8_tuned_vs_untuned_test.csv`](../results/phase8_tuned_vs_untuned_test.csv).
The tuned model improves ranking and recall, at the cost of precision and more
false positives.

### Temporal stability

Time buckets 2–5 of the reserved test split sit between **0.723480** and
**0.764482** PR-AUC. The earliest bucket is **0.654729**, but it also has the
lowest fraud rate (2.41%), and PR-AUC is prevalence-sensitive.

### Explainability

Top global SHAP features: `V258`, `C1`, `C13`, `C14`, `TransactionAmt`, and
`TransactionDT`. These are model associations, not causal effects.

## What I learned

- **Evaluation design drives the conclusion.** Metric choice, split design, and
  threshold policy mattered more than adding model capacity.
- **Thresholds are decisions, not hyperparameters.** The 0.35 choice encodes an
  explicit cost assumption that should be reviewed with stakeholders.
- **Explainability needs its own verification.** Class alignment and additivity
  checks are part of the deliverable, not an afterthought.
- **Reproducibility is a feature.** Hash manifests, pinned versions, and clean
  clones prevent quiet drift.
- **Restraint is part of the engineering.** Documenting what the model cannot do
  is as important as reporting what it can.

## What I would do next

1. Build a strict forward-in-time training and evaluation split.
2. Calibrate probabilities and report calibration curves.
3. Replace the illustrative cost ratio with a stakeholder-reviewed cost model.
4. Add drift monitoring and a human-review workflow.
5. Containerize the service and add authentication, rate limiting, and TLS
   before any networked deployment.
6. Run a fairness and subgroup analysis where legally and ethically appropriate.
7. Reduce model size or explore a distilled model for lower-latency inference.

## Links

- [Repository README](../README.md)
- [Architecture](ARCHITECTURE.md)
- [Verified results](RESULTS.md)
- [API demo](DEMO.md)
- [Model card](../MODEL_CARD.md)
- [Data and model rights](../DATA_LICENSE.md)

## Honest limitations

- The test split is retrospective and reused across phases.
- The train/test F1 gap is about 0.2657; overfitting remains a concern.
- Probabilities are not calibrated.
- The threshold cost ratio is illustrative.
- The split is not strictly time-based.
- The service is local-only and not production-ready.
