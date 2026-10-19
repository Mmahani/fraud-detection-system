# Phase 10 — SHAP Explainability Report

## Model and configuration

The inspected model is a `RandomForestClassifier` with 20 estimators and 495 transformed input features. It is not a scikit-learn Pipeline; preprocessing is stored separately in `models/preprocessor.pkl`. The frozen classification threshold is **0.35**. Risk boundaries remain Low `<0.30`, Medium `0.30 <= p < 0.70`, and High `>=0.70`.

## Feature alignment

The saved Phase 4 preprocessor produced exactly **495** transformed feature names, matching the 495 columns used by the model. SHAP used the processed sparse data converted to dense form only for a reproducible sample of **500** training rows, selected with random state 42. No model retraining, tuning, resampling, or threshold changes were performed.

## SHAP output verification

The TreeExplainer output was normalized to the class-1 Fraud contribution with shape `(500, 495)`. The class-1 additivity mean absolute error was **0.00001165**. Positive SHAP values were interpreted as pushing the class-1 output toward Fraud only after this class-1 alignment check.

## Global interpretation

The complete ranked table is in `results/shap_global_importance.csv`; the first 20 rows are in `results/shap_global_importance_top20.csv`. These features contributed most to the model's decisions in the sampled transactions. They are associations with model output, not claims that the features cause fraud.

## Fraud-specific and local interpretation

Fraud-specific directional contributions are saved in `results/shap_fraud_direction.csv`. Local explanations for real test examples, including available false-positive and false-negative examples at the frozen threshold, are saved in `results/shap_local_explanations.csv`.

## Decision flow

A transaction is transformed with the saved Phase 4 preprocessing, passed to the frozen Phase 8 Random Forest, converted to a Fraud probability, mapped to a risk level, and compared with the frozen Phase 9 classification threshold. SHAP explains which transformed features contributed toward or away from the Fraud output for a particular decision.

**PHASE 10 COMPLETE — SHAP EXPLANATIONS GENERATED.**
