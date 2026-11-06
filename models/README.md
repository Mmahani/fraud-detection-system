# Models

Frozen inference artifacts for the fraud-detection research service.

## Tracked artifacts

| File | Size | Purpose |
|---|---:|---|
| [`best_tuned_model.pkl`](best_tuned_model.pkl) | ~31.5 MB | Phase 8 tuned `RandomForestClassifier` |
| [`preprocessor.pkl`](preprocessor.pkl) | ~0.09 MB | Fitted Phase 4 preprocessing pipeline |
| [`threshold_config.json`](threshold_config.json) | <1 KB | Frozen threshold and risk bands |
| [`model_metadata.json`](model_metadata.json) | <1 KB | Model type, feature count, and threshold |
| [`explainability_config.json`](explainability_config.json) | <1 KB | SHAP configuration |

These files are serialized with **scikit-learn 1.9.0**. Install the pinned
version from [`requirements.txt`](../requirements.txt) to avoid
`InconsistentVersionWarning`.

## Loading the preprocessor

Load it through the project helper:

```python
from src.prediction import load_preprocessor

preprocessor = load_preprocessor()
```

Direct `joblib.load("models/preprocessor.pkl")` can fail because the artifact
references compatibility symbols defined in `src.prediction`.

## Not tracked

The comparison models are ignored by Git because of size:

- `baseline_sgd.pkl`
- `gradient_boosting.pkl`
- `improved_model.pkl` (earlier experiment, not used)
- `random_forest.pkl` (~157 MB)
- `xgboost.pkl`

`xgboost.pkl` also requires the optional `xgboost` package.

## Verification

The artifact manifest records sizes and SHA-256 hashes:

```powershell
python -m src.fraud.artifacts validate
```

## Rights

The model and preprocessor were fitted on IEEE-CIS competition data and are
provided for **research and education only**. The MIT License covers the source
code and documentation, not the dataset or the trained model. See
[`DATA_LICENSE.md`](../DATA_LICENSE.md).
