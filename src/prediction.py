from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from scipy import sparse as scipy_sparse
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Compatibility class required to deserialize the saved preprocessor."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.maps_ = []
        for column in X.columns:
            self.maps_.append(X[column].value_counts(normalize=True, dropna=False).to_dict())
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        result = pd.DataFrame(index=X.index)
        for i, column in enumerate(X.columns):
            result[column] = X[column].map(self.maps_[i]).fillna(0).astype('float32')
        return result.to_numpy()


def to_sparse(X):
    return scipy_sparse.csr_matrix(X)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / 'models' / 'best_tuned_model.pkl'
PREPROCESSOR_PATH = PROJECT_ROOT / 'models' / 'preprocessor.pkl'
THRESHOLD_PATH = PROJECT_ROOT / 'models' / 'threshold_config.json'


def _register_pickle_symbols() -> None:
    import __main__
    __main__.FrequencyEncoder = FrequencyEncoder
    __main__.to_sparse = to_sparse


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def load_preprocessor(path: Path = PREPROCESSOR_PATH):
    _register_pickle_symbols()
    return joblib.load(path)


def load_configuration(path: Path = THRESHOLD_PATH) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def get_required_features(preprocessor) -> list[str]:
    features = []
    for _, _, columns in preprocessor.transformers_:
        features.extend(list(columns))
    return features


def validate_transaction(transaction: Mapping[str, Any] | pd.DataFrame, required_features: list[str]) -> pd.DataFrame:
    if isinstance(transaction, pd.Series):
        frame = transaction.to_frame().T
    elif isinstance(transaction, Mapping):
        frame = pd.DataFrame([dict(transaction)])
    elif isinstance(transaction, pd.DataFrame):
        frame = transaction.copy()
    else:
        raise TypeError('transaction must be a mapping, Series, or pandas DataFrame')
    if frame.empty:
        raise ValueError('transaction must contain at least one row')
    # These two numeric columns were created during the original preprocessing
    # from raw high-cardinality fields and must be recreated at inference time.
    for derived, raw_name in [('DeviceInfo_is_missing', 'DeviceInfo'), ('id_33_is_missing', 'id_33')]:
        if derived in required_features and derived not in frame.columns:
            if raw_name in frame.columns:
                frame[derived] = frame[raw_name].isna().astype('int8')
            else:
                raise ValueError(f'Missing required raw feature for derived column {derived}: {raw_name}')
    missing = [c for c in required_features if c not in frame.columns]
    if missing:
        raise ValueError(f'Missing required features ({len(missing)}): {missing[:10]}')
    unexpected = [c for c in frame.columns if c not in required_features and c not in ['TransactionID', 'isFraud']]
    if unexpected:
        raise ValueError(f'Unexpected features ({len(unexpected)}): {unexpected[:10]}')
    feature_frame = frame[required_features].copy()
    for column in feature_frame.columns:
        if pd.api.types.is_numeric_dtype(feature_frame[column]):
            values = pd.to_numeric(feature_frame[column], errors='coerce')
            if np.isinf(values.to_numpy(dtype='float64', na_value=np.nan)).any():
                raise ValueError(f'Infinite numeric value found in {column}')
            feature_frame[column] = values
    return frame


def get_risk_level(probability: float, risk_config: Mapping[str, str]) -> str:
    low = float(str(risk_config['low']).split('<')[1].strip())
    medium_match = str(risk_config['medium'])
    medium = float(medium_match.split('<=')[0].strip())
    high = float(str(risk_config['high']).split('>=')[1].strip())
    if probability < low:
        return 'Low'
    if probability < high:
        return 'Medium'
    return 'High'


def _transform(transaction, preprocessor, required_features):
    frame = validate_transaction(transaction, required_features)
    transformed = preprocessor.transform(frame[required_features])
    return frame, transformed


def predict_transaction(transaction, model=None, preprocessor=None, threshold_config=None) -> dict[str, Any]:
    model = model or load_model()
    preprocessor = preprocessor or load_preprocessor()
    threshold_config = threshold_config or load_configuration()
    required = get_required_features(preprocessor)
    frame, transformed = _transform(transaction, preprocessor, required)
    probability = float(model.predict_proba(transformed)[0, 1])
    threshold = float(threshold_config['classification_threshold'])
    prediction = int(probability >= threshold)
    result = {
        'transaction_id': frame.iloc[0].get('TransactionID', None),
        'fraud_probability': probability,
        'classification_threshold': threshold,
        'prediction': 'Fraud' if prediction else 'Legitimate',
        'risk_level': get_risk_level(probability, threshold_config['risk_levels']),
    }
    return result


def predict_batch(data: pd.DataFrame, model=None, preprocessor=None, threshold_config=None) -> pd.DataFrame:
    model = model or load_model()
    preprocessor = preprocessor or load_preprocessor()
    threshold_config = threshold_config or load_configuration()
    required = get_required_features(preprocessor)
    frame, transformed = _transform(data, preprocessor, required)
    probabilities = model.predict_proba(transformed)[:, 1]
    threshold = float(threshold_config['classification_threshold'])
    output = pd.DataFrame({
        'TransactionID': frame['TransactionID'].to_numpy() if 'TransactionID' in frame else np.arange(len(frame)),
        'Fraud_Probability': probabilities,
        'Classification_Threshold': threshold,
        'Prediction': np.where(probabilities >= threshold, 'Fraud', 'Legitimate'),
        'Risk_Level': [get_risk_level(float(p), threshold_config['risk_levels']) for p in probabilities],
    })
    return output


def explain_transaction(transaction, model=None, preprocessor=None, threshold_config=None, top_n: int = 5) -> dict[str, Any]:
    result = predict_transaction(transaction, model=model, preprocessor=preprocessor, threshold_config=threshold_config)
    model = model or load_model()
    preprocessor = preprocessor or load_preprocessor()
    required = get_required_features(preprocessor)
    frame, transformed = _transform(transaction, preprocessor, required)
    feature_name_parts = []
    for branch_name, branch, columns in preprocessor.transformers_:
        columns = list(columns)
        if branch_name == 'numeric':
            feature_name_parts.extend([f'numeric__{column}' for column in columns])
        elif branch_name == 'low_cardinality':
            one_hot = branch.named_steps['one_hot']
            feature_name_parts.extend([f'low_cardinality__{name}' for name in one_hot.get_feature_names_out(columns)])
        elif branch_name == 'high_cardinality':
            feature_name_parts.extend([f'high_cardinality__{column}' for column in columns])
    feature_names = list(feature_name_parts)
    if len(feature_names) != transformed.shape[1]:
        raise ValueError(f'Feature alignment failed: {len(feature_names)} names for {transformed.shape[1]} transformed columns')
    import shap
    explainer = shap.TreeExplainer(model)
    shap_input = transformed.toarray().astype(np.float32) if hasattr(transformed, 'toarray') else np.asarray(transformed, dtype=np.float32)
    raw = explainer.shap_values(shap_input)
    if isinstance(raw, list):
        values = np.asarray(raw[1])[0]
    else:
        arr = np.asarray(raw)
        values = arr[0, :, 1] if arr.ndim == 3 else arr[0]
    order = np.argsort(np.abs(values))[::-1]
    positive = [{'feature': feature_names[i], 'contribution': float(values[i])} for i in order if values[i] > 0][:top_n]
    negative = [{'feature': feature_names[i], 'contribution': float(values[i])} for i in order if values[i] < 0][:top_n]
    result['top_fraud_factors'] = positive
    result['top_legitimate_factors'] = negative
    return result
