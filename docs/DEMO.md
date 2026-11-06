# API Demo

A local walkthrough of the FastAPI service. The demo uses only synthetic data and
does not require the IEEE-CIS dataset.

> The service is a **local research tool**. It has no authentication, rate
> limiting, or TLS. Run it on `127.0.0.1` and never expose it directly to the
> public internet.

## 1. Prerequisites

- Python 3.12.x (tested with 3.12.6)
- The repository dependencies installed:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, replace `py` with `python` and activate with
`source .venv/bin/activate`.

## 2. Start the service

From the project root:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Wait until the log reports `Application startup complete`. The first start can
take several seconds because the model and SHAP stack are imported.

Interactive docs: <http://127.0.0.1:8000/docs>

## 3. Run the automated demo

In a second terminal, with the same environment active:

```powershell
python examples/demo_api.py
```

The script calls `/health`, `/model/info`, `/model/features`, `/predict`,
`/predict/batch`, and `/explain`, then prints a compact summary. Use a different
port if needed:

```powershell
python examples/demo_api.py --base-url http://127.0.0.1:8001
```

Source: [`examples/demo_api.py`](../examples/demo_api.py)

## 4. Manual walkthrough

### 4.1 Health

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "classification_threshold": 0.35
}
```

### 4.2 Model metadata

```powershell
Invoke-RestMethod http://127.0.0.1:8000/model/info
```

```json
{
  "model_type": "RandomForestClassifier",
  "raw_feature_count": 434,
  "transformed_feature_count": 495,
  "classification_threshold": 0.35,
  "risk_levels": {
    "low": "probability < 0.30",
    "medium": "0.30 <= probability < 0.70",
    "high": "probability >= 0.70"
  },
  "explainability_available": true
}
```

### 4.3 Single prediction

```powershell
$body = Get-Content examples/sample_transaction.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" -Body $body
```

Example response for the synthetic payload:

```json
{
  "transaction_id": 990000001,
  "fraud_probability": 0.09821428571428573,
  "classification_threshold": 0.35,
  "prediction": "Legitimate",
  "risk_level": "Low"
}
```

The payload is fully synthetic. It has 433 fields: 432 model inputs plus
`TransactionID`. The service derives the two `*_is_missing` flags from the raw
`DeviceInfo` and `id_33` fields, which brings the model input to 434 features.

### 4.4 Explanation

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/explain?top_n=3" `
  -ContentType "application/json" -Body $body
```

Example response (values rounded):

```json
{
  "transaction_id": 990000001,
  "fraud_probability": 0.0982,
  "classification_threshold": 0.35,
  "prediction": "Legitimate",
  "risk_level": "Low",
  "top_fraud_factors": [
    {"feature": "numeric__V187", "contribution": 0.0140},
    {"feature": "numeric__C4", "contribution": 0.0111},
    {"feature": "numeric__V91", "contribution": 0.0075}
  ],
  "top_legitimate_factors": [
    {"feature": "numeric__C14", "contribution": -0.0340},
    {"feature": "numeric__card3", "contribution": -0.0264},
    {"feature": "numeric__V258", "contribution": -0.0225}
  ]
}
```

Positive contributions push the class-1 (Fraud) output up; negative
contributions push it down. They are model associations, not causal effects.

### 4.5 Batch prediction

```powershell
$items = @(Get-Content examples/sample_transaction.json -Raw | ConvertFrom-Json,
           Get-Content examples/sample_transaction.json -Raw | ConvertFrom-Json)
$batch = $items | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict/batch `
  -ContentType "application/json" -Body $batch
```

The array may contain up to 500 transactions. An empty array returns `422`;
more than 500 returns `413`.

### 4.6 Validation error

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" -Body "{}"
```

```json
{"detail": "transaction must contain at least one row"}
```

Missing or unexpected fields also return `422`.

## 5. Python client example

```python
import json
import httpx

payload = json.load(open("examples/sample_transaction.json", encoding="utf-8"))

with httpx.Client(base_url="http://127.0.0.1:8000", timeout=60.0) as client:
    health = client.get("/health").json()
    prediction = client.post("/predict", json=payload).json()
    explanation = client.post("/explain", params={"top_n": 3}, json=payload).json()

print(health)
print(prediction)
print(explanation["top_fraud_factors"])
```

## 6. Endpoint summary

| Method | Path | Body | Purpose |
|---|---|---|---|
| `GET` | `/` | — | Service name and version |
| `GET` | `/health` | — | Liveness and model-load status |
| `GET` | `/model/info` | — | Model type, feature counts, threshold, risk bands |
| `GET` | `/model/features` | — | Required raw features and derived fields |
| `POST` | `/predict` | one transaction object | Score one transaction |
| `POST` | `/predict/batch` | array of transaction objects | Score up to 500 transactions |
| `POST` | `/explain` | one transaction object | Score and return SHAP factors |

## 7. Cleanup

Stop the server with `Ctrl+C` in the terminal running `uvicorn`.

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` | Server not started or wrong port | Start `uvicorn`; check the URL and port |
| Model fails to load | Missing `models/*.pkl` or wrong scikit-learn version | Use `scikit-learn==1.9.0` from `requirements.txt` |
| `InconsistentVersionWarning` | Different scikit-learn version | Reinstall the pinned version |
| `422` on `/predict` | Missing or unexpected fields | Compare the payload with `/model/features` |
| `explain` is slow | SHAP explainer initialization | Allow several seconds; it is expected |

## 9. Safety notes

- Use only synthetic or appropriately licensed data.
- Do not send real transaction data to this local service unless you are
  authorized to process it.
- Do not expose the service to a network without authentication and TLS.
- The output is a research signal, not a fraud decision.
