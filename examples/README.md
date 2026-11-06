# Examples

Local API examples for the fraud-detection research service.

| File | Purpose |
|---|---|
| [`sample_transaction.json`](sample_transaction.json) | Synthetic transaction payload used by the API examples |
| [`demo_api.py`](demo_api.py) | Runs the main API endpoints against a local service |

## The payload

`sample_transaction.json` is a **fully synthetic** transaction record with 433
fields: 432 model inputs plus `TransactionID` for reference. No value in the file
comes from the IEEE-CIS dataset or from a real transaction.

The model expects 434 inputs because two missingness flags are derived by the
service from raw fields:

- `DeviceInfo_is_missing` (from `DeviceInfo`)
- `id_33_is_missing` (from `id_33`)

The API adds those flags during preprocessing, so clients do not need to send
them. `TransactionID` is accepted but is not a model feature.

## Run the demo

Start the API from the project root:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Then, in a second terminal:

```powershell
python examples/demo_api.py
```

The script prints:

- `/health` status
- model metadata and feature counts
- one prediction
- a two-row batch prediction
- a SHAP explanation
- the `422` response for an empty object

Use `--base-url` for a different port:

```powershell
python examples/demo_api.py --base-url http://127.0.0.1:8001
```

## Safety

- The payload is synthetic; do not replace it with real customer data.
- The service is local-only and has no authentication, rate limiting, or TLS.
- See [`docs/DEMO.md`](../docs/DEMO.md) for the full walkthrough and
  [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the request flow.
- See [`DATA_LICENSE.md`](../DATA_LICENSE.md) for the data and model rights
  notice.
