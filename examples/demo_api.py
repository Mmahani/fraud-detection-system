"""Local demo client for the fraud-detection API.

Start the service first:

    python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

Then run:

    python examples/demo_api.py

The script uses the synthetic payload in ``examples/sample_transaction.json``.
It does not require the IEEE-CIS dataset and does not send data anywhere except
the local base URL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
PAYLOAD_PATH = Path(__file__).with_name("sample_transaction.json")


def _print(title: str, value: Any) -> None:
    print(f"\n{title}")
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _fail(message: str) -> int:
    print(f"\nERROR: {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Exercise the local fraud-detection API.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Service base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of SHAP factors to request (1-20, default: 3)",
    )
    args = parser.parse_args()

    if not PAYLOAD_PATH.is_file():
        return _fail(f"payload not found: {PAYLOAD_PATH}")

    payload = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))

    try:
        with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
            _print("GET /health", client.get("/health").raise_for_status().json())

            info = client.get("/model/info").raise_for_status().json()
            _print("GET /model/info", info)

            features = client.get("/model/features").raise_for_status().json()
            _print(
                "GET /model/features",
                {
                    "count": features["count"],
                    "derived_features": features["derived_features"],
                },
            )

            prediction = client.post("/predict", json=payload).raise_for_status().json()
            _print("POST /predict", prediction)

            batch = (
                client.post("/predict/batch", json=[payload, payload])
                .raise_for_status()
                .json()
            )
            _print("POST /predict/batch", {"count": batch["count"], "predictions": batch["predictions"]})

            explanation = (
                client.post("/explain", params={"top_n": args.top_n}, json=payload)
                .raise_for_status()
                .json()
            )
            _print(
                "POST /explain",
                {
                    "fraud_probability": explanation["fraud_probability"],
                    "prediction": explanation["prediction"],
                    "risk_level": explanation["risk_level"],
                    "top_fraud_factors": explanation["top_fraud_factors"],
                    "top_legitimate_factors": explanation["top_legitimate_factors"],
                },
            )

            invalid = client.post("/predict", json={})
            _print(
                "POST /predict with an empty object (expected 422)",
                {"status_code": invalid.status_code, "detail": invalid.json().get("detail")},
            )

    except httpx.ConnectError:
        return _fail(
            f"could not connect to {args.base_url}. Start the API first:\n"
            "  python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000"
        )
    except httpx.TimeoutException:
        return _fail("the request timed out; /explain can take several seconds on first use")
    except httpx.HTTPStatusError as exc:
        return _fail(f"HTTP {exc.response.status_code}: {exc.response.text}")
    except json.JSONDecodeError as exc:
        return _fail(f"invalid JSON: {exc}")

    print("\nDemo completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
