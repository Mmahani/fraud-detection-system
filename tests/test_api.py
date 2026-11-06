"""Unit tests for the FastAPI prediction service."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api.main import app

SAMPLE_PATH = PROJECT_ROOT / "examples" / "sample_transaction.json"
MODEL_PATH = PROJECT_ROOT / "models" / "best_tuned_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "merged_data.csv.gz"


@unittest.skipUnless(MODEL_PATH.is_file(), "model artifacts are not available")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.client.__enter__()
        cls.sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_root(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["docs"], "/docs")

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_loaded"])
        self.assertAlmostEqual(body["classification_threshold"], 0.35)

    def test_model_info(self) -> None:
        response = self.client.get("/model/info")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_type"], "RandomForestClassifier")
        self.assertEqual(body["raw_feature_count"], 434)
        self.assertEqual(body["transformed_feature_count"], 495)
        self.assertAlmostEqual(body["classification_threshold"], 0.35)

    def test_features(self) -> None:
        response = self.client.get("/model/features")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 434)
        self.assertIn("TransactionAmt", body["features"])
        self.assertIn("DeviceInfo_is_missing", body["features"])

    def test_predict(self) -> None:
        response = self.client.post("/predict", json=self.sample)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertGreaterEqual(body["fraud_probability"], 0.0)
        self.assertLessEqual(body["fraud_probability"], 1.0)
        self.assertIn(body["prediction"], {"Fraud", "Legitimate"})
        self.assertIn(body["risk_level"], {"Low", "Medium", "High"})
        self.assertAlmostEqual(body["classification_threshold"], 0.35)

    def test_predict_missing_feature(self) -> None:
        incomplete = dict(self.sample)
        incomplete.pop("TransactionAmt")
        response = self.client.post("/predict", json=incomplete)
        self.assertEqual(response.status_code, 422)
        self.assertIn("Missing required features", response.json()["detail"])

    @unittest.skipUnless(DATA_PATH.is_file(), "processed data is not available")
    def test_json_nulls_match_training_path(self) -> None:
        import pandas as pd

        from src.prediction import predict_transaction

        frame = pd.read_csv(DATA_PATH, nrows=1)
        direct = predict_transaction(frame)
        payload = json.loads(frame.to_json(orient="records"))[0]
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertAlmostEqual(
            response.json()["fraud_probability"],
            direct["fraud_probability"],
            places=12,
        )

    def test_predict_unexpected_feature(self) -> None:
        extended = dict(self.sample)
        extended["not_a_model_feature"] = 1
        response = self.client.post("/predict", json=extended)
        self.assertEqual(response.status_code, 422)
        self.assertIn("Unexpected features", response.json()["detail"])

    def test_batch_predict(self) -> None:
        response = self.client.post("/predict/batch", json=[self.sample, self.sample])
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["count"], 2)
        self.assertEqual(len(body["predictions"]), 2)
        self.assertEqual(body["predictions"][0]["prediction"], body["predictions"][1]["prediction"])

    def test_empty_batch(self) -> None:
        response = self.client.post("/predict/batch", json=[])
        self.assertEqual(response.status_code, 422)

    def test_explain(self) -> None:
        response = self.client.post("/explain?top_n=2", json=self.sample)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("top_fraud_factors", body)
        self.assertIn("top_legitimate_factors", body)
        self.assertLessEqual(len(body["top_fraud_factors"]), 2)
        self.assertLessEqual(len(body["top_legitimate_factors"]), 2)


if __name__ == "__main__":
    unittest.main()
