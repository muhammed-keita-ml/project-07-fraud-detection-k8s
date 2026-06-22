"""Tests for Project 07 FastAPI inference service.

Run with: pytest tests/ -v

These tests validate the API endpoints without requiring a real
model to be loaded — they mock the model and scaler so the test
suite runs in CI without needing the full model artifacts.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


SAMPLE_TRANSACTION = {
    "Time": 80000.0,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
    "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.14, "V10": -0.33,
    "V11": -0.47, "V12": 0.21, "V13": 0.02, "V14": 0.40, "V15": 0.09,
    "V16": 0.59, "V17": -0.27, "V18": 0.82, "V19": 0.75, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
    "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62,
}

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def make_mock_model(proba: float = 0.05):
    """Return a mock XGBoost model that always predicts `proba`."""
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[1 - proba, proba]])
    return mock


def make_mock_scaler():
    """Return a mock StandardScaler that returns input unchanged."""
    mock = MagicMock()
    mock.transform.side_effect = lambda x: x
    return mock


@pytest.fixture
def client():
    """FastAPI test client with mocked model and scaler."""
    import src.main as main_module
    main_module.model = make_mock_model(proba=0.05)
    main_module.scaler = make_mock_scaler()
    main_module.feature_columns = FEATURE_COLUMNS
    main_module.model_source = "mock:test"

    from src.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    # Reset after test
    main_module.model = None


@pytest.fixture
def fraud_client():
    """FastAPI test client mocked to return high fraud probability."""
    import src.main as main_module
    main_module.model = make_mock_model(proba=0.95)
    main_module.scaler = make_mock_scaler()
    main_module.feature_columns = FEATURE_COLUMNS
    main_module.model_source = "mock:test"

    from src.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    main_module.model = None


@pytest.fixture
def no_model_client():
    """FastAPI test client with no model loaded — for 503 tests."""
    import src.main as main_module

    from src.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        # Set model to None AFTER startup so the lifespan doesn't reload it
        original_model = main_module.model
        main_module.model = None
        main_module.model_source = None
        yield c
        # Restore after test
        main_module.model = original_model


class TestRootEndpoint:
    def test_root_returns_service_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Fraud Detection API"
        assert "predict" in data["endpoints"]
        assert "health" in data["endpoints"]
        assert "metrics" in data["endpoints"]


class TestHealthEndpoint:
    def test_health_returns_healthy(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_returns_503_when_model_not_loaded(self, no_model_client):
        response = no_model_client.get("/health")
        assert response.status_code == 503


class TestPredictEndpoint:
    def test_predict_legitimate_transaction(self, client):
        response = client.post("/predict", json=SAMPLE_TRANSACTION)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "legitimate"
        assert data["fraud_probability"] < 0.5
        assert data["confidence"] in ["high", "medium", "low"]
        assert "latency_ms" in data
        assert data["latency_ms"] > 0

    def test_predict_fraudulent_transaction(self, fraud_client):
        response = fraud_client.post("/predict", json=SAMPLE_TRANSACTION)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "fraud"
        assert data["fraud_probability"] > 0.5
        assert data["confidence"] == "high"

    def test_predict_returns_503_when_model_not_loaded(self, no_model_client):
        response = no_model_client.post("/predict", json=SAMPLE_TRANSACTION)
        assert response.status_code == 503

    def test_predict_rejects_negative_amount(self, client):
        bad_transaction = {**SAMPLE_TRANSACTION, "Amount": -10.0}
        response = client.post("/predict", json=bad_transaction)
        assert response.status_code == 422

    def test_predict_rejects_missing_field(self, client):
        incomplete = {k: v for k, v in SAMPLE_TRANSACTION.items() if k != "V1"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_confidence_levels(self, client):
        import src.main as main_module

        for proba, expected_confidence in [
            (0.95, "high"),
            (0.65, "medium"),
            (0.45, "low"),
        ]:
            main_module.model = make_mock_model(proba=proba)
            from src.main import app
            with TestClient(app) as c:
                response = c.post("/predict", json=SAMPLE_TRANSACTION)
                assert response.json()["confidence"] == expected_confidence


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_format(self, client):
        client.post("/predict", json=SAMPLE_TRANSACTION)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"fraud_predictions_total" in response.content
        assert b"fraud_prediction_latency_seconds" in response.content