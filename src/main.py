"""Project 07 — Fraud Detection Inference API

FastAPI service wrapping the winning model from Project 06
(XGBoost + class weighting, PR-AUC 0.880, recall@P90 0.837).

Model loading strategy:
  1. Try MLflow Model Registry on DagsHub (production pattern)
  2. Fall back to local models/ artifact if Registry unavailable

Endpoints:
  POST /predict     - fraud probability for a single transaction
  GET  /health      - liveness + readiness probe for Kubernetes
  GET  /metrics     - Prometheus scrape endpoint
  GET  /            - API info
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Prometheus metrics ---
PREDICTION_COUNTER = Counter(
    "fraud_predictions_total",
    "Total number of predictions made",
    ["result"]
)
PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
FRAUD_PROBABILITY = Histogram(
    "fraud_probability_distribution",
    "Distribution of fraud probability scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

MODELS_DIR = Path("models")
LOCAL_MODEL_PATH = MODELS_DIR / "fraud_xgboost_class_weighted.json"
LOCAL_SCALER_PATH = MODELS_DIR / "scaler.joblib"
LOCAL_FEATURES_PATH = MODELS_DIR / "feature_columns.json"

model = None
scaler = None
feature_columns = None
model_source = None


def load_from_registry() -> bool:
    """Try to load model from MLflow Model Registry on DagsHub."""
    try:
        import dagshub
        import dagshub.auth

        token = os.environ.get("DAGSHUB_TOKEN")
        if not token:
            logger.warning("DAGSHUB_TOKEN not set, skipping registry load")
            return False

        dagshub.auth.add_app_token(token)
        dagshub.init(
            repo_owner=os.environ.get("DAGSHUB_USERNAME", "muhammed-keita-ml"),
            repo_name=os.environ.get("DAGSHUB_REPO", "project-06-fraud-detection-pipeline"),
            mlflow=True
        )

        model_name = os.environ.get("MLFLOW_MODEL_NAME", "fraud-xgboost-class-weighted")
        model_stage = os.environ.get("MLFLOW_MODEL_STAGE", "Production")

        global model, scaler, feature_columns, model_source

        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[model_stage])
        if not versions:
            logger.warning("No model version found in stage: %s", model_stage)
            return False

        run_id = versions[0].run_id
        artifact_path = versions[0].source.split("/artifacts/")[-1]
        logger.info("Downloading artifact from run %s: %s", run_id, artifact_path)

        # Download artifact to a controlled local path and load with XGBoost
        # native API — avoids path scheme issues in MLflow
        import tempfile, shutil
        tmp_dir = tempfile.mkdtemp()
        try:
            local_path = client.download_artifacts(run_id, artifact_path, tmp_dir)
            logger.info("Artifact downloaded to: %s", local_path)
            loaded = xgb.XGBClassifier()
            loaded.load_model(local_path)
            model = loaded
            logger.info("XGBoost model loaded from artifact")
        except Exception as load_err:
            logger.warning("Artifact load failed: %s", load_err)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Load scaler and feature columns — from local artifacts if available,
        # otherwise use defaults that work for the standard ULB dataset schema
        if LOCAL_SCALER_PATH.exists():
            scaler = joblib.load(LOCAL_SCALER_PATH)
            feature_columns = json.loads(LOCAL_FEATURES_PATH.read_text())
            logger.info("Scaler loaded from local artifacts")
        else:
            # In containerized deployments without local model files,
            # use a fresh scaler fitted on typical ULB dataset statistics
            # and the standard feature column order
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            default_scaler = StandardScaler()
            # Fit on approximate ULB dataset statistics for Time and Amount
            default_scaler.mean_ = np.array([94813.86, 88.35])
            default_scaler.scale_ = np.array([47488.15, 250.12])
            default_scaler.var_ = np.array([47488.15**2, 250.12**2])
            default_scaler.n_features_in_ = 2
            scaler = default_scaler
            feature_columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
            logger.info("Using default scaler statistics (no local scaler found)")

        model_source = f"mlflow_registry:{model_name}/{model_stage}"
        logger.info("Model loaded from Registry, scaler ready")
        return True

    except Exception as e:
        logger.warning("Registry load failed: %s", e)
        return False

    return False


def load_from_local() -> bool:
    """Fall back to local model artifacts from Project 06."""
    try:
        global model, scaler, feature_columns, model_source

        if not LOCAL_MODEL_PATH.exists():
            logger.error(
                "Local model not found at %s. Copy models/ from Project 06.",
                LOCAL_MODEL_PATH
            )
            return False

        m = xgb.XGBClassifier()
        m.load_model(LOCAL_MODEL_PATH)
        model = m
        scaler = joblib.load(LOCAL_SCALER_PATH)
        feature_columns = json.loads(LOCAL_FEATURES_PATH.read_text())
        model_source = f"local:{LOCAL_MODEL_PATH}"
        logger.info("Model loaded from local artifacts: %s", LOCAL_MODEL_PATH)
        return True

    except Exception as e:
        logger.error("Local model load failed: %s", e)
        return False


def load_model():
    """Registry first, local fallback."""
    if not load_from_registry():
        logger.info("Falling back to local model artifacts")
        if not load_from_local():
            raise RuntimeError(
                "Model could not be loaded from Registry or local artifacts. "
                "Ensure models/ directory contains Project 06 artifacts."
            )
    logger.info("Model ready. Source: %s", model_source)


# --- Pydantic schemas ---
class TransactionFeatures(BaseModel):
    """30 features matching the ULB Credit Card Fraud dataset schema.
    V1-V28 are PCA-anonymized components; Time and Amount are raw values.
    """
    Time: float = Field(..., description="Seconds since first transaction in dataset")
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")


class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: str
    confidence: str
    model_source: str
    latency_ms: float


# --- App ---
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Only attempt model loading if nothing has been injected already
    # (e.g. by test fixtures). This allows tests to pre-set the module-level
    # model/scaler without triggering a real load_model() call.
    global model
    if model is None:
        try:
            load_model()
        except RuntimeError as e:
            # In test environments without model artifacts, allow startup to
            # succeed — individual endpoints return 503 when model is None.
            logger.warning("Model loading skipped during startup: %s", e)
    yield

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Project 07 — XGBoost fraud detection model serving. "
        "Model: XGBoost + class weighting, PR-AUC 0.880, recall@P90 0.837. "
        "See github.com/muhammed-keita-ml/project-07-fraud-detection-k8s"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "model_source": model_source,
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "metrics": "GET /metrics",
        }
    }


@app.get("/health")
def health():
    """Kubernetes liveness and readiness probe."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_source": model_source,
    }


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionFeatures):
    """Classify a single transaction as fraud or legitimate."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()

    row = transaction.model_dump()
    X = pd.DataFrame([row])[feature_columns]
    X_scaled = X.copy()
    X_scaled[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])

    proba = float(model.predict_proba(X_scaled)[0, 1])
    prediction = "fraud" if proba >= 0.5 else "legitimate"

    if proba >= 0.8 or proba <= 0.2:
        confidence = "high"
    elif proba >= 0.6 or proba <= 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    latency_ms = (time.perf_counter() - start) * 1000

    PREDICTION_COUNTER.labels(result=prediction).inc()
    PREDICTION_LATENCY.observe(latency_ms / 1000)
    FRAUD_PROBABILITY.observe(proba)

    logger.info(
        "prediction=%s proba=%.4f latency=%.2fms",
        prediction, proba, latency_ms
    )

    return PredictionResponse(
        fraud_probability=round(proba, 6),
        prediction=prediction,
        confidence=confidence,
        model_source=model_source,
        latency_ms=round(latency_ms, 3),
    )
