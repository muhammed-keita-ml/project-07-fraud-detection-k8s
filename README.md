# Project 07 — Fraud Detection: Cloud-Native Deployment + Kubernetes
[**Technical Report**](./project07_technical_report.md) — Full methodology, results, and research directions

## Overview
Cloud-native deployment of the winning fraud detection model from Project 06
(XGBoost + class weighting, PR-AUC 0.880, recall@P90 0.837). The repository
contains a FastAPI inference service, Docker packaging, MLflow Model Registry
integration, AWS ECR publishing through GitHub Actions, Prometheus metrics,
and Kubernetes manifests. Claims below distinguish implemented configuration
from runtime behavior that has not yet been verified.

## Architecture

```
XGBoost Fraud Model (Project 06)
        ↓
MLflow Model Registry (DagsHub) ──► fallback: local artifacts
        ↓
FastAPI Inference Service
  POST /predict   fraud probability + confidence
  GET  /health    liveness + readiness probes
  GET  /metrics   Prometheus scrape endpoint
        ↓
Docker Container (multi-stage, non-root user)
        ↓
AWS ECR (597819998212.dkr.ecr.eu-west-1.amazonaws.com/fraud-detection-api)
        ↓
┌────────────────────────────────────────┐
│           AWS EC2 (cloud)              │
│  Instance: t3.micro · eu-west-1        │
│  Docker container running from ECR     │
│  Previously smoke-tested endpoint       │
│                                        │
│  K8s manifests: 6 resources             │
│  Configuration-validated only           │
└────────────────────────────────────────┘
        ↓
Prometheus instrumentation and local Compose configuration
  Runtime operation of the full monitoring stack not verified
        ↓
GitHub Actions CI/CD
  test → build → push to ECR (on push to main)
```

## Project Structure

```
project-07-fraud-detection-k8s/
├── src/
│   └── main.py              # FastAPI inference service
├── k8s/
│   ├── deployment.yaml      # K8s Deployment (2 replicas, health checks)
│   └── manifests.yaml       # Service, ConfigMap, Secret, HPA, Ingress
├── monitoring/
│   └── prometheus.yml       # Prometheus scrape config
├── tests/
│   └── test_api.py          # API test suite (mocked model)
├── .github/workflows/
│   └── ci-cd.yml            # Build → test → push to ECR
├── Dockerfile               # Multi-stage build, non-root user
├── docker-compose.yml       # Local dev: API + Prometheus + Grafana
├── requirements.txt
└── README.md
```

## Running Locally

### 1. Set up environment
```bash
python -m venv venv
venv\Scripts\Activate.ps1      # Windows
pip install -r requirements.txt
```

### 2. Copy model artifacts from Project 06
```bash
cp ..\project-06-fraud-detection-pipeline\models\* .\models\
```

### 3. Configure credentials
```bash
copy .env.example .env
# Edit .env with your DAGSHUB_TOKEN
```

### 4. Run tests
```bash
pytest tests/ -v
```

### 5. Run the API
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Run full stack (API + Prometheus + Grafana)
```bash
docker-compose up --build
```

## Docker Build & Push to ECR

```bash
# Authenticate with ECR
aws ecr get-login-password --region eu-west-1 | \
  docker login --username AWS --password-stdin \
  597819998212.dkr.ecr.eu-west-1.amazonaws.com

# Build and push
docker build -t fraud-detection-api .
docker tag fraud-detection-api:latest \
  597819998212.dkr.ecr.eu-west-1.amazonaws.com/fraud-detection-api:latest
docker push \
  597819998212.dkr.ecr.eu-west-1.amazonaws.com/fraud-detection-api:latest
```

## Kubernetes manifests (runtime not yet verified)

The repository includes six Kubernetes resources. The manifests have been
configuration/YAML-validated, but deployment to a running Minikube or EKS
cluster has not been verified. The commands below are a deployment procedure,
not evidence that the deployment has been tested. Local Minikube may also
require hardware virtualization to be enabled.

```bash
# Start Minikube
minikube start --driver=docker

# Enable Ingress
minikube addons enable ingress

# Create secret (replace with real token)
kubectl create secret generic fraud-detection-secrets \
  --from-literal=DAGSHUB_TOKEN=<your_token>

# Apply all manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/manifests.yaml

# Check status
kubectl get pods
kubectl get services

# Get service URL
minikube service fraud-detection-api --url
```

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 80000,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.14, "V10": -0.33, "V11": -0.47, "V12": 0.21,
    "V13": 0.02, "V14": 0.40, "V15": 0.09, "V16": 0.59,
    "V17": -0.27, "V18": 0.82, "V19": 0.75, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62
  }'
```

Example response shape (values and model source vary by request and runtime):
```json
{
  "fraud_probability": 0.023451,
  "prediction": "legitimate",
  "confidence": "high",
  "model_source": "<loaded model source>",
  "latency_ms": 2.341
}
```

## GitHub Actions CI/CD

Required GitHub Secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `DAGSHUB_TOKEN`

On every push to `main`: tests run → Docker image builds → pushes to ECR
with both `latest` and commit SHA tags.

## Project Status
- **Project 06**: Model training, MLflow, and imbalance strategy comparison completed.
- **Project 07** (this repo): FastAPI service, registry integration, Docker packaging,
  AWS ECR CI workflow, Prometheus instrumentation, and Kubernetes manifests are present.
- Kubernetes cluster runtime and full Prometheus/Grafana stack runtime have not been verified.
- GitHub Actions builds and pushes images to ECR; it does **not** automatically deploy them to EC2.
- **Project 08 (FRIP)**: Capstone — fraud detection + credit scoring +
  monitoring as a unified multi-model platform

## Tech Stack
FastAPI, Uvicorn, XGBoost, MLflow, DagsHub, Docker, AWS ECR, AWS EC2,
Kubernetes manifests, Prometheus instrumentation, Docker Compose configuration, GitHub Actions, pytest

## Links
- **Project 06**: https://github.com/muhammed-keita-ml/project-06-fraud-detection-pipeline
- **DagsHub**: https://dagshub.com/muhammed-keita-ml/project-06-fraud-detection-pipeline
- **ECR Repository**: 597819998212.dkr.ecr.eu-west-1.amazonaws.com/fraud-detection-api
- **Medium**: https://medium.com/@mkeitaone/from-model-to-production-what-building-a-cloud-native-fraud-detection-api-revealed-about-ml-e7c1f655e072

## Model Registry

This service loads its model from the MLflow Model Registry in the
Project 06 repository — the deployment repo and the model repo are
intentionally separate, following the principle that model versioning
and infrastructure versioning should be independent.

- **Registry**: https://dagshub.com/muhammed-keita-ml/project-06-fraud-detection-pipeline
- **Model name**: fraud-xgboost-class-weighted
- **Stage**: Production
- **Version/stage**: Production registry reference (the exact resolved model version/run can change; verify the registry before deployment)
- **Trained in**: [Project 06 — Credit Card Fraud Detection Pipeline](https://github.com/muhammed-keita-ml/project-06-fraud-detection-pipeline)

## Deployment Status
An EC2 endpoint was smoke-tested previously, but current availability and the exact image digest running on EC2 have not been freshly verified. The GitHub Actions workflow publishes images to ECR; EC2 deployment is a separate operation.

## Links
- **GitHub**: https://github.com/muhammed-keita-ml/project-07-fraud-detection-k8s
- **Live API**: http://108.128.140.230:8000/docs
- **ECR Repository**: 597819998212.dkr.ecr.eu-west-1.amazonaws.com/fraud-detection-api
- **Project 06**: https://github.com/muhammed-keita-ml/project-06-fraud-detection-pipeline
- **DagsHub / MLflow**: https://dagshub.com/muhammed-keita-ml/project-06-fraud-detection-pipeline
