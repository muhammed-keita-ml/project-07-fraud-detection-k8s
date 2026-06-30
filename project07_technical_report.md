# Cloud-Native ML Deployment: Architecture, Reproducibility Constraints, and Production Infrastructure for a Fraud Detection Inference Service

**Author:** Muhammed Keita
**Date:** 2026
**Repository:** https://github.com/muhammed-keita-ml/project-07-fraud-detection-k8s
**Live System:** http://108.128.140.230:8000/docs
**Model Registry:** https://dagshub.com/muhammed-keita-ml/project-06-fraud-detection-pipeline

-----

## Abstract

The deployment of trained ML models into production systems introduces a distinct class of engineering and research problems that model training pipelines do not address: model versioning and registry integration, containerisation for reproducibility, orchestration for reliability, observability for operational monitoring, and CI/CD automation for safe, auditable deployments. This report documents the design and implementation of a cloud-native inference service for the fraud detection model developed in Project 06 (XGBoost + class weighting, PR-AUC 0.880, Recall@P90 0.837). The system serves predictions via a FastAPI REST API, loads the production model from the MLflow Model Registry on DagsHub at container startup with local artifact fallback, builds and pushes Docker images to AWS ECR via GitHub Actions CI/CD, and deploys on AWS EC2 with Prometheus instrumentation and six validated Kubernetes manifests. A central design contribution is the Registry-first loading strategy, which decouples model versioning from infrastructure versioning – enabling model updates without container rebuilds. This work raises open questions about model-infrastructure versioning coordination, deployment performance gating, and serving latency characterisation that motivate the multi-model architecture of Project 08 (FRIP).

-----

## 1. Introduction

### 1.1 Problem Context

The gap between a working ML model and a production ML system is larger than most ML engineering tutorials suggest. A trained model in a notebook or experiment run has no mechanism for: serving predictions reliably under concurrent load; recovering from container failures without manual intervention; loading the correct model version without human oversight; or exposing enough observability that degraded performance is detected before users are affected.

Project 06 produced a fraud detection model that works. Project 07 addresses the harder question: how do you deploy it so it can be trusted in production? The answer requires deliberate architectural decisions at the serving, containerisation, orchestration, and observability layers – each of which introduces tradeoffs that are not visible at the model training layer.

### 1.2 Engineering and Research Questions Addressed

This project addresses three interrelated questions:

1. **Model-infrastructure versioning:** How should model versioning (MLflow Registry stages and versions) and infrastructure versioning (Docker image tags, git SHA) be coordinated in a system where both may change independently?
1. **Reproducibility under environment constraints:** How should CI/CD pipelines be designed when the development environment cannot run Docker locally (due to hardware virtualisation constraints), and what guarantees does a cloud-based build provide?
1. **Serving observability:** What instrumentation is necessary and sufficient for a fraud detection API to be operationally observable – and what questions does instrumentation surface that experiment tracking cannot answer?

### 1.3 Contributions

- A Registry-first model loading strategy that decouples model versioning from infrastructure versioning, with a local artifact fallback for resilience
- A multi-stage Dockerfile with non-root user and health check, resolving a silent failure mode where missing home directory caused DagsHub token cache writes to fail with a PermissionError
- A GitHub Actions CI/CD pipeline that builds and pushes Docker images to AWS ECR on every green push to main – specifically designed for environments where local Docker builds are not available
- A 10-test pytest suite covering API contract, input validation, error handling, and Prometheus metrics output, with a fix for the FastAPI lifespan/test fixture interaction that causes 503 test cases to pass spuriously
- Six validated Kubernetes manifests (Deployment, Service, HPA, ConfigMap, Secret, Ingress) with documented design rationale
- Prometheus instrumentation exposing prediction counter, latency histogram, and fraud probability distribution – the last of which provides a lightweight proxy for covariate drift detection
- A live inference endpoint deployed on AWS EC2 (eu-west-1) with Elastic IP for stable addressability

-----

## 2. Related Work

**Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., et al. (2015).** Hidden Technical Debt in Machine Learning Systems. NeurIPS. Provides the foundational taxonomy of ML system complexity that motivates the architectural decisions here. Their concept of “pipeline glue code” – the coupling between data processing, model training, and serving – directly motivated the decision to maintain separate repositories for the training pipeline (Project 06) and the serving infrastructure (Project 07), with the MLflow Registry as the explicit interface between them.

**Breck, E., Cai, S., Nielsen, E., Salib, M., and Sculley, D. (2017).** The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. IEEE Big Data. Provides the testing rubric used to design the 10-test API suite. Their data tests, model tests, and infrastructure tests map directly to the three test classes implemented: input validation tests (Pydantic schema), model behaviour tests (mocked XGBoost predictions), and infrastructure tests (health check 503 under no-model conditions, Prometheus metric output).

**Gama, J., Zliobaite, I., Bifet, A., Pechenizkiy, M., and Bouchachia, A. (2014).** A Survey on Concept Drift Adaptation. ACM Computing Surveys. Motivates the fraud_probability_distribution Prometheus histogram. Their taxonomy of drift types – covariate shift, concept drift, label drift – clarifies why aggregate probability distribution monitoring is a proxy, not a substitute, for explicit drift detection. The histogram provides early-warning signal; the Project 06 pipeline provides the ground-truth metric (Recall@P90) that would be needed to validate whether the proxy is predictive of actual performance degradation.

**Klaise, J., Van Looveren, A., Vacanti, G., and Coca, A. (2021).** Alibi Detect: Algorithms for Outlier, Adversarial and Drift Detection. JMLR. Provides the reference architecture for production drift detection systems. The Prometheus instrumentation in this project is a lightweight precursor to the statistical test-based monitoring that Alibi Detect implements. The design decision to expose raw probability distributions (rather than pre-computed drift scores) at the metrics endpoint preserves flexibility to add statistical tests downstream without modifying the serving code.

-----

## 3. System Architecture

### 3.1 Overview

The system consists of five layers, each independently versioned and replaceable:

```
MLflow Model Registry (DagsHub) -- model versioning layer
|
FastAPI inference service -- serving layer
|
Docker container (AWS ECR) -- packaging layer
|
AWS EC2 (eu-west-1) -- compute layer
|
Prometheus + Grafana -- observability layer
```

Kubernetes manifests provide the orchestration specification for promotion to a managed cluster (EKS) when the workload requires horizontal scaling.

### 3.2 Model Loading Strategy

The central architectural decision is how the serving container loads its model. Three approaches were considered:

**Option 1: Bake the model into the container image.** The model artifact is copied into the Docker image at build time. Simple and self-contained, but couples model versioning to container versioning – updating the model requires a container rebuild and redeployment.

**Option 2: Mount the model as a volume.** The container loads the model from a mounted filesystem path at runtime. Flexible, but requires volume management infrastructure and introduces a dependency on the host filesystem.

**Option 3: Load from a model registry at startup (chosen).** The container queries the MLflow Model Registry at startup and downloads the Production-stage model. Model updates are applied by promoting a new Registry version – no container rebuild required. A local artifact fallback provides resilience if the Registry is unavailable.

Option 3 was selected because it implements the principle from Sculley et al. (2015) that model versioning and infrastructure versioning should be decoupled. The Registry is the explicit, versioned interface between the training pipeline and the serving infrastructure.

### 3.3 Registry Loading Implementation

The implementation uses MLflow’s MlflowClient directly rather than the high-level mlflow.xgboost.load_model interface:

```python
client = mlflow.tracking.MlflowClient()
versions = client.get_latest_versions(model_name, stages=["Production"])
run_id = versions[0].run_id
artifact_path = versions[0].source.split("/artifacts/")[-1]

tmp_dir = tempfile.mkdtemp()
local_path = client.download_artifacts(run_id, artifact_path, tmp_dir)
loaded = xgb.XGBClassifier()
loaded.load_model(local_path)
```

This approach was adopted after discovering that mlflow.xgboost.load_model resolves the downloaded artifact path through MLflow’s artifact scheme resolver, which fails on certain platforms with the error “Could not find a registered artifact repository for: [path]”. The lower-level download_artifacts + native XGBoost load_model bypasses the scheme resolver, providing more portable behaviour across Linux (production) and Windows (development) environments.

### 3.4 Container Design

The Dockerfile uses a two-stage build:

**Stage 1 (builder):** Installs all Python dependencies into a separate prefix directory using pip install –prefix=/install. Build tools (gcc, pip cache) remain in this stage only.

**Stage 2 (runtime):** Copies only the installed packages from the builder stage. The runtime image contains no build tooling, reducing image size and attack surface.

Non-root user setup:

```dockerfile
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser
```

The -m -d /home/appuser flags create the home directory. This resolved a silent failure mode discovered during deployment: DagsHub’s token cache writes to ~/.dagshub at authentication time. Without a home directory, this write fails with PermissionError – the error is caught by the Registry load fallback, causing the container to silently load from local artifacts (if present) or fail to start (if not). The failure produces no clear error message at the Docker or orchestration layer.

### 3.5 Kubernetes Manifests

Six manifests were written and validated:

|Resource |Purpose |
|----------|----------------------------------------------------------------------------------------------|
|Deployment|2 replicas, liveness/readiness probes on /health, resource limits (256Mi-512Mi, 250m-500m CPU)|
|Service |NodePort exposure; port 80 -> container 8000 |
|HPA |Scale 2-5 replicas at 70% CPU / 80% memory utilisation |
|ConfigMap |Non-sensitive configuration (DagsHub username, repo, model name, stage) |
|Secret |DagsHub token (populated at deploy time, not committed) |
|Ingress |nginx ingress controller; host fraud-detection.local |

The minimum replica count of 2 in the HPA is deliberate. During a model version promotion (updating the Production stage in the Registry), one replica will restart to load the new model. With minimum replicas = 1, this creates a brief availability gap. With minimum replicas = 2, one replica continues serving while the other restarts.

The ConfigMap/Secret separation mirrors the .env/.env.example pattern from local development: non-sensitive configuration is committed (ConfigMap), sensitive credentials are populated at deploy time (Secret).

-----

## 4. CI/CD Pipeline

### 4.1 Design Rationale

Local Docker builds were not available during development: the host OS does not support hardware virtualisation (VT-x/AMD-V), which Docker Desktop requires on Windows. Rather than work around this at the OS level, the CI/CD pipeline was redesigned to build in GitHub Actions (which runs on Ubuntu VMs with full virtualisation support) and push the resulting image to AWS ECR.

This constraint produced a stronger design than the originally planned local-build approach: the image in ECR is always built in a clean, controlled environment (GitHub’s Ubuntu runner), not a developer machine whose local environment may differ from production. The guarantee – “this image was built from a green test run in a clean environment” – is stronger than “this image was built locally and pushed.”

### 4.2 Pipeline Structure

```
push to main
|
test job (ubuntu-latest)
|-- checkout code
|-- setup Python 3.11
|-- pip install -r requirements.txt
|-- pytest tests/ -v --tb=short
|
[if tests pass AND branch is main]
|
build-and-push job
|-- configure AWS credentials (from GitHub Secrets)
|-- ECR login
|-- docker build -t [registry]/[repo]:latest -t [registry]/[repo]:[sha] .
|-- docker push :latest
|-- docker push :[sha]
```

### 4.3 Dual Tagging

Every image is tagged with both latest and the git commit SHA (github.sha). The latest tag provides a stable reference for deployment. The SHA tag provides immutability and traceability: every image in ECR corresponds to an exact, identifiable commit. If a model serving problem appears in production, the running image can be traced to a specific commit and the build reproduced exactly.

-----

## 5. API Design

### 5.1 Endpoint Structure

|Endpoint|Method|Purpose |
|--------|------|----------------------------------------------|
|/ |GET |Service info, model source, endpoint directory|
|/health |GET |Liveness and readiness probe |
|/metrics|GET |Prometheus scrape endpoint |
|/predict|POST |Fraud classification for a single transaction |

### 5.2 Input Validation

The 30-feature input schema uses Pydantic with a non-negativity constraint on Amount:

```python
Amount: float = Field(..., ge=0, description="Transaction amount in USD")
```

A negative transaction amount is a data quality issue, not a fraud signal. It should be rejected before reaching the model – a 422 validation error with a clear message is more useful than a model prediction on malformed input.

### 5.3 Response Structure

```json
{
"fraud_probability": 0.000001,
"prediction": "legitimate",
"confidence": "high",
"model_source": "mlflow_registry:fraud-xgboost-class-weighted/Production",
"latency_ms": 87.818
}
```

The confidence field is derived from the probability score: high (>=0.8 or <=0.2), medium (>=0.6 or <=0.4), low (otherwise). This is an operational affordance for fraud analysts: a binary prediction without confidence information forces every flagged transaction through the same review process regardless of model certainty.

The model_source field exposes the exact model version serving the prediction, enabling traceability from a specific prediction to the exact MLflow Registry version and run that produced the model weights.

### 5.4 Prometheus Instrumentation

Three metrics are exposed at /metrics:

```python
PREDICTION_COUNTER = Counter(
"fraud_predictions_total", "Total predictions", ["result"]
)
PREDICTION_LATENCY = Histogram(
"fraud_prediction_latency_seconds", "Prediction latency in seconds",
buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
FRAUD_PROBABILITY = Histogram(
"fraud_probability_distribution", "Distribution of fraud scores",
buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
```

The fraud_probability_distribution histogram is the most operationally significant. In a healthy fraud detection system, the distribution should be strongly bimodal: most transactions cluster near 0 (clearly legitimate), a small fraction cluster near 1 (clearly fraudulent). Distribution shift – mass accumulating in the 0.3-0.7 range, or the near-zero cluster moving upward – is a proxy signal for input distribution change or model degradation. This does not replace explicit drift detection (which requires reference distributions and statistical tests, per Klaise et al. 2021) but provides a lightweight early-warning indicator available without ground truth labels.

-----

## 6. Test Suite Design

### 6.1 Test Architecture

The 10 tests cover four categories:

1. **Endpoint structure tests:** Root endpoint returns expected keys and service info
1. **Health check tests:** /health returns 200 with model loaded; 503 without model
1. **Prediction contract tests:** Valid input returns expected response structure; fraudulent input is correctly classified; negative Amount is rejected with 422; missing field is rejected with 422; confidence levels match probability ranges
1. **Observability tests:** /metrics returns Prometheus-formatted output containing expected metric names after a prediction

All tests mock the XGBoost model – they do not load the real artifact. This keeps the CI test run under 10 seconds and tests API behaviour independently of model correctness. Model correctness is validated in the Project 06 pipeline; the API tests validate that the serving code handles inputs and errors correctly regardless of which model is loaded.

### 6.2 The Lifespan Fixture Problem

A non-trivial testing challenge arose from the interaction between FastAPI’s lifespan handler and pytest fixtures. The 503 test cases (health check and prediction when model is None) require the API to start without a loaded model. The naive approach – setting model = None before starting TestClient – fails because the lifespan handler fires on TestClient startup and loads the real model from the models/ directory.

The fix is to set model = None after TestClient startup:

```python
@pytest.fixture
def no_model_client():
from src.main import app
with TestClient(app, raise_server_exceptions=False) as c:
# Set model to None AFTER startup -- lifespan fires during __enter__
original_model = main_module.model
main_module.model = None
yield c
main_module.model = original_model # restore after test
```

This pattern – modifying module-level state after context manager entry – is a standard approach for testing FastAPI lifespan-dependent behaviour and is not documented in the FastAPI testing guide.

-----

## 7. Discussion

### 7.1 Registry-First Loading in Practice

The Registry-first loading strategy achieved its design goal: the startup log confirms the model source in production:

```
Model ready. Source: mlflow_registry:fraud-xgboost-class-weighted/Production
```

The local artifact fallback was exercised during development (before the DagsHub token was configured) and confirmed to work correctly. In a real production system, the fallback would serve the last known-good model version while the Registry connectivity issue is resolved, providing graceful degradation rather than a hard failure.

### 7.2 The Silent Home Directory Failure Mode

The PermissionError from the missing home directory is worth documenting in detail because it exemplifies a class of production failure modes that are difficult to diagnose: the system starts, the health check passes, but the system is not operating as designed.

In this case:

- DagsHub auth call returned HTTP 200 (token valid)
- Token cache write raised PermissionError
- Exception was caught by the Registry load exception handler
- System logged “Registry load failed” and fell back to local artifacts
- Health check returned 200 (model loaded from local artifacts)
- model_source in API response read “local:models/…” instead of “mlflow_registry:…”

Without examining the model_source field in the API response, the failure was not visible. This motivates including model_source in the health check response and in structured logs – not just in the predict response where it might only be examined during debugging.

### 7.3 Limitations

**Single-replica deployment.** The EC2 deployment runs one container instance. The Kubernetes HPA is configured for 2-5 replicas, but this configuration has not been exercised at scale. Actual autoscaling behaviour under load has not been characterised.

**No temporal model performance validation in CI.** The CI/CD pipeline runs API contract tests but does not validate that the model’s Recall@P90 remains at 0.837. A performance regression in the model (caused by, for example, a change to the scaler or feature column order) would pass the test suite and deploy successfully.

**Default scaler statistics.** As documented in the Project 06 technical report, the containerised deployment uses default scaler statistics for Time and Amount when local scaler artifacts are not available. This introduces a modest approximation that has not been quantified in terms of its effect on Recall@P90.

**Synchronous Prometheus logging.** Metric recording in the predict endpoint is synchronous. At high request rates, this introduces latency that is not inherent to the model inference. The observed first-call latency of ~88ms includes model loading; steady-state inference latency is expected to be 2-5ms for this model size, but this has not been characterised under load.

-----

## 8. Future Work and Research Directions

**Model-infrastructure versioning traceability.** A specific prediction should be traceable to both the model version (MLflow Registry version and run ID) and the infrastructure version (Docker image SHA) that served it. The current system exposes model_source in the predict response but does not log the image SHA. Structured logging that captures both dimensions per request would enable complete traceability and support root cause analysis when prediction quality issues emerge.

**Performance gating in CI/CD.** The current pipeline gates on API contract tests. Adding a performance gate – a step that loads the model and runs it against a held-out test set, blocking deployment if Recall@P90 drops below a threshold – would prevent model performance regressions from reaching production. The threshold (0.01 degradation? 0.05?) is an operational decision, but the mechanism is a straightforward extension of the existing test job.

**Latency characterisation under load.** The observed latency at low traffic is ~88ms (first call) and expected ~2-5ms (warm inference). The p95 and p99 latency under concurrent load (50, 200, 500 simultaneous requests) has not been measured. For a fraud detection system where the payment authorisation window may impose a latency SLA, characterising the latency distribution under load is a prerequisite for production readiness. This is a direct input to Project 08’s multi-model architecture, where latency budgets must be coordinated across fraud detection, credit scoring, and monitoring components.

**Hot model reload.** The current system requires a container restart to load a new model version from the Registry. A production system should support hot reload: a background thread polls the Registry for new Production versions and swaps the model reference atomically without dropping in-flight requests. This is a non-trivial concurrency problem (the model reference must be swapped atomically while the endpoint is actively serving) but is the difference between a 0-downtime model update and a 30-second availability gap on every model promotion.

**Alibi Detect integration.** The fraud_probability_distribution histogram provides a lightweight proxy for drift detection. Integrating Alibi Detect (Klaise et al., 2021) for statistical covariate shift detection – using the V1-V28 feature distributions as the monitored input – would provide a more principled early-warning system. The monitoring endpoint could expose drift test statistics (KS p-values, MMD scores) alongside the probability distribution, enabling automated alerting on distribution shift before Recall@P90 has demonstrably degraded.

-----

## 9. Conclusion

This project produced a production-grade inference system for the Project 06 fraud detection model:

- FastAPI REST API – /predict, /health, /metrics, Pydantic validation, confidence levels
- Multi-stage Docker build – non-root user with home directory, HEALTHCHECK, reproducible image
- GitHub Actions CI/CD – 10 API contract tests, ECR push on green main
- AWS ECR – dual-tagged (latest + commit SHA), immutable and traceable
- AWS EC2 – Elastic IP 108.128.140.230, eu-west-1
- MLflow Registry integration – Registry-first loading with local artifact fallback
- Prometheus instrumentation – prediction counter, latency histogram, probability distribution
- Kubernetes manifests – 6 resources validated, ready for EKS

The gap between “I deployed a model” and “I built a system that serves it reliably” is exactly what this project closes. Project 06 answered whether the model works. Project 07 answers whether it can be trusted in production.

Project 08 (FRIP – Financial Risk Intelligence Platform) takes this further: fraud detection, credit scoring, and real-time monitoring unified into a single multi-model platform, where the versioning, performance gating, and latency questions raised here all compound across components.

-----

## References

Breck, E., Cai, S., Nielsen, E., Salib, M., and Sculley, D. (2017). The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. IEEE International Conference on Big Data, 1123-1132.

Gama, J., Zliobaite, I., Bifet, A., Pechenizkiy, M., and Bouchachia, A. (2014). A Survey on Concept Drift Adaptation. ACM Computing Surveys, 46(4), 1-37.

Klaise, J., Van Looveren, A., Vacanti, G., and Coca, A. (2021). Alibi Detect: Algorithms for Outlier, Adversarial and Drift Detection. Journal of Machine Learning Research, 22(147), 1-11.

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems. Advances in Neural Information Processing Systems, 28.


Muhammed Keita
ML Systems Engineer · MLOps · FinTech ML

GitHub: github.com/muhammed-keita-ml
LinkedIn: linkedin.com/in/muhammed-keita
Live API: project-05-heart-disease-api-production.up.railway.app/docs
Portfolio: read.cv/muhammed-keita
Medium: medium.com/@mkeitaone

GCP ML Engineer Certified · Machine Learning in Production (DeepLearning.AI)
