# Cloud-Native ML Deployment: Architecture, Reproducibility Constraints, and Production Infrastructure for a Fraud Detection Inference Service

**Author:** Muhammed Keita  
**Date:** 2026  
**Repository:** https://github.com/muhammed-keita-ml/project-07-fraud-detection-k8s  
**Live System:** http://108.128.140.230:8000/docs  
**Model Registry:** https://dagshub.com/muhammed-keita-ml/project-06-fraud-detection-pipeline

---

## Abstract

Deploying trained machine learning models introduces engineering problems that model training pipelines alone do not address: model versioning and registry integration, containerisation, orchestration, observability, and CI/CD automation. This report documents the design and implementation of a fraud-detection inference service built around the model developed in Project 06 (XGBoost with class weighting, PR-AUC 0.880, Recall@P90 0.837).

The system serves predictions through a FastAPI REST API, attempts to load the Production-stage model from the MLflow Model Registry on DagsHub at startup, supports local artifact fallback, builds and pushes Docker images to AWS ECR through GitHub Actions, and runs on AWS EC2. Prometheus instrumentation exposes prediction counts, latency measurements, and fraud-probability distributions. Six Kubernetes manifests describe a potential cluster deployment; although their configuration was validated, runtime behaviour on Minikube or Amazon EKS has not been verified.

A central architectural decision is to separate model versioning from infrastructure versioning by retrieving model artifacts from a registry rather than requiring every model update to be baked into a new image. The implementation also exposes important limitations involving artifact consistency, image provenance, deployment validation, model-performance gating, and serving latency. These limitations inform the future development of Project 08 (FRIP), a proposed multi-model financial-risk platform.

## 1. Introduction

### 1.1 Problem Context

The gap between a working ML model and a reliable deployed ML system is larger than model training alone suggests. A trained model does not, by itself, provide a mechanism for serving predictions reliably under concurrent load, recovering from container failures, loading the intended model version, or exposing sufficient operational information to investigate degraded performance.

Project 06 produced a fraud-detection model and evaluated its performance. Project 07 addresses the engineering problem of serving that model through an API and supporting it with containerisation, deployment automation, registry integration, and observability.

These components introduce trade-offs that are not visible at the model-training layer. A successful API response demonstrates that a request can be processed; it does not, on its own, demonstrate model accuracy, resilience under failure, scalability, or production readiness.

### 1.2 Engineering and Research Questions Addressed

This project addresses three interrelated questions:

1. **Model-infrastructure versioning:** How should model versioning through the MLflow Registry and infrastructure versioning through Docker images and Git commits be coordinated when both can change independently?
2. **Reproducibility under environment constraints:** How can CI/CD be designed when the development environment cannot run Docker locally because hardware virtualisation is unavailable, and what guarantees does a cloud-based build provide?
3. **Serving observability:** What instrumentation is needed to understand the operational behaviour of a fraud-detection API, and which questions remain unanswered by experiment tracking alone?

### 1.3 Contributions

- A registry-first model-loading strategy that attempts to decouple model versioning from infrastructure versioning, with a local-artifact fallback.
- A multi-stage Dockerfile using a non-root user and a home directory, addressing a deployment failure in which DagsHub token-cache writes raised a `PermissionError`.
- A GitHub Actions CI/CD pipeline designed to run tests and build and push Docker images to AWS ECR when the relevant workflow conditions are satisfied.
- A pytest suite with **12 passing tests** covering API behaviour, input validation, error handling, and Prometheus metrics. These tests do not establish model accuracy or production performance.
- Six Kubernetes manifests covering Deployment, Service, HPA, ConfigMap, Secret, and Ingress resources. Their YAML/configuration was validated, but they have not been runtime-tested on Minikube or EKS.
- Prometheus instrumentation exposing prediction counts, latency measurements, and fraud-probability distributions. The probability distribution is a potential monitoring signal, not a validated drift detector.
- A deployed inference API on AWS EC2 in the `eu-west-1` region, verified through basic health and prediction smoke tests.

## 2. Related Work

**Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., et al. (2015).** *Hidden Technical Debt in Machine Learning Systems.* NeurIPS. This work describes the complexity that emerges from dependencies between data processing, model training, and serving. The separation between Projects 06 and 07, with the MLflow Registry serving as an interface between training and inference, reflects the need to manage these components explicitly.

**Breck, E., Cai, S., Nielsen, E., Salib, A., and Sculley, D. (2017).** *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction.* IEEE International Conference on Big Data. This work provides a framework for evaluating ML production readiness. The tests in this project cover selected API and infrastructure behaviours; they do not constitute a complete production-readiness assessment.

**Gama, J., Zliobaite, I., Bifet, A., Pechenizkiy, A., and Bouchachia, A. (2014).** *A Survey on Concept Drift Adaptation.* ACM Computing Surveys. This survey provides context for distinguishing changes in input distributions from changes in the relationship between inputs and outcomes. The probability-distribution metric exposed by this service is a possible monitoring signal, but it does not independently establish concept drift or degraded model performance.

**Klaise, J., Van Looveren, A., Vacanti, G., and Coca, A. (2021).** *Alibi Detect: Algorithms for Outlier, Adversarial and Drift Detection.* Journal of Machine Learning Research. This work describes statistical approaches to outlier and drift detection. The monitoring implemented here is more limited: it exposes probability distributions without implementing the full reference-based statistical testing required for validated drift detection.

## 3. System Architecture

### 3.1 Overview

The system is organised into several components with distinct responsibilities:

```text
MLflow Model Registry (DagsHub)
              |
              v
      FastAPI inference service
              |
              v
       Docker container
              |
              v
          AWS EC2

GitHub Actions ---> AWS ECR

FastAPI metrics endpoint ---> Prometheus instrumentation
```

The MLflow Registry is responsible for model-version management. FastAPI serves predictions. Docker packages the application and its dependencies. GitHub Actions automates testing and image publishing to ECR. The deployed EC2 service provides the current inference runtime.

Prometheus metrics are exposed by the API. The availability and runtime configuration of a complete Prometheus/Grafana monitoring stack should not be inferred solely from the existence of instrumentation or configuration files.

Kubernetes manifests describe a possible orchestration configuration for future testing on a Kubernetes cluster. They do not demonstrate that the application has been successfully deployed to EKS.

### 3.2 Model Loading Strategy

Three approaches were considered.

**Option 1: Bake the model into the container image.** The model is copied into the image at build time. This can make the image self-contained, but changing the model requires rebuilding and redeploying the image.

**Option 2: Mount the model as a volume.** The application loads the model from a mounted filesystem path. This separates the model from the image but requires management of the volume and its contents.

**Option 3: Load the model from a registry at startup.** The application retrieves the intended model artifacts from the MLflow Registry when it starts. This allows model versioning to be managed separately from application image versioning, provided the registry is accessible and the associated artifacts are available and compatible.

The third approach was selected. A local-artifact fallback is also implemented. However, the presence of fallback code alone does not guarantee that the fallback will always be available, compatible, or equivalent to the intended registry version.

### 3.3 Registry Loading Implementation

The registry-loading implementation uses MLflow's tracking client to identify the registered model version and retrieve its artifacts. The current implementation also associates the model with its preprocessing artifacts, including the scaler and feature-column information, through the relevant MLflow run.

The intended sequence is:

1. Identify the Production-stage model version.
2. Obtain the associated MLflow run identifier.
3. Retrieve the model and preprocessing artifacts associated with that run.
4. Load the artifacts into the inference service.
5. Confirm that the required artifacts and expected feature schema are available before serving predictions.

This design addresses the risk of loading a model with preprocessing artifacts from a different experiment or run.

Earlier development encountered an artifact-path resolution problem when using a high-level MLflow XGBoost loading interface. The lower-level artifact download approach and native XGBoost loading were used to address that issue.

The implementation also includes regression tests for loading artifacts from the same run and preventing a partial model load when a required registry artifact is missing.

### 3.4 Container Design

The Dockerfile uses a multi-stage build.

**Builder stage:** Installs the Python dependencies into a separate prefix directory. Build-time dependencies are kept separate from the runtime image where the build configuration permits.

**Runtime stage:** Copies the installed dependencies and application into the runtime image, reducing the need to include build tooling in the deployed environment.

The container creates a non-root user and a home directory:

```dockerfile
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser
```

The home directory is relevant because DagsHub authentication can write token-cache files under the user's home directory. During development, the absence of a suitable home directory caused a `PermissionError`. Depending on the exception handling and the availability of local artifacts, this failure could lead to fallback behaviour or startup failure.

This incident demonstrates why container identity, filesystem permissions, credential handling, and error reporting must be tested together rather than treated as independent configuration details.

### 3.5 Kubernetes Manifests

Six Kubernetes manifests were created for the following resources:

| Resource | Intended purpose |
|---|---|
| Deployment | Defines application replicas, probes, and resource settings |
| Service | Exposes the application within the cluster or through the configured service type |
| HPA | Specifies autoscaling targets and replica bounds |
| ConfigMap | Stores non-sensitive application configuration |
| Secret | Provides a location for deployment-time credentials |
| Ingress | Describes HTTP routing through an ingress controller |

The configuration includes a Deployment with two replicas, health probes, and resource limits, as well as an HPA configured for a replica range of two to five. The Service, ConfigMap, Secret, and Ingress provide the corresponding networking and configuration structures.

These manifests were validated as YAML/configuration. They have **not** been runtime-tested on Minikube or EKS. Therefore, successful scheduling, secret injection, ingress routing, probe behaviour, autoscaling, rolling updates, and application startup in Kubernetes remain unverified.

The intended use of a minimum of two replicas is to provide the possibility of maintaining service during an individual replica restart. That outcome depends on the deployment strategy, readiness behaviour, available capacity, and successful operation of the remaining replica. The manifest configuration alone does not prove uninterrupted availability.

The ConfigMap/Secret separation follows the principle of keeping non-sensitive configuration separate from credentials. A Kubernetes Secret also requires appropriate access controls and secure deployment practices; its resource type alone does not guarantee that credentials are securely managed.

## 4. CI/CD Pipeline

### 4.1 Design Rationale

Local Docker builds were unavailable during development because hardware virtualisation was disabled or unavailable in the Windows environment. The workflow was therefore designed to perform image builds in GitHub Actions rather than depend on a successful local Docker build.

A cloud-based build provides a more controlled build environment than relying on the developer's machine. However, a successful CI build does not by itself prove that the resulting image has been deployed, that the running service uses that image, or that the service is reliable under production traffic.

### 4.2 Pipeline Structure

The intended workflow is:

```text
Push to main
     |
     v
Test job (GitHub Actions runner)
     |
     +-- Check out source
     +-- Set up Python
     +-- Install dependencies
     +-- Run pytest
     |
     v
Build-and-push job
     |
     +-- Configure AWS credentials
     +-- Authenticate to ECR
     +-- Build Docker image
     +-- Tag image as latest and with commit SHA
     +-- Push image tags to ECR
```

The workflow is configured to run tests and, when the required conditions are met, build and push the Docker image to ECR. It **does not automatically deploy the image to EC2**.

A push to the repository should therefore be followed by checking the GitHub Actions result and confirming that the expected image was published to ECR. EC2 deployment is a separate operation requiring its own verification and rollback plan.

### 4.3 Dual Tagging and Image Provenance

The workflow tags images with `latest` and the Git commit SHA supplied by GitHub Actions. The SHA tag helps identify the source revision associated with a particular build.

However, the source revision of the currently running EC2 image has not been independently verified. The running image digest previously observed differed from the ECR `latest` digest. The image currently running on EC2 must not therefore be assumed to correspond to the latest repository commit.

A reliable provenance chain requires identifying the image digest running on EC2 and matching it to the digest and commit SHA recorded in ECR and the CI workflow. Reproducibility also depends on dependency versions, base-image versions, build inputs, and other relevant environment details.

The current workflow provides a foundation for image traceability, but the complete connection between source revision, built image, and running deployment remains a verification task.

## 5. API Design

### 5.1 Endpoint Structure

The service exposes the following endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Service information and endpoint directory |
| `/health` | GET | Reports application and model-loading status |
| `/metrics` | GET | Exposes Prometheus-format metrics |
| `/predict` | POST | Returns a fraud prediction for a transaction |

A successful request to an endpoint establishes that the endpoint responded under the tested conditions. It does not establish performance under concurrent traffic or predictive accuracy on unseen data.

### 5.2 Input Validation

The input schema expects 30 features. The `Amount` field has a non-negativity constraint:

```python
Amount: float = Field(..., ge=0, description="Transaction amount in USD")
```

A negative amount is treated as invalid input and rejected before inference. This helps prevent malformed requests from being processed as if they were valid transactions.

The schema also provides a defined input contract. Feature ordering, preprocessing compatibility, and the availability of all expected fields remain important when the model and its associated artifacts are loaded.

### 5.3 Response Structure

A previously observed smoke-test response had the following general structure:

```json
{
  "fraud_probability": 0.000001,
  "prediction": "legitimate",
  "confidence": "high",
  "model_source": "mlflow_registry:fraud-xgboost-class-weighted/Production",
  "latency_ms": 87.818
}
```

The values shown are an illustrative example from a smoke test, not a benchmark or a claim about model performance across a dataset.

The `confidence` field is derived from the predicted probability according to configured thresholds. Such a label is an application-defined probability band; it should not be interpreted as a calibrated estimate of the probability that a prediction is correct unless calibration has been evaluated.

The `model_source` field indicates the source reported by the serving application. It supports operational investigation, although complete prediction-level provenance would also require recording the exact registry version, run identifier, and infrastructure image digest.

### 5.4 Prometheus Instrumentation

The API exposes metrics for prediction counts, prediction latency, and fraud-probability distributions. The instrumentation is intended to support operational observation of request volume, latency, and score patterns.

The fraud-probability distribution can help identify changes in the scores produced by the model. For example, a shift toward more intermediate scores could prompt investigation. However, a histogram of output probabilities is not, by itself, a validated drift detector or proof that model performance has degraded.

A reliable performance-monitoring process would require suitable reference data, statistical comparisons, and, when outcomes become available, evaluation against labelled results.

## 6. Test Suite Design

### 6.1 Test Architecture

The current pytest suite has **12 passing tests**. It covers API and serving behaviour, including:

1. Endpoint structure and expected response fields.
2. Health-check behaviour when a model is available or unavailable.
3. Prediction response contracts.
4. Input validation, including invalid or missing fields.
5. Confidence-level behaviour for configured probability ranges.
6. Prometheus metrics output.
7. Registry artifact-loading behaviour and related regression cases.

The tests mock the XGBoost model for API-contract scenarios where appropriate. This isolates application behaviour from the model's predictive performance. The registry regression tests cover important artifact-loading failure cases.

The passing suite is evidence that the tested code paths behaved as expected in the test environment. It does not establish model accuracy, production latency, resilience under live registry outages, Kubernetes runtime behaviour, or load capacity.

### 6.2 The Lifespan Fixture Problem

A testing challenge arose from the interaction between FastAPI's lifespan handler and pytest fixtures. Tests that require the application to behave as if no model is loaded can fail to exercise that state if the application startup handler loads a model when `TestClient` starts.

The relevant fixture pattern is to establish the client context first and then set the module-level model state for the test:

```python

@pytest.fixture
def no_model_client():
    import src.main as main_module
    from src.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        original_state = (
            main_module.model,
            main_module.scaler,
            main_module.feature_columns,
            main_module.model_source,
        )

        try:
            main_module.model = None
            main_module.scaler = None
            main_module.feature_columns = None
            main_module.model_source = None

            yield c
        finally:
            (
                main_module.model,
                main_module.scaler,
                main_module.feature_columns,
                main_module.model_source,
            ) = original_state

```

This pattern allows tests to exercise the intended no-model condition after application startup. The exact fixture implementation should remain consistent with the current test module and its imports.

The general lesson is that tests for application lifecycle behaviour must account for startup and shutdown side effects. Otherwise, a test can appear to cover an error state without actually reaching it.

## 7. Discussion

### 7.1 Registry-First Loading in Practice

The deployed service has returned a health response indicating that a model was loaded from the MLflow Registry. A previous registry check also confirmed that the model and its associated preprocessing artifacts were available and that the expected feature count was 30.

These observations support the claim that registry-backed loading worked under the tested conditions. They do not prove that registry loading will succeed during every startup, that the registry will always be available, or that a new model promotion will be safe without further compatibility checks.

The local-artifact fallback was also exercised during development. The implementation's fallback path is a useful design feature, but its reliability depends on artifact availability, artifact compatibility, and the handling of loading errors. A production-grade last-known-good strategy would require explicit version selection, validation, and tests of registry-failure scenarios.

### 7.2 The Silent Home Directory Failure Mode

The missing-home-directory incident illustrates a class of deployment failures in which an authentication or filesystem problem changes the application's behaviour without necessarily producing an immediate, obvious failure at the service level.

During development, a token-cache write raised a `PermissionError`. Depending on the exception handling and available local artifacts, the application could attempt fallback loading or fail to start. A health check that only reports whether some model is loaded may not distinguish the intended registry version from an unexpected fallback model.

This motivates exposing model-source information and recording clear, structured startup logs. For more complete traceability, the service should report the registry version and run identifier, and the deployment should record the image digest used to start the container.

### 7.3 Limitations

**Single-container EC2 deployment.** The EC2 deployment runs one container instance. The Kubernetes manifests specify multiple replicas, but this configuration has not been exercised at runtime. Availability under container failure and autoscaling behaviour have not been demonstrated.

**No model-performance gate in CI.** The automated tests focus on API and selected loading behaviour. They do not run a held-out evaluation that verifies Recall@P90 or another model-performance metric. A regression in model quality could therefore remain undetected by the current API test suite.

**Preprocessing consistency.** Model inference depends on using the correct preprocessing artifacts and feature ordering. The registry-loading implementation now includes regression tests for loading associated artifacts from the same run. Further validation should verify preprocessing compatibility against the training pipeline and evaluate the effect of any fallback preprocessing behaviour on predictive performance.

**Latency not characterised under load.** A smoke test observed a prediction latency of approximately 88 ms. A single observation is not a reliable estimate of steady-state latency or a percentile benchmark. Latency under concurrency, repeated warm requests, network variation, and different startup conditions has not been characterised.

**Monitoring limitations.** The Prometheus probability histogram describes the distribution of model scores, not actual fraud outcomes. Without suitable reference distributions and labelled outcomes, it cannot establish whether Recall@P90 has degraded.

**Image provenance.** Although the CI workflow produces SHA-tagged images, the exact source commit for the image currently running on EC2 has not been independently established. This limits the strength of any claim that the running deployment corresponds to a particular repository revision.

**Kubernetes runtime not verified.** The Kubernetes configuration has not been exercised on Minikube or EKS. Cluster-level networking, secrets, ingress, health probes, resource constraints, and autoscaling remain to be tested.

## 8. Future Work and Research Directions

**Model-infrastructure versioning traceability.** Each prediction should be traceable to the registry model version and run identifier as well as the container image digest. Structured logs containing both identifiers would support root-cause analysis when prediction behaviour changes.

**Performance gating in CI/CD.** The pipeline could be extended to load the intended model and evaluate it against a fixed, versioned holdout dataset. A deployment gate could compare Recall@P90 and other selected metrics with documented baselines and tolerances. The evaluation dataset, threshold, preprocessing version, and evaluation procedure would need to be fixed and auditable.

**Latency characterisation under load.** Repeated measurements should separate startup latency from warm inference latency. A controlled load test should record throughput, error rate, and p50, p95, and p99 latency at several concurrency levels. The results would help establish whether the service meets a defined latency budget.

**Kubernetes runtime validation.** The manifests should be tested in a real Kubernetes environment. Validation should cover application startup, registry credentials, readiness and liveness probes, service routing, ingress configuration, resource limits, scaling, and rolling updates. Any claims of high availability should be based on observed failure and recovery tests.

**Safer model fallback and updates.** The service could be extended to validate all required artifacts before making a model available for predictions. A robust last-known-good mechanism would need to persist or retrieve a specific compatible model version, reject incomplete artifact sets, and record the reason for fallback. Hot model reload would require careful coordination so that in-flight requests continue to use a consistent model and preprocessing configuration.

**Statistical drift detection.** The current score histogram could be supplemented with reference-based statistical tests for input and score distributions. Detection thresholds should be evaluated for false positives and sensitivity. Where labelled outcomes are available, the system should compare monitoring alerts with actual changes in metrics such as Recall@P90.

**Multi-model financial-risk platform.** Project 08 (FRIP) is intended to explore the integration of fraud detection, credit scoring, and monitoring. The engineering questions raised in this project—model and infrastructure versioning, evaluation gates, monitoring, and latency budgets—will become more complex when multiple models and services interact.

## 9. Conclusion

This project produced a deployed fraud-detection inference service and supporting deployment infrastructure. The API has passed automated tests and basic live smoke tests. These results demonstrate that the tested application paths can operate under the conditions evaluated; they do not establish end-to-end production reliability.

The project includes:

- A FastAPI inference API with `/predict`, `/health`, and `/metrics` endpoints and Pydantic input validation.
- A multi-stage Docker build using a non-root user and a configured health check.
- A GitHub Actions workflow that tests the application and builds and pushes images to AWS ECR under the configured conditions.
- ECR image tagging using `latest` and the Git commit SHA.
- An AWS EC2 deployment in `eu-west-1` with a stable public endpoint.
- MLflow Registry integration with registry-first loading and local-artifact fallback.
- Prometheus instrumentation for prediction counts, latency, and fraud-probability distributions.
- Six Kubernetes manifests prepared for further validation on a Kubernetes cluster.

Several important areas remain unverified: Kubernetes runtime behaviour, high-availability guarantees, autoscaling, load capacity, model-performance gating, and the exact source revision associated with the running EC2 image. These are substantive engineering limitations, not merely documentation details.

Project 06 evaluated the fraud-detection model. Project 07 demonstrates how the model can be served and monitored, while identifying the additional validation needed to establish production reliability.

Project 08 (FRIP — Financial Risk Intelligence Platform) is intended to build on these findings by exploring a unified platform for fraud detection, credit scoring, and monitoring, with explicit attention to model versioning, evaluation, observability, and operational constraints.

## References

Breck, E., Cai, S., Nielsen, E., Salib, A., and Sculley, D. (2017). *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction.* IEEE International Conference on Big Data, 1123–1132.

Gama, J., Zliobaite, I., Bifet, A., Pechenizkiy, A., and Bouchachia, A. (2014). *A Survey on Concept Drift Adaptation.* ACM Computing Surveys, 46(4), 1–37.

Klaise, J., Van Looveren, A., Vacanti, G., and Coca, A. (2021). *Alibi Detect: Algorithms for Outlier, Adversarial and Drift Detection.* Journal of Machine Learning Research, 22(147), 1–11.

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, D., Ebner, D., et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* Advances in Neural Information Processing Systems, 28.

---

**Muhammed Keita**  
ML Systems Engineer · MLOps · FinTech ML

GitHub: https://github.com/muhammed-keita-ml  
LinkedIn: https://linkedin.com/in/muhammed-keita  
Live API: http://108.128.140.230:8000/docs  
Medium: https://medium.com/@mkeitaone

Credentials and training should be listed separately in the portfolio and kept consistent with the certifications actually earned.
