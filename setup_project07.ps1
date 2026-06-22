# =============================================================
# Project 07 - Fraud Detection: Cloud Deployment + Kubernetes
# Local repo skeleton setup script
#
# Run from your portfolio folder:
#   cd C:\Users\Administrator\PORTFOLIO\ml-mlops-portfolio\project-07-fraud-detection-k8s
#   powershell -ExecutionPolicy Bypass -File .\setup_project07.ps1
# =============================================================

$Root = Get-Location

Write-Host "Creating project structure at $Root ..."

$dirs = @(
    "src",
    "k8s",
    "tests",
    "monitoring",
    ".github\workflows",
    "models",
    "scripts"
)

foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null
}

function Write-FileContent {
    param([string]$Path, [string]$Content)
    [System.IO.File]::WriteAllText($Path, $Content)
}

# --- .gitkeep placeholders ---
Write-FileContent (Join-Path $Root "models\.gitkeep") ""

# --- src/__init__.py ---
New-Item -ItemType File -Force -Path (Join-Path $Root "src\__init__.py") | Out-Null
New-Item -ItemType File -Force -Path (Join-Path $Root "tests\__init__.py") | Out-Null

# --- .env.example ---
Write-FileContent (Join-Path $Root ".env.example") @'
DAGSHUB_TOKEN=your_token_here
DAGSHUB_USERNAME=muhammed-keita-ml
DAGSHUB_REPO=project-06-fraud-detection-pipeline
MLFLOW_MODEL_NAME=fraud-xgboost-class-weighted
MLFLOW_MODEL_STAGE=Production
ECR_REGISTRY=597819998212.dkr.ecr.eu-west-1.amazonaws.com
ECR_REPOSITORY=fraud-detection-api
AWS_REGION=eu-west-1
'@

# --- .gitignore ---
Write-FileContent (Join-Path $Root ".gitignore") @'
venv/
__pycache__/
*.pyc
.env
models/*.json
models/*.joblib
models/*.pkl
.pytest_cache/
*.egg-info/
.ipynb_checkpoints/
'@

# --- requirements.txt ---
Write-FileContent (Join-Path $Root "requirements.txt") @'
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.8.2
xgboost==3.2.0
scikit-learn==1.9.0
joblib==1.5.3
pandas==2.3.3
numpy==2.4.6
mlflow==2.22.5
dagshub==0.7.0
python-dotenv==1.2.2
prometheus-client==0.21.0
pytest==9.1.0
httpx==0.27.0
'@

Write-Host ""
Write-Host "Done. Next: copy this script output and run git init steps."
