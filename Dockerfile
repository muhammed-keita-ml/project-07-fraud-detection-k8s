# ============================================================
# Project 07 — Fraud Detection API
# Multi-stage Docker build
#
# Stage 1: builder  — install dependencies into a clean venv
# Stage 2: runtime  — minimal image, non-root user, health check
# ============================================================

# --- Stage 1: builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# --- Stage 2: runtime ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Non-root user — security best practice for production containers
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Expose the FastAPI port
EXPOSE 8000

# Health check — Kubernetes also uses /health endpoint for probes,
# but Docker's own HEALTHCHECK catches issues before K8s is involved
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the FastAPI server
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
