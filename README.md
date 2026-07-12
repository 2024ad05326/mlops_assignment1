# Heart Disease Risk Prediction API
## MLOps End-to-End Assignment

A reproducible, cloud-ready binary classifier for predicting heart disease risk from the UCI Heart Disease dataset. This repository implements a complete MLOps pipeline: data acquisition, EDA, model training with experiment tracking (MLflow), containerization (Docker), CI/CD (GitHub Actions), Kubernetes deployment, and monitoring (Prometheus metrics + application logging).

**Video Recording**: [mlops assignment.mov](mlops%20assignment.mov)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Setup & Installation](#setup--installation)
4. [Dataset & EDA](#dataset--eda)
5. [Model Development](#model-development)
6. [Experiment Tracking](#experiment-tracking)
7. [API & Model Packaging](#api--model-packaging)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Containerization](#containerization)
10. [Kubernetes Deployment](#kubernetes-deployment)
11. [Monitoring & Logging](#monitoring--logging)
12. [Testing](#testing)
13. [Repository Structure](#repository-structure)
14. [Screenshots](#screenshots)

---

## Project Overview

**Goal**: Predict the risk of heart disease based on patient health data.

**Dataset**: UCI Heart Disease Dataset (ID 45) fetched automatically via `ucimlrepo`.

**Tech Stack**:
- Python 3.10+
- Pandas / NumPy
- Scikit-learn + XGBoost
- MLflow (experiment tracking)
- FastAPI (model serving)
- Docker (containerization)
- Kubernetes (orchestration)
- GitHub Actions (CI/CD)
- Prometheus client (metrics)

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│   GitHub    │────▶│  GitHub      │────▶│  Kubernetes   │
│   Repo      │     │  Actions CI  │     │  Cluster      │
└─────────────┘     └──────────────┘     └───────┬───────┘
                                                  │
                    ┌──────────────────────────────┘
                    ▼
            ┌───────────────┐     ┌───────────────┐
            │  Docker       │────▶│  FastAPI      │
            │  Image        │     │  /predict     │
            └───────────────┘     └───────┬───────┘
                                            │
                    ┌───────────────────────┘
                    ▼
            ┌───────────────┐     ┌───────────────┐
            │  Prometheus   │◀────│ /metrics      │
            │  + Grafana    │     └───────────────┘
            └───────────────┘
```

**Data Flow**:
1. `src/download_data.py` fetches raw CSV from UCI.
2. `src/eda.py` generates histograms, correlation heatmaps, and class distribution plots.
3. `src/preprocessing.py` builds a reusable `ColumnTransformer` + `Pipeline` for numeric/categorical features.
4. `src/train.py` trains three models (Logistic Regression, Random Forest, XGBoost) with `GridSearchCV`, logs everything to MLflow, and persists the best pipeline as `models/best_model_pipeline.pkl`.
5. `src/app.py` loads the pipeline and exposes REST endpoints (`/health`, `/predict`, `/metrics`).
6. GitHub Actions runs lint, tests, data download, EDA, training, and Docker build on every push/PR.
7. Docker image is built from `deployment/Dockerfile`.
8. Kubernetes manifests in `deployment/` roll out 2 replicas behind a LoadBalancer/NodePort.
9. Prometheus scrapes `/metrics` and Grafana visualizes request rate, latency, and confidence.

---

## Setup & Installation

### Prerequisites

- Python >= 3.10
- `uv` (Python package manager)
- Docker Desktop (with Kubernetes enabled)
- `kubectl` configured

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/mlops-assignment1.git
cd mlops-assignment1

# 2. Install dependencies
uv sync --all-groups

# 3. Download the dataset
uv run src/download_data.py

# 4. Run EDA (generates plots in data/plots/)
uv run src/eda.py

# 5. Train models (logs to MLflow, saves best pipeline)
uv run src/train.py

# 6. Run tests
uv run pytest
```

### Running the API Locally

```bash
uv run uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Test with curl:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52.0,"sex":1.0,"cp":3.0,"trestbps":172.0,"chol":199.0,"fbs":1.0,"restecg":0.0,"thalach":162.0,"exang":0.0,"oldpeak":0.5,"slope":1.0,"ca":0.0,"thal":7.0}'
```

---

## Dataset & EDA

**Source**: UCI Machine Learning Repository, Heart Disease dataset (ID 45).

**Columns**: 14 features + target:
- Numeric: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`
- Categorical: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`
- Target: `num` (0 = no disease, 1-4 = disease) → binarized as `target` (0/1)

**EDA outputs** (saved under `data/plots/`):
1. Missing Value Heatmap
2. Class Distribution (balanced-ish, ~54% presence)
3. Continuous Feature Histograms (age, trestbps, chol, thalach, oldpeak)
4. Full Correlation Heatmap

Sample correlation insight: `cp` (chest pain type) and `thalach` (max heart rate) show strong negative correlation with disease presence.

---

## Model Development

### Preprocessing

- **Numeric**: `SimpleImputer(median)` → `StandardScaler`
- **Categorical**: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')`
- Combined in a `ColumnTransformer` wrapped inside a `Pipeline`

### Models & Hyperparameter Search

| Model | Algorithm | Hyperparameters Tuned | CV F1 (mean ± std) | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | sklearn `LogisticRegression` | C, penalty, solver | 0.8363 ± 0.0127 | 0.8621 | 0.9632 |
| Random Forest | sklearn `RandomForestClassifier` | n_estimators, max_depth, min_samples_split | 0.8000 ± 0.0263 | 0.8621 | 0.9535 |
| XGBoost | `XGBClassifier` | n_estimators, max_depth, learning_rate | 0.8017 ± 0.0337 | 0.8772 | 0.9491 |

**Best Model**: XGBoost (test F1 = 0.8772) → saved to `models/best_model_pipeline.pkl`.

**Cross-validation**: 5-fold Stratified K-Fold, scoring = F1.

---

## Experiment Tracking

MLflow experiment: `Heart_Disease_MLOps`.

For each run we log:
- **Parameters**: model type, test_size, random_state, cv_folds, best hyperparameters
- **Metrics**: accuracy, precision, recall, f1_score, roc_auc, cv_f1_mean, cv_f1_std
- **Artifacts**: confusion matrix PNG, ROC curve PNG

Start local MLflow UI:
```bash
mlflow ui
# Open http://localhost:5000
```

---

## API & Model Packaging

**Framework**: FastAPI.

**Endpoints**:
- `GET /health` → `{"status": "ok", "model_loaded": true}`
- `POST /predict` → `{"heart_disease_present": true/false, "confidence": float}`
- `GET /metrics` → Prometheus text exposition format

**Model format**: `joblib` + `sklearn Pipeline`. Preprocessing steps are embedded in the same pipeline, guaranteeing reproducible inference.

---

## CI/CD Pipeline

**Tool**: GitHub Actions (`.github/workflows/ci-cd.yml`).

**Triggers**: push / pull_request on `main`.

**Steps**:
1. Checkout code
2. Set up `uv`
3. Sync dependencies
4. Lint (`ruff check src/ tests/`)
5. Download data
6. Run EDA
7. Train & track with MLflow
8. Run pytest
9. Build Docker image (`heart-disease-api:latest`)

Pipeline fails on any step error.

---

## Containerization

**Dockerfile** (at `deployment/Dockerfile`):
```dockerfile
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models/ ./models/
COPY data/raw/heart_disease_raw.csv ./data/raw/heart_disease_raw.csv
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build & run**:
```bash
docker build -t heart-disease-api:latest -f deployment/Dockerfile .
docker run -d -p 9000:8000 heart-disease-api:latest
```

---

## Kubernetes Deployment

**Manifests**:
- `deployment/deployment.yaml` (2 replicas, LoadBalancer Service on port 80)
- `deployment/service.yaml` (NodePort on 30007 for local Kubernetes)
- `deployment/prometheus.yaml` (Prometheus deployment + Service on NodePort 30090)

**Deploy**:
```bash
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
kubectl apply -f deployment/prometheus.yaml

kubectl get pods
kubectl get svc
```

Access API: `http://localhost/` (LoadBalancer) or `http://localhost:30007/` (NodePort).

---

## Monitoring & Logging

### Application Logging
- Structured log messages for every prediction (client IP, prediction, confidence).
- Errors are logged with stack traces.

### Prometheus Metrics
Exposed at `GET /metrics`:
- `heart_disease_predictions_total` (status, prediction labels)
- `heart_disease_prediction_latency_seconds` (histogram)
- `heart_disease_prediction_confidence` (gauge)

### Grafana (optional)
1. Add Prometheus data source (`http://prometheus-service:9090`).
2. Import dashboards or create panels for:
   - Request rate per endpoint
   - p95/p99 latency
   - Prediction class distribution
   - Confidence score trends

---

## Testing

**Framework**: pytest.

**Test files**:
- `tests/test_app.py` → API endpoints (health, predict, invalid input, batch consistency)
- `tests/test_data_processing.py` → raw data existence, columns, preprocessing transforms, model serialization
- `tests/test_monitoring.py` → prometheus client registration, counter increments

Run:
```bash
uv run pytest tests/ -v
```

---

## Repository Structure

```
mlops-assignment1/
├── src/
│   ├── app.py              # FastAPI app with /health, /predict, /metrics
│   ├── train.py            # ML training + MLflow logging
│   ├── preprocessing.py    # Reusable preprocessing pipeline
│   ├── eda.py              # EDA visualizations
│   ├── monitoring.py       # Prometheus metrics + logging setup
│   └── download_data.py    # Download UCI dataset
├── tests/
│   ├── test_app.py
│   ├── test_data_processing.py
│   └── test_monitoring.py
├── deployment/
│   ├── Dockerfile
│   ├── deployment.yaml
│   ├── service.yaml
│   └── prometheus.yaml
├── data/
│   ├── raw/heart_disease_raw.csv
│   └── plots/...
├── models/
│   ├── best_model_pipeline.pkl
│   └── preprocessor.pkl
├── pyproject.toml
├── requirements.txt
├── screenshots/
│   ├── eda/
│   ├── mlflow/
│   ├── ci-cd/
│   ├── kubernetes/
│   └── monitoring/
├── README.md
└── report.md
```

---

## Screenshots

Place your assignment screenshots in the `screenshots/` directory.

| Directory | Contents |
|---|---|
| `screenshots/eda/` | Missing values heatmap, class distribution, histograms, correlation matrix |
| `screenshots/mlflow/` | MLflow UI showing experiment runs, parameters, metrics, artifact plots |
| `screenshots/ci-cd/` | GitHub Actions workflow run with all steps passing |
| `screenshots/kubernetes/` | `kubectl get pods` showing running pods, service routing test |
| `screenshots/monitoring/` | Grafana dashboard with latency/request rate, Prometheus query results |

---

## License

Academic assignment use only.
