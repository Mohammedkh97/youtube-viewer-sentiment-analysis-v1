<h1 align="center">
  <br>
  🎬 YouTube Viewer Sentiment Analysis
  <br>
</h1>

<p align="center">
  A production-grade, end-to-end MLOps pipeline for classifying YouTube comment sentiment — powered by LightGBM, FastAPI, MLflow, DVC, and a React Chrome Extension.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/FastAPI-0.136-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/LightGBM-4.6-02B875?style=for-the-badge&logo=lightgbm&logoColor=white" alt="LightGBM"/>
  <img src="https://img.shields.io/badge/MLflow-3.11-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow"/>
  <img src="https://img.shields.io/badge/DVC-S3-13ADC7?style=for-the-badge&logo=dvc&logoColor=white" alt="DVC"/>
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React"/>
  <img src="https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=github-actions&logoColor=white" alt="GitHub Actions"/>
</p>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [ML Pipeline (DVC Stages)](#-ml-pipeline-dvc-stages)
- [API Reference](#-api-reference)
- [Chrome Extension](#-chrome-extension)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration)
- [CI/CD Pipeline](#-cicd-pipeline)
- [License](#-license)

---

## 🚀 Project Overview

**YouTube Viewer Sentiment Analysis** is a full production MLOps system that predicts the sentiment of YouTube comments in real time. The system ingests raw social media text, runs it through a carefully tuned ML pipeline, serves predictions via a versioned REST API, and surfaces results directly in the browser via a Chrome Extension.

The project is built with a strong focus on **reproducibility**, **scalability**, and **observability**:

- **Reproducibility**: Every pipeline stage (data ingestion → preprocessing → training → evaluation → registration) is tracked and versioned using **DVC** with an **AWS S3** remote backend.
- **Observability**: All experiments, hyperparameters, metrics, and artifacts (including confusion matrix visualizations) are tracked in **MLflow**, hosted at a persistent remote dashboard.
- **Scalability**: The inference backend is containerized with **Docker** and served through **FastAPI + Uvicorn**, ready to be horizontally scaled.
- **Accessibility**: A **React-based Chrome Extension** (built with Vite + Tailwind CSS + Recharts) connects directly to the backend, fetching YouTube comments via the YouTube Data API and displaying sentiment dashboards in real time.

**Sentiment Labels:**

| Label | Meaning |
|-------|---------|
| `positive` | The comment expresses a positive opinion |
| `neutral` | The comment is neutral or factual |
| `negative` | The comment expresses a negative opinion |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CHROME EXTENSION                           │
│  React 19 + Vite + Tailwind CSS + Recharts + YouTube Data API  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP POST /api/v1/predict/batch
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND                              │
│    Uvicorn · Pydantic v2 · Joblib · PredictionService           │
│    /api/v1/predict      (single comment inference)              │
│    /api/v1/predict/batch (batch comment inference)              │
│    /api/v1/health        (liveness check)                       │
└───────────┬───────────────────────────────────┬─────────────────┘
            │                                   │
            ▼                                   ▼
┌───────────────────────┐           ┌───────────────────────────┐
│    ARTIFACTS (joblib) │           │     MLFLOW DASHBOARD      │
│  · model.joblib       │           │  mlflow-dashboard.duckdns │
│  · preprocessor.joblib│           │  · Experiments            │
│  · label_encoder.jobl │           │  · Metrics & Params       │
└───────────────────────┘           │  · Confusion Matrix Plots │
            ▲                       └───────────────────────────┘
            │
┌───────────────────────────────────────────────────────────────┐
│                    DVC ML PIPELINE                            │
│                                                               │
│  [Stage 1] data_ingestion                                     │
│      └─> Downloads Reddit sentiment CSV (GitHub raw)         │
│          Maps labels: {-1→2, 0→0, 1→1}                       │
│          Splits into train/test (80/20 stratified)           │
│                                                               │
│  [Stage 2] data_preprocessing                                 │
│      └─> Text cleaning, null handling, deduplication         │
│                                                               │
│  [Stage 3] model_building                                     │
│      └─> TF-IDF vectorization (n-grams 1–3, 10k features)   │
│          Imbalance handling (SMOTE / ADASYN / SMOTEENN)      │
│          Hyperparameter tuning (Bayesian / Random / Grid)    │
│          Trains LightGBM / XGBoost / Random Forest           │
│                                                               │
│  [Stage 4] model_evaluation                                   │
│      └─> Accuracy, F1-macro, Confusion Matrix → MLflow       │
│                                                               │
│  [Stage 5] model_registration                                 │
│      └─> Registers best model in MLflow Model Registry       │
└───────────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────┐
│                    DVC REMOTE STORAGE                         │
│                       AWS S3 Bucket                           │
│     (data/raw, data/processed, artifacts/)                    │
└───────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🧠 Machine Learning Pipeline
- **5-stage DVC pipeline** with full dependency tracking and reproducibility
- **Three classifier options**: LightGBM, XGBoost, Random Forest (selectable via `params.yaml`)
- **Three hyperparameter search strategies**:
  - **Bayesian Optimization** (Optuna TPE Sampler — default)
  - **Random Search** (Scikit-learn `RandomizedSearchCV`)
  - **Grid Search** (Scikit-learn `GridSearchCV`)
- **Cross-validation strategies**: Stratified K-Fold, K-Fold (configurable splits)
- **Four class-imbalance handling methods**:
  - SMOTE (Synthetic Minority Oversampling)
  - ADASYN (Adaptive Synthetic Sampling)
  - Random Undersampling
  - SMOTEENN (Combined SMOTE + Edited Nearest Neighbours)
  - Class Weights (via `class_weight="balanced"`)
- **TF-IDF Vectorizer** with configurable n-gram ranges and max features

### 📊 Experiment Tracking & Model Registry
- Full **MLflow integration** with a remote tracking server
- Per-run logging of: hyperparameters, accuracy, F1-macro score
- **Confusion matrix heatmaps** auto-generated with Seaborn and logged as MLflow artifacts
- Automatic **model registration** in the MLflow Model Registry after evaluation

### 🌐 FastAPI Inference Backend
- **Versioned REST API** under `/api/v1`
- **Single-comment inference** (`POST /api/v1/predict/`)
- **Batch inference** (`POST /api/v1/predict/batch`) for processing multiple comments at once
- **Liveness health check** endpoint (`GET /api/v1/health`)
- Singleton **PredictionService** with startup artifact loading via FastAPI `lifespan`
- Pydantic v2 request/response schema validation
- Production-grade **Uvicorn** ASGI server

### 🧩 Chrome Extension (React)
- **React 19** frontend built with **Vite** and **@crxjs/vite-plugin**
- Styled with **Tailwind CSS v4**
- **Recharts** for sentiment distribution visualizations (pie/bar charts)
- Fetches comments from YouTube videos using the **YouTube Data API v3**
- Sends comments to the FastAPI backend for real-time batch sentiment analysis
- Displays per-comment sentiment labels and confidence scores

### 🔄 Data Version Control (DVC)
- Entire data and artifact lifecycle managed by **DVC**
- Remote storage on **AWS S3** (`dvc-s3` plugin)
- Lock file (`dvc.lock`) ensures full pipeline reproducibility
- Pipeline parameters externalized in `params.yaml`

### 🐳 Containerization
- **Multi-stage optimized Dockerfile** using `python:3.11-slim-bookworm`
- **Docker Compose** for one-command local deployment
- Environment variables injected at runtime (no secrets baked into the image)
- Artifacts mounted as a **read-only volume** for live model updates without rebuilds

### ⚙️ CI/CD (GitHub Actions)
- Automated **test** job on every push and pull request to `main`:
  - Sets up Python 3.11 with pip caching
  - Installs all dependencies
  - Validates FastAPI app imports and startup
- Automated **build** job (on push to `main` only):
  - Builds the production Docker image
  - Extensible to push to container registry or deploy to cloud

---

## 🛠️ Tech Stack

### Backend & ML
| Category | Technology | Version |
|----------|-----------|---------|
| Language | Python | 3.11 |
| API Framework | FastAPI | 0.136 |
| ASGI Server | Uvicorn | 0.44 |
| Data Validation | Pydantic v2 + pydantic-settings | 2.13 |
| ML Core | Scikit-learn | 1.8 |
| Gradient Boosting | LightGBM | 4.6 |
| Gradient Boosting | XGBoost | 3.2 |
| Hyperparameter Tuning | Optuna | 4.8 |
| Imbalance Handling | imbalanced-learn | 0.14 |
| Vectorization | TF-IDF (Scikit-learn) | 1.8 |
| Experiment Tracking | MLflow | 3.11 |
| Serialization | Joblib | 1.5 |
| Data Manipulation | Pandas | 2.3 |
| Numerical Computing | NumPy | 2.4 |
| Visualization | Matplotlib + Seaborn | 3.10 / 0.13 |
| Configuration | PyYAML + python-dotenv | 6.0 / 1.2 |
| NLP Utilities | NLTK | 3.9 |

### Data & MLOps
| Category | Technology |
|----------|-----------|
| Pipeline Orchestration | DVC (Data Version Control) |
| Artifact Storage | AWS S3 (via `dvc-s3`) |
| Cloud SDK | boto3 / AWS CLI |
| Experiment Tracking | MLflow (remote server) |
| Model Registry | MLflow Model Registry |

### Frontend (Chrome Extension)
| Category | Technology | Version |
|----------|-----------|---------|
| Framework | React | 19 |
| Build Tool | Vite | 8 |
| Extension Plugin | @crxjs/vite-plugin | 2.0-beta |
| Styling | Tailwind CSS | 4 |
| Charts | Recharts | 3.8 |
| Icons | Lucide React | 1.14 |
| API Client | YouTube Data API v3 | — |
| Linting | ESLint | 10 |

### Infrastructure & DevOps
| Category | Technology |
|----------|-----------|
| Containerization | Docker + Docker Compose |
| Base Image | `python:3.11-slim-bookworm` |
| CI/CD | GitHub Actions |
| Secrets Management | `.env` + Docker environment variables |

---

## 📁 Project Structure

```
YouTube Viewer Sentiment Analysis/
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI/CD pipeline
│
├── app/                            # FastAPI application entry point
│   ├── main.py                     # App factory, lifespan manager
│   └── dependencies.py             # Singleton dependency injection
│
├── src/                            # Core source code
│   ├── config/
│   │   └── settings.py             # Pydantic-settings configuration
│   ├── data/
│   │   ├── data_ingestion.py       # DVC Stage 1: Download & split data
│   │   ├── data_preprocessing.py   # DVC Stage 2: Text cleaning
│   │   ├── preprocessor.py         # VectorizerFactory, TextCleanerTransformer
│   │   ├── data_pipeline.py        # Combined data pipeline orchestrator
│   │   └── loader.py               # Data loading utilities
│   ├── model/
│   │   ├── model_building.py       # DVC Stage 3: Build & train model
│   │   ├── trainer.py              # Trainer class (encode→vectorize→tune→log)
│   │   ├── tuner.py                # ModelFactory with Optuna/GridSearch/Random
│   │   ├── model_evaluation.py     # DVC Stage 4: Evaluate & log metrics
│   │   ├── evaluator.py            # Evaluation helper
│   │   └── register_model.py       # DVC Stage 5: MLflow model registration
│   ├── pipelines/
│   │   └── train_pipeline.py       # High-level training pipeline runner
│   ├── routers/
│   │   ├── api.py                  # Root API router
│   │   └── v1/
│   │       ├── predict.py          # POST /predict & /predict/batch endpoints
│   │       └── health.py           # GET /health endpoint
│   ├── schemas/
│   │   ├── request.py              # Pydantic request models
│   │   └── response.py             # Pydantic response models
│   └── services/
│       └── prediction_service.py   # PredictionService: load artifacts & infer
│
├── chrome-extension/               # Chrome Extension frontend
│   ├── src/
│   │   ├── App.jsx                 # Main React component & UI logic
│   │   ├── App.css                 # Component styles
│   │   ├── main.jsx                # React DOM entry point
│   │   └── index.css               # Global styles
│   ├── manifest.json               # Chrome extension manifest (MV3)
│   ├── vite.config.js              # Vite + CRXJS build configuration
│   └── package.json                # Node.js dependencies
│
├── notebooks/                      # Jupyter notebooks for EDA & experiments
├── data/                           # DVC-managed data (gitignored)
│   ├── raw/                        # train.csv, test.csv (DVC output)
│   └── processed/                  # Cleaned datasets (DVC output)
├── artifacts/                      # DVC-managed model artifacts (gitignored)
│   ├── model.joblib                # Trained classifier
│   ├── preprocessor.joblib         # Fitted TF-IDF vectorizer
│   └── label_encoder.joblib        # Fitted label encoder
├── metrics/                        # Evaluation metrics output (DVC)
│   └── experiment_info.json
│
├── Dockerfile                      # Production Docker image
├── docker-compose.yml              # Docker Compose orchestration
├── dvc.yaml                        # DVC pipeline definition
├── dvc.lock                        # DVC pipeline lock file
├── params.yaml                     # Model & pipeline hyperparameters
├── requirements.txt                # Python dependencies (pinned)
├── .env                            # Environment variables (not committed)
└── .gitignore
```

---

## 🔄 ML Pipeline (DVC Stages)

All stages are defined in `dvc.yaml` and parameterized via `params.yaml`. Run the full pipeline with:

```bash
dvc repro
```

### Stage 1 — Data Ingestion
```bash
python -m src.data.data_ingestion
```
- Downloads the raw Reddit sentiment dataset from GitHub
- Re-maps sentiment labels: `{-1 → 2 (negative), 0 → 0 (neutral), 1 → 1 (positive)}`
- Performs a stratified 80/20 train/test split
- Outputs: `data/raw/train.csv`, `data/raw/test.csv`

### Stage 2 — Data Preprocessing
```bash
python -m src.data.data_preprocessing
```
- Cleans text: lowercasing, punctuation removal, null/duplicate handling
- Outputs: `data/processed/train_processed.csv`, `data/processed/test_processed.csv`

### Stage 3 — Model Building
```bash
python -m src.model.model_building
```
- Fits TF-IDF vectorizer (n-grams 1–3, up to 10,000 features)
- Applies imbalance correction (SMOTE by default)
- Runs Bayesian hyperparameter search via Optuna (30 trials, stratified 5-fold CV, optimized for F1-macro)
- Outputs: `artifacts/model.joblib`, `artifacts/preprocessor.joblib`, `artifacts/label_encoder.joblib`

### Stage 4 — Model Evaluation
```bash
python -m src.model.model_evaluation
```
- Evaluates the trained model on the held-out test set
- Logs accuracy, F1-macro, and confusion matrix heatmap to MLflow
- Outputs: `metrics/experiment_info.json`

### Stage 5 — Model Registration
```bash
python -m src.model.register_model
```
- Reads `metrics/experiment_info.json` to identify the best run
- Registers the model in the **MLflow Model Registry**

---

## 📡 API Reference

Base URL: `http://localhost:8000`

### `GET /api/v1/health`
Returns the service liveness status.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /api/v1/predict/`
Predicts sentiment for a single comment.

**Request Body:**
```json
{
  "comment": "This video is absolutely amazing!"
}
```

**Response:**
```json
{
  "comment": "This video is absolutely amazing!",
  "sentiment": "positive",
  "confidence": 0.9421
}
```

---

### `POST /api/v1/predict/batch`
Predicts sentiment for multiple comments in a single request.

**Request Body:**
```json
{
  "comments": [
    "Love this content!",
    "Not impressed at all.",
    "Just watched it."
  ]
}
```

**Response:**
```json
{
  "predictions": [
    { "comment": "Love this content!", "sentiment": "positive", "confidence": 0.97 },
    { "comment": "Not impressed at all.", "sentiment": "negative", "confidence": 0.89 },
    { "comment": "Just watched it.", "sentiment": "neutral", "confidence": 0.76 }
  ]
}
```

Interactive API docs are available at: `http://localhost:8000/docs`

---

## 🧩 Chrome Extension

The Chrome Extension connects to the FastAPI backend and provides a live sentiment dashboard directly in the browser.

### Features
- 🔍 Fetches comments from any YouTube video using the **YouTube Data API v3**
- 📊 Sends comments in batch to `/api/v1/predict/batch` for instant classification
- 📈 Visualizes the sentiment distribution with **Recharts** (pie/bar charts)
- 🏷️ Displays per-comment sentiment label and confidence score

### Build the Extension
```bash
cd chrome-extension
npm install
npm run build
```

Load `chrome-extension/dist/` as an **unpacked extension** in Chrome DevTools.

### Extension Configuration
Create `chrome-extension/.env`:
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_YOUTUBE_API_KEY=your_youtube_data_api_v3_key
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Node.js 18+ (for Chrome Extension)
- AWS credentials (for DVC remote)
- YouTube Data API v3 key (for Chrome Extension)

### 1. Clone the Repository
```bash
git clone https://github.com/mohammedkh97/youtube-viewer-sentiment-analysis.git
cd youtube-viewer-sentiment-analysis
```

### 2. Configure Environment Variables
Copy the example and fill in your values:
```bash
cp .env.example .env
```

`.env` keys:
```env
MLFLOW_TRACKING_URI=https://mlflow-dashboard.duckdns.org
APP_PORT=8000
APP_HOST=0.0.0.0
DEBUG=False
```

### 3. Install Python Dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 4. Pull Data & Artifacts from DVC Remote
```bash
dvc pull
```

### 5. Run the Full ML Pipeline (Optional — if artifacts not pulled)
```bash
dvc repro
```

### 6. Run the FastAPI Backend

**Option A — Directly with Uvicorn:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Option B — With Docker Compose (Recommended):**
```bash
docker-compose up --build
```

The API will be live at `http://localhost:8000`.
Swagger UI: `http://localhost:8000/docs`

---

## ⚙️ Configuration

All model and pipeline parameters live in `params.yaml`:

```yaml
data_ingestion:
  test_size: 0.20

model_training:
  vectorizer: "tfidf"
  ngram_range: [1, 3]
  max_features: 10000
  imbalance_method: "oversampling"   # oversampling | adasyn | undersampling | smote_enn | class_weights
  learning_rate: 0.09
  max_depth: 20
  n_estimators: 367

bayesian_tuning:
  model_name: "lightgbm"             # lightgbm | xgboost | random_forest
  search_strategy: "bayesian"        # bayesian | random | grid | manual
  cv_strategy: "stratified_kfold"    # stratified_kfold | kfold
  n_splits: 5
  n_trials: 30
```

Application settings are loaded from environment variables (`.env`) via `pydantic-settings`.

---

## 🔁 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and pull request to `main`:

```
┌──────────────────────────────┐
│           TEST JOB           │
│  Ubuntu Latest · Python 3.11 │
│  ─────────────────────────── │
│  1. Checkout code            │
│  2. Set up Python (pip cache)│
│  3. Install requirements     │
│  4. Validate app startup     │
└──────────┬───────────────────┘
           │ (success, push to main only)
           ▼
┌──────────────────────────────┐
│          BUILD JOB           │
│  Ubuntu Latest               │
│  ─────────────────────────── │
│  1. Checkout code            │
│  2. Build Docker image       │
└──────────────────────────────┘
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using Python, FastAPI, LightGBM, MLflow, DVC, React, and Docker.
</p>

---

## 📬 Contact

<p align="center">
  <a href="https://linkedin.com/in/mohammed-khalaf97"><img src="https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
  &nbsp;
  <a href="https://github.com/mohammedkh97"><img src="https://img.shields.io/badge/-GitHub-000?style=flat&logo=github&logoColor=white" alt="GitHub"/></a>
  &nbsp;
  <a href="mailto:mohamedkhalaf20172020@gmail.com"><img src="https://img.shields.io/badge/-Gmail-c14438?style=flat&logo=gmail&logoColor=white" alt="Gmail"/></a>
  &nbsp;
  <a href="https://wa.me/+971547331688"><img src="https://img.shields.io/badge/-WhatsApp-25D366?style=flat&logo=whatsapp&logoColor=white" alt="WhatsApp"/></a>
</p>
