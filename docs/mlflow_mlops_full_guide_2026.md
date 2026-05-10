# MLflow for MLOps — Complete Practical Guide (2026)

## Table of Contents
1. Introduction to MLflow
2. Why MLflow Matters in MLOps
3. MLflow Architecture
4. Installing MLflow
5. MLflow Tracking
6. Experiment Management
7. Logging Parameters, Metrics, and Artifacts
8. Model Logging
9. MLflow Projects
10. MLflow Models
11. MLflow Model Registry
12. MLflow Deployment
13. MLflow with FastAPI
14. MLflow with Docker
15. MLflow with Kubernetes
16. MLflow with AWS
17. MLflow with Azure
18. MLflow with Databricks
19. MLflow Pipelines
20. MLflow for Deep Learning
21. MLflow for NLP Projects
22. MLflow for Computer Vision
23. MLflow with LangChain and LLMs
24. MLflow Evaluation
25. MLflow Monitoring
26. MLflow Best Practices
27. CI/CD for MLflow
28. Production MLOps Architecture
29. End-to-End Real Project
30. Interview Questions
31. Common Errors and Fixes
32. Learning Roadmap
33. Useful Resources

---

# 1. Introduction to MLflow

## What is MLflow?

MLflow is an open-source MLOps platform designed to manage the complete machine learning lifecycle.

It helps with:

- Experiment tracking
- Model packaging
- Model registry
- Deployment
- Reproducibility
- Collaboration
- Monitoring

MLflow was originally developed by Databricks.

---

# 2. Why MLflow Matters in MLOps

Without MLflow:

- Experiments become hard to track
- Model versions become confusing
- Reproducing results becomes difficult
- Deployment pipelines become inconsistent
- Collaboration becomes messy

MLflow solves these issues by centralizing the ML lifecycle.

---

# 3. MLflow Architecture

MLflow contains four major components:

## 1. MLflow Tracking
Tracks:
- Parameters
- Metrics
- Artifacts
- Source code
- Environment

## 2. MLflow Projects
Packages ML code into reproducible projects.

## 3. MLflow Models
Standard model packaging format.

## 4. MLflow Model Registry
Centralized model version management.

---

# 4. Installing MLflow

## Basic Installation

```bash
pip install mlflow
```

## Verify Installation

```bash
mlflow --version
```

## Install with Extras

### Scikit-learn

```bash
pip install mlflow scikit-learn
```

### PyTorch

```bash
pip install mlflow torch torchvision
```

### TensorFlow

```bash
pip install mlflow tensorflow
```

### LangChain

```bash
pip install mlflow langchain openai
```

---

# 5. MLflow Tracking

## Starting the Tracking UI

```bash
mlflow ui
```

Default:

```text
http://127.0.0.1:5000
```

---

## First Tracking Example

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", 10)

    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.12)
```

---

# 6. Experiment Management

## Create Experiment

```python
import mlflow

mlflow.set_experiment("classification_experiment")
```

---

## Start Run

```python
with mlflow.start_run(run_name="baseline_model"):
    pass
```

---

## Nested Runs

```python
with mlflow.start_run(run_name="parent"):

    with mlflow.start_run(run_name="child", nested=True):
        pass
```

---

# 7. Logging Parameters, Metrics, and Artifacts

## Logging Parameters

```python
mlflow.log_param("batch_size", 32)
```

---

## Logging Multiple Parameters

```python
params = {
    "lr": 0.001,
    "optimizer": "adam",
    "epochs": 20
}

mlflow.log_params(params)
```

---

## Logging Metrics

```python
mlflow.log_metric("accuracy", 0.91)
```

---

## Logging Multiple Metrics

```python
metrics = {
    "accuracy": 0.91,
    "f1": 0.89,
    "precision": 0.88
}

mlflow.log_metrics(metrics)
```

---

## Logging Files

```python
mlflow.log_artifact("model.pkl")
```

---

## Logging Entire Folder

```python
mlflow.log_artifacts("outputs/")
```

---

# 8. Model Logging

## Scikit-learn Example

```python
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestClassifier()
model.fit(X_train, y_train)

with mlflow.start_run():

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )
```

---

## Loading Logged Model

```python
logged_model = 'runs:/RUN_ID/model'

loaded_model = mlflow.sklearn.load_model(logged_model)
```

---

# 9. MLflow Projects

MLflow Projects standardize execution.

## Project Structure

```text
project/
│
├── MLproject
├── conda.yaml
├── train.py
└── requirements.txt
```

---

## MLproject File

```yaml
name: house_price_prediction

entry_points:
  main:
    parameters:
      learning_rate: {type: float, default: 0.001}
    command: "python train.py --lr {learning_rate}"
```

---

## Running Project

```bash
mlflow run . -P learning_rate=0.01
```

---

# 10. MLflow Models

MLflow supports many flavors:

- sklearn
- pytorch
- tensorflow
- xgboost
- transformers
- langchain
- pyfunc

---

## PyFunc Format

Universal inference format.

```python
import mlflow.pyfunc
```

---

# 11. MLflow Model Registry

Model registry manages:

- Model versions
- Stage transitions
- Production deployment
- Approval workflows

---

## Register Model

```python
result = mlflow.register_model(
    "runs:/RUN_ID/model",
    "iris_classifier"
)
```

---

## Model Stages

Stages:

- None
- Staging
- Production
- Archived

---

## Transition Stage

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

client.transition_model_version_stage(
    name="iris_classifier",
    version=1,
    stage="Production"
)
```

---

# 12. MLflow Deployment

## Local Deployment

```bash
mlflow models serve -m runs:/RUN_ID/model -p 5001
```

---

## Test Endpoint

```bash
curl http://127.0.0.1:5001/invocations
```

---

# 13. MLflow with FastAPI

## FastAPI Integration Example

```python
from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

model = mlflow.pyfunc.load_model(
    "models:/iris_classifier/Production"
)

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return {
        "prediction": prediction.tolist()
    }
```

---

# 14. MLflow with Docker

## Dockerfile

```dockerfile
FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
```

---

## Build Docker Image

```bash
docker build -t mlflow-app .
```

---

## Run Container

```bash
docker run -p 8000:8000 mlflow-app
```

---

# 15. MLflow with Kubernetes

## Deployment Example

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow
        ports:
        - containerPort: 5000
```

---

# 16. MLflow with AWS

Typical stack:

- EC2
- S3
- RDS
- ECR
- ECS/EKS

---

## Remote Artifact Storage

```bash
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root s3://mlflow-artifacts \
--host 0.0.0.0
```

---

# 17. MLflow with Azure

Azure services:

- Azure Blob Storage
- Azure ML
- AKS
- Azure Container Registry

---

# 18. MLflow with Databricks

Databricks has native MLflow integration.

Benefits:

- Managed tracking
- Managed registry
- Scalable training
- Distributed computing

---

# 19. MLflow Pipelines

MLflow pipelines automate:

- Data ingestion
- Training
- Evaluation
- Packaging
- Deployment

---

# 20. MLflow for Deep Learning

## PyTorch Example

```python
import mlflow.pytorch
import torch

model = MyModel()

with mlflow.start_run():
    mlflow.pytorch.log_model(model, "model")
```

---

## TensorFlow Example

```python
import mlflow.tensorflow

mlflow.tensorflow.log_model(model, "model")
```

---

# 21. MLflow for NLP Projects

## Hugging Face Transformers

```python
import mlflow.transformers

mlflow.transformers.log_model(
    transformers_model=pipeline,
    artifact_path="sentiment_model"
)
```

---

# 22. MLflow for Computer Vision

Track:

- Images
- Confusion matrices
- ROC curves
- Model checkpoints
- Augmentations

---

## Log Image

```python
mlflow.log_artifact("sample_prediction.png")
```

---

# 23. MLflow with LangChain and LLMs

## Logging LangChain Models

```python
import mlflow

mlflow.langchain.autolog()
```

---

## Benefits

- Prompt tracking
- Token usage tracking
- Chain tracking
- Response logging
- Evaluation

---

# 24. MLflow Evaluation

## Evaluate Model

```python
result = mlflow.evaluate(
    model_uri,
    eval_data,
    targets="label",
    model_type="classifier"
)
```

---

# 25. MLflow Monitoring

Monitor:

- Drift
- Latency
- Accuracy
- Throughput
- Resource usage

---

# 26. MLflow Best Practices

## 1. Use Structured Experiment Names

Good:

```text
fraud_detection_xgboost_v2
```

Bad:

```text
test1
```

---

## 2. Always Log Environment

```bash
pip freeze > requirements.txt
```

---

## 3. Version Everything

- Data
- Code
- Models
- Configurations

---

## 4. Use Remote Artifact Storage

Avoid local-only storage in production.

---

## 5. Use Model Registry

Never deploy directly from training outputs.

---

# 27. CI/CD for MLflow

## Recommended Tools

- GitHub Actions
- Jenkins
- GitLab CI/CD
- ArgoCD
- Kubeflow

---

## CI/CD Flow

```text
Code Push
    ↓
Tests
    ↓
Training
    ↓
MLflow Tracking
    ↓
Model Registry
    ↓
Validation
    ↓
Deployment
```

---

# 28. Production MLOps Architecture

```text
Data Sources
      ↓
Data Pipeline
      ↓
Training Pipeline
      ↓
MLflow Tracking
      ↓
Model Registry
      ↓
CI/CD
      ↓
Deployment
      ↓
Monitoring
```

---

# 29. End-to-End Real Project

## Problem

Customer churn prediction.

---

## Step 1 — Data Loading

```python
import pandas as pd

train_df = pd.read_csv("train.csv")
```

---

## Step 2 — Training

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)
```

---

## Step 3 — Tracking

```python
with mlflow.start_run():

    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "model")
```

---

## Step 4 — Register Model

```python
mlflow.register_model(
    "runs:/RUN_ID/model",
    "customer_churn_model"
)
```

---

## Step 5 — Deployment

```bash
mlflow models serve \
-m models:/customer_churn_model/Production
```

---

# 30. Interview Questions

## Beginner

1. What is MLflow?
2. Why is MLflow important?
3. What are MLflow components?
4. Difference between artifacts and metrics?
5. What is model registry?

---

## Intermediate

1. How does MLflow handle reproducibility?
2. Explain model stages.
3. Difference between MLflow Projects and Pipelines?
4. How do you deploy models?
5. How does MLflow integrate with Docker?

---

## Advanced

1. Design an enterprise MLflow architecture.
2. How would you scale MLflow?
3. Explain multi-user tracking.
4. How do you implement governance?
5. How do you monitor production drift?

---

# 31. Common Errors and Fixes

## Error

```text
MlflowException: Run not found
```

## Fix

Ensure correct run ID.

---

## Error

```text
Connection refused
```

## Fix

Ensure tracking server is running.

---

## Error

```text
Artifact upload failed
```

## Fix

Check permissions for S3 or storage backend.

---

# 32. Learning Roadmap

## Stage 1 — Fundamentals

Learn:

- Experiment tracking
- Logging
- Registry
- Deployment

---

## Stage 2 — Production

Learn:

- Docker
- Kubernetes
- CI/CD
- Cloud deployment

---

## Stage 3 — Advanced MLOps

Learn:

- Monitoring
- Drift detection
- LLMOps
- Distributed training
- Governance

---

# 33. Useful Resources

## Official Documentation

- MLflow Docs
- Databricks MLflow Guide

---

## Recommended Projects

1. Fraud detection
2. Resume screening
3. OCR document extraction
4. Sentiment analysis
5. CV-job matching
6. Chatbot evaluation

---

# Recommended Tech Stack

## Beginner Stack

- MLflow
- Scikit-learn
- FastAPI
- Docker

---

## Advanced Stack

- MLflow
- Kubernetes
- Airflow
- Spark
- Kafka
- Prometheus
- Grafana
- AWS/GCP/Azure

---

# Final Advice

Master these topics deeply:

1. Experiment Tracking
2. Model Registry
3. Deployment
4. Monitoring
5. CI/CD
6. Cloud Infrastructure
7. Reproducibility
8. Distributed Systems
9. LLMOps
10. Observability

MLflow becomes extremely powerful when combined with:

- FastAPI
- Docker
- Kubernetes
- Airflow
- LangChain
- Vector Databases
- Cloud Infrastructure

This combination forms a modern production-grade MLOps platform.

