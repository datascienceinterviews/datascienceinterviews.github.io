---
title: Deploying Machine Learning Models to Production
description: Complete guide to ML deployment - Docker containerization, Kubernetes orchestration, Flask/FastAPI APIs, AWS/GCP/Azure cloud deployment, CI/CD pipelines, and MLOps best practices.
---

# 🚀 Deploying Machine Learning Models to Production

A model that lives in a notebook helps no one. This guide walks through the full path from a trained model to a reliable production service: packaging, serving, containerization, orchestration, cloud deployment, and the MLOps practices that keep it healthy after launch.

## ✍️ Overview

Deploying a machine learning model means exposing it so that other systems (or users) can get predictions from it reliably, at the required scale, with acceptable latency. The main deployment patterns are:

| Pattern | How it works | Best for |
|---|---|---|
| **Batch (offline)** | Score data on a schedule, store results | Nightly risk scores, recommendations refreshed daily |
| **Online (real-time API)** | Model behind an HTTP/gRPC endpoint | Fraud checks, search ranking, chat features |
| **Streaming** | Model consumes an event stream (Kafka, Kinesis) | Clickstream enrichment, IoT anomaly detection |
| **Edge / on-device** | Model ships inside the app or device | Mobile vision, offline inference, privacy-sensitive use |

Most interview questions and most real systems revolve around the **online API** pattern, so that is the focus below.

## 📦 Step 1 — Package the Model

Serialize the trained model into a portable artifact:

```python
# scikit-learn
import joblib
joblib.dump(model, "model.joblib")

# PyTorch
import torch
torch.save(model.state_dict(), "model.pt")

# TensorFlow / Keras
model.save("saved_model/")   # SavedModel format
```

**Good practice:**

- Version every artifact (`model-v1.3.2.joblib`), never overwrite in place.
- Store artifacts in an object store or a model registry (MLflow, Weights & Biases, SageMaker Model Registry), not in git.
- Save the **preprocessing pipeline together with the model** (e.g. a scikit-learn `Pipeline`) so training and serving transformations can never drift apart.
- Record the exact library versions used at training time — a model pickled under one scikit-learn version may not load under another.

## 🌐 Step 2 — Serve It Behind an API

### FastAPI (recommended)

FastAPI is the modern default: async, type-validated, self-documenting.

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Churn Model API")
model = joblib.load("model.joblib")   # load once at startup, not per request

class Features(BaseModel):
    tenure_months: int
    monthly_charges: float
    num_support_tickets: int

@app.post("/predict")
def predict(payload: Features):
    X = [[payload.tenure_months, payload.monthly_charges, payload.num_support_tickets]]
    proba = float(model.predict_proba(X)[0][1])
    return {"churn_probability": proba, "model_version": "1.3.2"}

@app.get("/health")
def health():
    return {"status": "ok"}
```

Run it with a production server:

```shell
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Flask (classic)

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    X = [[data["tenure_months"], data["monthly_charges"], data["num_support_tickets"]]]
    return jsonify({"churn_probability": float(model.predict_proba(X)[0][1])})
```

Serve Flask with `gunicorn` in production — never the built-in dev server:

```shell
gunicorn -w 4 -b 0.0.0.0:8000 main:app
```

**API design essentials:** validate inputs (Pydantic does this for free), return a model version with every response, expose a `/health` endpoint for load balancers, and set request timeouts.

## 🐳 Step 3 — Containerize with Docker

A container makes the service reproducible on any machine.

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py model.joblib ./

EXPOSE 8000
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```shell
docker build -t churn-api:1.3.2 .
docker run -p 8000:8000 churn-api:1.3.2
```

**Good practice:** pin dependency versions in `requirements.txt`, use slim base images, tag images with the model version, and keep images small (multi-stage builds; don't ship training code or datasets).

## ☸️ Step 4 — Orchestrate with Kubernetes

Kubernetes handles scaling, rolling updates, and self-healing for containerized services.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-api
spec:
  replicas: 3
  selector:
    matchLabels: { app: churn-api }
  template:
    metadata:
      labels: { app: churn-api }
    spec:
      containers:
        - name: churn-api
          image: registry.example.com/churn-api:1.3.2
          ports: [{ containerPort: 8000 }]
          resources:
            requests: { cpu: "250m", memory: "512Mi" }
            limits:   { cpu: "1",    memory: "1Gi" }
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
---
apiVersion: v1
kind: Service
metadata:
  name: churn-api
spec:
  selector: { app: churn-api }
  ports: [{ port: 80, targetPort: 8000 }]
```

Key concepts worth knowing for interviews:

- **Deployment** — declares the desired number of identical pods and handles rolling updates.
- **Service / Ingress** — stable networking in front of ephemeral pods.
- **Horizontal Pod Autoscaler (HPA)** — scales replicas on CPU/memory or custom metrics (e.g. request latency).
- **Readiness vs liveness probes** — readiness gates traffic; liveness restarts stuck containers.

For teams that don't need full Kubernetes control, managed serverless containers (Cloud Run, AWS App Runner, Azure Container Apps) offer most of the benefit with far less operational load.

## ☁️ Step 5 — Deploy to the Cloud

| | AWS | GCP | Azure |
|---|---|---|---|
| **Managed ML platform** | SageMaker | Vertex AI | Azure ML |
| **Serverless containers** | App Runner / Fargate | Cloud Run | Container Apps |
| **Functions (light models)** | Lambda | Cloud Functions | Azure Functions |
| **Kubernetes** | EKS | GKE | AKS |
| **Model registry** | SageMaker Registry | Vertex Model Registry | Azure ML Registry |

Rules of thumb:

- **Small/simple model, spiky traffic** → serverless functions or serverless containers (scale to zero, pay per use). Watch out for cold starts.
- **Steady traffic, standard API** → serverless containers or a small Kubernetes deployment.
- **Heavy models / GPUs / autoscaling inference** → the managed ML platforms (SageMaker, Vertex AI, Azure ML) provide GPU serving, A/B endpoints, and built-in monitoring.

## 🔁 Step 6 — CI/CD for ML

Automate the path from commit to deployment:

1. **CI (on every commit):** run unit tests, data-schema tests, and a quick model smoke test (load artifact, predict on fixture rows, assert output shape/range).
2. **Build:** package the service into a versioned Docker image.
3. **CD (on approval or tag):** deploy with a **rolling update**, **blue-green**, or **canary** strategy — canary (send 5–10% of traffic to the new model, compare metrics, then promote) is the safest for model changes.
4. **Rollback plan:** keep the previous image and model artifact one command away.

GitHub Actions sketch:

```yaml
on:
  push:
    tags: ["model-v*"]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/
      - run: docker build -t $REGISTRY/churn-api:${{ github.ref_name }} .
      - run: docker push $REGISTRY/churn-api:${{ github.ref_name }}
      - run: kubectl set image deployment/churn-api churn-api=$REGISTRY/churn-api:${{ github.ref_name }}
```

## 📈 Step 7 — Monitor and Maintain (MLOps)

Deployment is the beginning, not the end. Monitor four layers:

1. **Service health** — latency (p50/p95/p99), error rate, throughput, saturation. Standard tools: Prometheus + Grafana, CloudWatch, Datadog.
2. **Input data quality** — missing fields, out-of-range values, schema changes from upstream.
3. **Drift** — compare live input distributions to training distributions (PSI, KL divergence) and watch prediction distributions shift. Tools: Evidently, whylogs, SageMaker Model Monitor.
4. **Model performance** — when ground-truth labels arrive (often delayed), track live accuracy/AUC against the offline baseline and alert on degradation.

**Retraining strategy:** decide upfront whether retraining is scheduled (weekly/monthly), triggered by drift alerts, or continuous — and make sure every retrained model goes through the same evaluation gate before promotion.

## ⚠️ Common Pitfalls

- **Training–serving skew** — preprocessing implemented twice (once in the notebook, once in the API) drifts apart. Ship one pipeline artifact.
- **Loading the model per request** — load once at process startup; per-request loading destroys latency.
- **No model versioning** — you cannot debug "the model got worse" if you don't know which artifact is live.
- **Silent input changes** — an upstream team renames a field and your model quietly predicts on defaults. Validate schemas loudly.
- **Ignoring cold starts** — large models on serverless functions can add seconds of latency; keep models warm or use provisioned concurrency.
- **No rollback path** — every deployment should be reversible in one step.

## 💡 Interview Questions

- What is the difference between batch and online inference, and when would you choose each?
- How do you prevent training–serving skew?
- Walk through deploying a scikit-learn model as a REST API with Docker and Kubernetes.
- What is a canary deployment, and why is it especially useful for model releases?
- How would you detect data drift in production, and what would you do when it's detected?
- Why shouldn't you use Flask's development server (or load the model per request) in production?
- How do readiness and liveness probes differ in Kubernetes?

## 📚 References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker — Getting Started](https://docs.docker.com/get-started/)
- [Kubernetes Documentation](https://kubernetes.io/docs/home/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Evidently — ML Monitoring](https://www.evidentlyai.com/)
- [Google MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [AWS SageMaker Deployment](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
