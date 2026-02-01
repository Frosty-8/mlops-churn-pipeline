# End-to-End MLOps Pipeline for Customer Churn Prediction

This repository implements a **production-style MLOps pipeline** for a machine learning system that predicts **customer churn**.
It demonstrates how to **design, train, monitor, retrain, and deploy** an ML model while maintaining a **clean separation between training and serving**.

The project is intentionally designed to resemble **real-world MLOps workflows**, not notebook-based demos.

---

## 🔍 What This Project Does

At a high level, the system:

1. **Generates large-scale synthetic data** that mimics real customer behavior
2. **Trains and evaluates a machine learning model** with experiment tracking
3. **Detects data drift** between historical and live data
4. **Automatically retrains the model** using CI/CD (GitHub Actions)
5. **Deploys a FastAPI inference service** on Render
6. **Serves real-time predictions** with structured logging and observability

This mirrors how modern ML systems are operated in production.

---

## 🧠 Why This Is an MLOps Project (Not Just ML)

Traditional ML projects stop after training a model.

This project goes further by addressing **operational ML concerns**:

* How do we **retrain models safely**?
* How do we **detect when a model becomes stale**?
* How do we **separate training from serving**?
* How do we **automate model lifecycle events**?
* How do we **observe and debug live predictions**?

Each module in this repository exists to answer one of those questions.

---

## 🏗️ System Architecture (Conceptual)

```
Synthetic Data Generation (Faker)
            ↓
Feature Engineering Pipeline
            ↓
Model Training + Evaluation
            ↓
MLflow Experiment Tracking
            ↓
Model Artifact (best_model.pkl)
            ↓
────────────────────────────────
 Training Plane (CI / Local)
────────────────────────────────
            ↓
 GitHub Actions (Scheduled Retraining)
            ↓
 Commit Updated Model to GitHub
            ↓
 Render Auto-Redeploy
────────────────────────────────
 Serving Plane (Render)
────────────────────────────────
            ↓
 FastAPI Inference Service
            ↓
 Real-Time Predictions + Logs
```

**Key design principle:**
👉 *Training and serving never run in the same environment.*

---

## 📂 Repository Structure (Why It’s Organized This Way)

```
mlops-churn-pipeline/
├── src/
│   ├── api/        # Inference service (FastAPI)
│   ├── data/       # Synthetic data generation
│   ├── features/   # Feature engineering logic
│   ├── training/   # Training & retraining pipelines
│   ├── drift/      # Data drift detection
│   └── utils/      # Logging & shared utilities
│
├── models/         # Persisted model artifacts
├── data/           # Raw / live datasets
├── mlruns/         # MLflow experiment tracking
├── .github/        # CI/CD workflows (GitHub Actions)
└── start.sh        # Render entrypoint
```

This structure ensures:

* Reproducibility
* Clear ownership of responsibilities
* CI/CD compatibility
* Cloud deployment readiness

---

## 🔁 Training & Retraining Workflow

### Initial Training

Training can be run locally or in CI:

```bash
uv run -m src.training.train
```

What happens:

* Loads historical data
* Applies feature engineering
* Trains a RandomForest model
* Logs metrics and artifacts to MLflow
* Saves the best model to `models/best_model.pkl`

---

### Automated Retraining (CI/CD)

Retraining is handled by **GitHub Actions**, not the deployed service.

The workflow:

1. Scheduled job runs (or manual trigger)
2. New synthetic data is generated
3. Drift is evaluated
4. Model is retrained
5. Updated model artifact is committed
6. Render automatically redeploys the API

This ensures:

* No downtime
* No training on production servers
* Full traceability of model changes

---

## 📉 Drift Detection

The system includes **statistical drift detection** using the Kolmogorov–Smirnov (KS) test.

Purpose:

* Detect changes in feature distributions
* Identify when historical assumptions no longer hold
* Justify retraining decisions

This simulates how real ML systems monitor data health over time.

---

## 🌐 Inference Service (FastAPI)

The deployed service exposes:

* `GET /` → Health check
* `POST /predict` → Real-time churn prediction
* `/docs` → Interactive Swagger UI

The API:

* Loads the latest trained model at startup
* Applies the same feature pipeline used during training
* Logs every request and prediction

This ensures **training–serving consistency**.

---

## 📊 Logging & Observability

Structured logging is implemented across:

* Data generation
* Training
* Drift detection
* Retraining
* Inference requests

Logs are written to STDOUT, which means they are automatically captured by:

* Render logs
* GitHub Actions logs
* Local execution

This enables:

* Debugging
* Auditability
* Production observability

---

## ☁️ Deployment Strategy (Why Render)

* FastAPI is deployed as a **Web Service**
* Training is **never run on Render**
* Render only serves inference traffic
* Redeployment happens automatically on Git push

This matches industry best practices for:

* Cost control
* Reliability
* Separation of concerns

---

## 🧪 How to Run Locally

```bash
uv pip install -r requirements.txt
uv run -m src.data.generate_data
uv run -m src.training.train
uvicorn src.api.main:app --reload
```

Then open:

```
http://127.0.0.1:8000/docs
```

---

## 🎯 Who This Project Is For

* ML Engineers
* Applied AI Engineers
* MLOps Engineers
* Students transitioning from ML → production systems

This repository is intentionally designed to be **explainable in interviews** and **maintainable in teams**.

---

## 🧠 Key MLOps Takeaways

* Training ≠ Serving
* Automation > manual retraining
* Observability is mandatory
* CI/CD is part of ML systems
* Models are products, not files

---

## 📌 Final Note

This project prioritizes **system design and operational correctness** over leaderboard accuracy.
It is meant to demonstrate **how ML works in the real world**, not just how models are trained.