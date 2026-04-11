# 🏦 Loan Approval Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-brightgreen?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end machine learning system that predicts whether a loan application should be
approved or rejected, achieving **99% accuracy** with a production-ready FastAPI
deployment and Dockerized API.

---


## Demo

<!-- Replace YOUR_VIDEO_ID with your YouTube video ID once ready -->
[![Loan Approval Prediction Demo](https://img.youtube.com/vi/V0YuGOytDMo/hqdefault.jpg)](https://www.youtube.com/watch?v=V0YuGOytDMo)

> Click the thumbnail above to watch the demo.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Model Results](#model-results)
- [Actual vs Fitted Values Analysis](#actual-vs-fitted-values-analysis)
- [Confusion Matrix Analysis](#confusion-matrix-analysis)
- [Key Insights](#key-insights)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Docker Deployment](#docker-deployment)
- [Key Design Decisions](#key-design-decisions)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [Technologies Used](#technologies-used)

---

## Overview

Banks process thousands of loan applications daily. Manual review is slow, inconsistent,
and expensive. This project builds an end-to-end ML pipeline that:

- Predicts loan approval (Approved / Rejected) from applicant financial and demographic data
- Achieves **99% accuracy** with **99% recall** on approved loans using a Decision Tree
- Returns the probability of approval alongside every prediction
- Deploys the trained model as a **production-ready FastAPI** with full Pydantic validation
- Ships as a **Dockerized API** deployable on any cloud platform

**Target variable:** `0` → Rejected · `1` → Approved

---

## Dataset

| Property | Details |
|---|---|
| File | `loan_approval_dataset.csv` |
| Domain | Banking / Financial Services |
| Task | Binary classification (Approved / Rejected) |
| Test set size | 854 samples |
| Class 0 (Rejected) | 318 samples |
| Class 1 (Approved) | 536 samples |

### Features

| Feature | Type | Description | Constraint |
|---|---|---|---|
| `no_of_dependents` | int | Number of financial dependents | 0–10 |
| `education` | categorical | Graduate / Not Graduate | — |
| `self_employed` | categorical | Self-employed status (Yes / No) | — |
| `income_annum` | float | Annual income (₹) | ≥ 0 |
| `loan_amount` | float | Requested loan amount (₹) | ≥ 0 |
| `loan_term` | int | Loan term in months | 6–360 |
| `cibil_score` | int | Credit score (CIBIL rating system) | 300–900 |
| `Movable_assets` | float | Value of movable assets (₹) | ≥ 0 |
| `Immovable_assets` | float | Value of immovable assets — property etc. (₹) | ≥ 0 |

> **Note on CIBIL score:** CIBIL is India's primary credit scoring system (equivalent to
> FICO in the US). A score above 750 is considered excellent and is the single strongest
> predictor of loan approval in this dataset.

---

## ML Pipeline

The entire pipeline — preprocessing + model — is saved as a single
`loan_pipeline_model.pkl` using sklearn's `Pipeline`:

```
Raw Input (9 features)
        │
        ▼
Preprocessing
  ├── StandardScaler   → income_annum, loan_amount, assets, cibil_score, loan_term
  └── OrdinalEncoder   → education, self_employed
        │
        ▼
Model Training & Comparison
  ├── Decision Tree    ← best model  ✅  (99% accuracy)
  └── Random Forest    ← ensemble comparison  (98% accuracy)
        │
        ▼
loan_pipeline_model.pkl   → saved best pipeline (Decision Tree)
        │
        ▼
Output: loan_status (Approved / Rejected) + approval_probability
```

---

## Model Results

### Overall comparison

| Model | Accuracy | Precision (Rejected) | Recall (Rejected) | F1 (Rejected) | Precision (Approved) | Recall (Approved) | F1 (Approved) |
|---|---|---|---|---|---|---|---|
| **Decision Tree** | **99%** | **0.98** | **0.98** | **0.98** | **0.99** | **0.99** | **0.99** |
| Random Forest | 98% | 0.98 | 0.95 | 0.97 | 0.97 | 0.99 | 0.98 |

### Decision Tree — full classification report (best model ✅)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 — Rejected | 0.98 | 0.98 | 0.98 | 318 |
| 1 — Approved | 0.99 | 0.99 | 0.99 | 536 |
| **Accuracy** | — | — | **0.99** | **854** |
| Macro avg | 0.99 | 0.98 | 0.98 | 854 |
| Weighted avg | 0.99 | 0.99 | 0.99 | 854 |

### Random Forest — full classification report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 — Rejected | 0.98 | 0.95 | 0.97 | 318 |
| 1 — Approved | 0.97 | 0.99 | 0.98 | 536 |
| **Accuracy** | — | — | **0.98** | **854** |
| Macro avg | 0.98 | 0.97 | 0.97 | 854 |
| Weighted avg | 0.98 | 0.98 | 0.98 | 854 |

---

## Actual vs Fitted Values Analysis

The density plots of actual vs fitted values reveal how confidently each model
separates the two classes:

**Decision Tree:**
Both the actual and fitted distributions show two sharp, well-separated peaks
tightly clustered at exactly 0 and 1, with near-zero density in between. The predicted
distribution almost perfectly overlaps with the actual distribution — confirming the
model has learned a clean binary separation with high confidence on both classes. The
near-zero overlap between the two peaks means the model is rarely uncertain.

**Random Forest:**
The Random Forest shows near-identical actual (red) and fitted (blue) distributions
— the two curves are virtually indistinguishable, confirming RF also learns the decision
boundary very well. However, the slightly flatter peak for class 0 (rejected loans)
compared to Decision Tree is consistent with its marginally lower recall (0.95 vs 0.98)
on rejected cases. Both models produce confident, bimodal predictions.

> **Key takeaway:** Both models produce near-perfect probability separation. The
> clean bimodal distributions in both plots confirm the dataset has clear, learnable
> decision boundaries — primarily driven by CIBIL score.

---

## Confusion Matrix Analysis

### Decision Tree (TN=311, FP=7, FN=5, TP=531)

| | Predicted: Rejected | Predicted: Approved |
|---|---|---|
| **Actual: Rejected** | ✅ 311 (TN) | ❌ 7 (FP) |
| **Actual: Approved** | ❌ 5 (FN) | ✅ 531 (TP) |

- Only **7 false approvals** — legitimate rejections incorrectly approved
- Only **5 missed approvals** — eligible applicants incorrectly rejected
- Total errors: **12 out of 854** (1.4% error rate)

### Random Forest (TN=303, FP=15, FN=6, TP=530)

| | Predicted: Rejected | Predicted: Approved |
|---|---|---|
| **Actual: Rejected** | ✅ 303 (TN) | ❌ 15 (FP) |
| **Actual: Approved** | ❌ 6 (FN) | ✅ 530 (TP) |

- **15 false approvals** — more than double Decision Tree's 7
- **6 missed approvals** — one more than Decision Tree's 5
- Total errors: **21 out of 854** (2.5% error rate)

### Why Decision Tree outperforms Random Forest here

This is a noteworthy result — an ensemble of many trees is normally expected to
outperform a single Decision Tree. Here the opposite is true. CIBIL score creates a
near-perfect linear threshold in this dataset (e.g. score above a cutoff → approve).
A single Decision Tree can express this boundary in one clean split. Random Forest's
averaging across many trees actually smooths this sharp boundary, introducing 8 extra
false approvals compared to the single tree. When data has clean, axis-aligned decision
boundaries, simpler models can and do win over ensembles.

---

## Key Insights

**CIBIL score is the dominant feature.** Credit score creates a near-perfect decision
boundary that drives the vast majority of predictions. This mirrors real-world banking
practice where CIBIL score is the first criterion any lender evaluates.

**Income-to-loan ratio matters more than raw income.** An applicant earning ₹10L
requesting ₹2L is a better risk than one earning ₹50L requesting ₹48L. The ratio of
`income_annum` to `loan_amount` is a stronger signal than either feature alone —
a feature that could be explicitly engineered in future iterations.

**Asset backing provides a strong collateral signal.** Both `Movable_assets` and
`Immovable_assets` contribute to approval — they represent recoverable value in the
event of default. Applicants with significant immovable assets (property) show
consistently higher approval rates.

**Clean decision boundaries enable simple models to win.** Decision Tree (99%) beats
Random Forest (98%) here — a result worth highlighting in interviews because it
demonstrates understanding of *when* and *why* ensembles don't always win.

---

## API Reference

The FastAPI app (`main.py`) exposes two endpoints:

### `GET /`

Health check — confirms the API is running.

```json
{ "message": "Loan Approval Prediction API is running 🚀" }
```

### `POST /predict`

Predict loan approval for a single applicant.

**Request body:**

```json
{
  "no_of_dependents": 2,
  "education": "Graduate",
  "self_employed": "No",
  "income_annum": 500000,
  "loan_amount": 200000,
  "loan_term": 120,
  "cibil_score": 750,
  "Movable_assets": 150000,
  "Immovable_assets": 500000
}
```

**Response:**

```json
{
  "loan_status": "Approved",
  "approval_probability": 0.9245
}
```

**Validation — all fields enforced by Pydantic. Invalid inputs return HTTP 422:**

| Field | Constraint |
|---|---|
| `no_of_dependents` | int, 0–10 |
| `education` | "Graduate" / "Not Graduate" |
| `self_employed` | "Yes" / "No" |
| `income_annum` | float, ≥ 0 |
| `loan_amount` | float, ≥ 0 |
| `loan_term` | int, 6–360 (months) |
| `cibil_score` | int, 300–900 |
| `Movable_assets` | float, ≥ 0 |
| `Immovable_assets` | float, ≥ 0 |

The interactive Swagger UI is available at `/docs` and ReDoc at `/redoc` when the
API is running — both are auto-generated by FastAPI with no extra setup needed.

---

## Project Structure

```
Loan_Approval/
│
├── Loan_Approval.ipynb          # Main notebook — EDA, training, evaluation
├── loan_approval_dataset.csv    # Dataset
├── loan_pipeline_model.pkl      # Trained sklearn pipeline (preprocessing + Decision Tree)
├── main.py                      # FastAPI application
├── Dockerfile                   # Docker image definition
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- Docker (optional — for containerized deployment)

### Install dependencies

```bash
git clone https://github.com/itsShivam73/Loan_Approval.git
cd Loan_Approval
pip install -r requirements.txt
```

---

## How to Run

### Run the API locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open your browser at:

- **API root:** `http://localhost:8000`
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### Test the API with curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "no_of_dependents": 1,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 800000,
    "loan_amount": 300000,
    "loan_term": 180,
    "cibil_score": 780,
    "Movable_assets": 200000,
    "Immovable_assets": 800000
  }'
```

### Run the notebook

Open `Loan_Approval.ipynb` in Jupyter or Google Colab and run cells top to bottom.
The notebook covers EDA, preprocessing, model training, confusion matrix visualization,
actual vs fitted density plots, and pipeline export.

---

## Docker Deployment

### Pull from Docker Hub

```bash
docker pull itsshivaam/loan-approval
docker run -p 8000:8000 itsshivaam/loan-approval
```

> Docker Hub: [hub.docker.com/u/itsshivaam](https://hub.docker.com/u/itsshivaam)

### Build locally

```bash
docker build -t loan-approval .
docker run -p 8000:8000 loan-approval
```

---

## Key Design Decisions

### Why Decision Tree over Random Forest as the final model?
Despite Random Forest being an ensemble, Decision Tree achieves higher accuracy
(99% vs 98%) on this dataset. CIBIL score creates a near-perfect linear threshold
that a single tree captures in one clean split. Random Forest's averaging smooths this
sharp boundary, introducing 8 extra false approvals (15 vs 7). When the dataset has
clean, axis-aligned decision boundaries, simpler models can outperform ensembles.

### Why a sklearn Pipeline for `loan_pipeline_model.pkl`?
Packaging preprocessing and the model into one Pipeline eliminates training-serving
skew — the API always applies the exact same transformations used during training.
`main.py` makes a single `model.predict()` and `model.predict_proba()` call with no
manual preprocessing, making it impossible to introduce transformation bugs at serving time.

### Why Pydantic validation in the API?
All 9 input fields have explicit constraints enforced by Pydantic's `Field()`. A
`cibil_score` of 950, a string for `income_annum`, or a missing field all return
HTTP 422 with a clear error message before the model is ever called — preventing
garbage-in-garbage-out predictions in production.

### Why return `approval_probability` alongside the decision?
A binary label alone is not enough for a real banking system. The probability allows
downstream systems to apply custom decision thresholds, route borderline cases
(e.g. 0.45–0.55) to human review, and give applicants meaningful feedback on how
close their application was.

### Why `try/except` in the prediction endpoint?
The prediction route wraps the model call in a try/except block and raises
`HTTPException(status_code=500)` on failure — ensuring internal errors return a
clean JSON error response rather than a raw Python traceback to the caller.

---

## Known Limitations

- Dataset is a structured benchmark — real-world bank data would include transaction
  history, existing debt load, employment duration, and many more features
- No threshold tuning applied — the default 0.5 probability cutoff is used; in
  production banking, the asymmetric cost of false approvals vs false rejections
  would justify a custom threshold
- CIBIL score dominates predictions — the model is highly reliant on a single feature,
  which could be a fairness concern if CIBIL scores carry socioeconomic bias
- No model explainability layer — SHAP values would show which features drove each
  individual prediction, required for RBI regulatory compliance in Indian banking

---

## Future Improvements

- [ ] **Threshold tuning** — analyze the cost of false approvals (risky loans) vs false
  rejections (lost business) and tune the probability cutoff for maximum business value
- [ ] **SHAP values** — add per-prediction explainability; critical for regulatory
  compliance in financial services
- [ ] **Feature engineering** — explicitly add `income_annum / loan_amount`
  (income-to-loan ratio) and `total_assets / loan_amount` (asset coverage ratio)
- [ ] **Streamlit UI** — build a user-friendly frontend for loan officers to interact
  with the model without using the raw API or Swagger UI directly
- [ ] **Cross-validation** — add k-fold CV to confirm 99% accuracy is stable across
  different data splits and not a result of a lucky test split
- [ ] **CI/CD pipeline** — automate model retraining and Docker image rebuild on
  new data using GitHub Actions

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| scikit-learn 1.6.1 | Preprocessing, pipeline, Decision Tree, Random Forest |
| pandas / numpy | Data manipulation and analysis |
| matplotlib / seaborn | Confusion matrix and density plot visualization |
| FastAPI | Production REST API |
| Pydantic | Input validation and schema enforcement |
| uvicorn | ASGI server for FastAPI |
| joblib | Model serialization (`loan_pipeline_model.pkl`) |
| Docker | Containerization and portable deployment |
| Jupyter Notebook | EDA, model training, and evaluation |

---

## Author

**Shivam Pandey**
Data Science Student | Machine Learning Enthusiast

---

## License

This project is licensed under the MIT License.
