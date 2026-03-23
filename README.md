# NoShowShield

A machine learning system for hotel booking cancellation prediction and overbooking optimization. NoShowShield helps hotels maximize revenue by predicting which bookings are likely to cancel and recommending the optimal number of additional bookings to accept — while keeping relocation risk within acceptable bounds.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Streamlit Dashboard](#streamlit-dashboard)
  - [REST API](#rest-api)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Data Pipeline](#data-pipeline)
- [Model Details](#model-details)
- [Deployment](#deployment)
- [Testing](#testing)

---

## Overview

Hotel no-shows and cancellations represent a significant source of lost revenue. NoShowShield addresses this with two complementary capabilities:

1. **Cancellation Prediction** — An XGBoost classifier trained on historical booking data predicts the probability that any given booking will be canceled, with model-level and booking-level explanations powered by SHAP values.

2. **Overbooking Optimization** — Given a set of predicted cancellation probabilities for an arrival date and room type, a Poisson-Binomial optimizer computes the exact probability distribution of show-ups and recommends the number of extra bookings to accept. The recommendation maximizes expected net revenue while keeping relocation probability below a configurable threshold.

---

## Features

- Predict individual booking cancellation probabilities
- Recommend optimal overbooking levels per room type and arrival date
- Configurable relocation cost and risk tolerance
- SHAP-based local explanations (which features drove a specific booking's risk)
- SHAP-based global explanations (most influential features across all bookings on a given date)
- Interactive Streamlit dashboard for business users
- FastAPI REST API for programmatic access
- Dockerized for local and cloud deployment

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        Interfaces                              │
│          Streamlit Dashboard          FastAPI REST API         │
└───────────────────────┬────────────────────────┬──────────────┘
                        │                        │
┌───────────────────────▼────────────────────────▼──────────────┐
│                    Inference Pipeline                          │
│  DataManager → FeatureEngineer → PreprocessorManager          │
│              → ModelManager → OverbookingOptimizer            │
│              → ExplainerManager                               │
└────────────────────────────────────────────────────────────────┘
```

**Tech Stack:**

| Layer | Technology |
|---|---|
| Language | Python 3.10.6 |
| ML Model | XGBoost |
| Preprocessing | scikit-learn ColumnTransformer |
| Explainability | SHAP |
| Optimization | Poisson-Binomial (exact, dynamic programming) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Model Persistence | joblib |
| Cloud | Google Cloud Run, Artifact Registry |
| Containerization | Docker |

---

## Project Structure

```
noshowshield/
├── app.py                    # Streamlit dashboard entry point
├── api/
│   ├── __init__.py
│   └── fast.py               # FastAPI application (5 endpoints)
├── eda_package/              # Core ML package
│   ├── data.py               # DataManager: load, clean, and split data
│   ├── features.py           # FeatureEngineer: derived feature creation
│   ├── preprocessor.py       # PreprocessorManager: sklearn pipeline
│   ├── model.py              # ModelManager: XGBoost training and inference
│   ├── optimiser.py          # OverbookingOptimizer: Poisson-Binomial logic
│   ├── explainer.py          # ExplainerManager: SHAP-based interpretability
│   ├── pipeline.py           # End-to-end inference pipeline
│   └── registry.py           # Global configuration constants
├── models/
│   ├── preprocessor.joblib   # Fitted sklearn ColumnTransformer
│   └── working_model.pkl     # Trained XGBoost classifier
├── raw_data/
│   └── hotel_bookings.csv    # Source dataset
├── tests/
│   └── package_tests.py      # Test suite
├── notebook/                 # Jupyter notebooks (EDA and experiments)
├── scripts/                  # Utility scripts
├── Dockerfile
├── Makefile
├── requirements.txt
├── requirements_dev.txt
└── setup.py
```

---

## Installation

### Prerequisites

- Python 3.10.6
- `pip`

### Steps

```bash
# Clone the repository
git clone <repository-url>
cd noshowshield

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

Or use the Makefile shortcuts:

```bash
make install_requirements
make install
```

### First-Time Setup

If the fitted preprocessor artifact does not exist, generate it before running the API or dashboard:

```bash
make create_preprocessor
```

This fits and saves the sklearn ColumnTransformer to `models/preprocessor.joblib`.

---

## Usage

### Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard opens in your browser and provides:

- **Sidebar controls** — Set relocation cost (€0–1000) and maximum acceptable relocation risk (0–10%)
- **Date and room type filters** — Narrow recommendations to specific arrival dates or room categories
- **Summary metrics** — Model accuracy, AUC, and aggregate expected revenue uplift
- **Recommendations table** — Per-group optimal overbooking levels, net benefit, and relocation probability
- **SHAP explanations** — Feature importance charts for selected dates

### REST API

```bash
# Start the server
make run_api
# or
uvicorn api.fast:app --reload --port 8000
```

Interactive API documentation is available at `http://localhost:8000/docs`.

---

## API Reference

### `GET /`

Health check.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /predict`

Predict the cancellation probability for a single booking.

**Request body** — `BookingInput` (27 fields):

```json
{
  "hotel": "City Hotel",
  "lead_time": 342,
  "arrival_date_year": 2017,
  "arrival_date_month": "July",
  "arrival_date_week_number": 27,
  "arrival_date_day_of_month": 1,
  "stays_in_weekend_nights": 0,
  "stays_in_week_nights": 2,
  "adults": 2,
  "children": 0,
  "babies": 0,
  "meal": "BB",
  "country": "PRT",
  "market_segment": "Direct",
  "distribution_channel": "Direct",
  "is_repeated_guest": 0,
  "previous_cancellations": 0,
  "previous_bookings_not_canceled": 0,
  "reserved_room_type": "C",
  "assigned_room_type": "C",
  "booking_changes": 3,
  "deposit_type": "No Deposit",
  "agent": 0,
  "company": 0,
  "days_in_waiting_list": 0,
  "customer_type": "Transient",
  "adr": 75.0,
  "required_car_parking_spaces": 0,
  "total_of_special_requests": 0
}
```

**Response:**
```json
{
  "prediction": 0,
  "cancellation_probability": 0.12
}
```

---

### `GET /optimise`

Compute overbooking recommendations across all arrival date / room type combinations in the dataset.

**Query parameters:**

| Parameter | Type | Description |
|---|---|---|
| `relocation_cost` | `float` | Cost per relocated guest in euros |
| `max_risk` | `float` | Maximum acceptable relocation probability (0–1) |

**Response:**
```json
{
  "model_info": {
    "model_type": "XGBClassifier",
    "model_params": { "...": "..." }
  },
  "metrics": {
    "accuracy": 0.79,
    "recall": 0.48,
    "precision": 0.67,
    "f1": 0.56,
    "auc": 0.81
  },
  "recommendations": [
    {
      "arrival_date": "2017-07-01",
      "assigned_room_type": "A",
      "capacity": 10,
      "total_bookings": 8,
      "expected_cancellations": 2.4,
      "expected_show_ups": 5.6,
      "recommended_extra": 3,
      "net_benefit": 150.25,
      "relocation_probability": 0.019
    }
  ]
}
```

---

### `POST /explain/local`

Return SHAP-based explanations for a single booking's cancellation probability.

**Request body:** Same as `/predict`.

**Response:**
```json
{
  "cancellation_probability": 0.35,
  "higher_cancellation_risk": [
    { "feature_group": "lead_time", "shap_value": 0.15 }
  ],
  "lower_cancellation_risk": [
    { "feature_group": "adr", "shap_value": -0.12 }
  ],
  "grouped_local_shap": [ "..." ]
}
```

---

### `GET /explain/global-by-date`

Return mean absolute SHAP values across all bookings on a given arrival date.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `selected_date` | `str` | — | Arrival date in `YYYY-MM-DD` format |
| `min_rows` | `int` | `5` | Minimum bookings required to compute explanations |

**Response:**
```json
{
  "selected_date": "2017-07-01",
  "n_bookings": 42,
  "message": null,
  "grouped_global_shap": [
    { "feature_group": "lead_time", "mean_abs_shap": 0.23 }
  ]
}
```

---

### `GET /explain/available-dates`

List all arrival dates present in the dataset along with their booking counts.

**Response:**
```json
{
  "dates": [
    { "arrival_date": "2017-07-01", "n_bookings": 42 },
    { "arrival_date": "2017-07-02", "n_bookings": 38 }
  ]
}
```

---

## Configuration

Global constants are defined in `eda_package/registry.py`:

| Constant | Default | Description |
|---|---|---|
| `DEFAULT_RELOCATION_COST` | `300` | Default cost per relocated guest (€) |
| `DEFAULT_MAX_RISK` | `0.02` | Default maximum relocation probability (2%) |
| `MAX_EXTRA_SWEEP` | `300` | Maximum number of extra bookings evaluated per group |
| `COUNTRY_LIMIT` | `30` | Minimum bookings for a country to have its own category |
| `SPLIT_YEAR` | `2017` | Year boundary for train/test split (prevents leakage) |

Environment variables for cloud deployment are stored in `.env` and `.env.yaml`:

| Variable | Description |
|---|---|
| `GCP_PROJECT` | Google Cloud project ID |
| `GCP_REGION` | Deployment region |
| `DOCKER_IMAGE_NAME` | Docker image name |
| `DOCKER_REPO_NAME` | Artifact Registry repository |
| `DOCKER_LOCAL_PORT` | Local Docker port (default: `8080`) |
| `GAR_MEMORY` | Cloud Run memory allocation (default: `4Gi`) |

---

## Data Pipeline

```
hotel_bookings.csv
    │
    ▼
DataManager          — load, deduplicate, fill nulls, parse dates,
                       group rare countries, train/test split (80/20)
    │
    ▼
FeatureEngineer      — create derived features:
                         room_type_mismatch (reserved ≠ assigned)
                         special_requests_per_guest
                         weekend_ratio
                       drop target and leakage columns
    │
    ▼
PreprocessorManager  — sklearn ColumnTransformer:
                         numerical  → RobustScaler + median imputation
                         binary     → pass-through
                         one-hot    → OneHotEncoder
                         ordinal    → OrdinalEncoder (months, meal types)
    │
    ▼
ModelManager         — XGBoost classifier → cancellation probabilities
    │
    ├──▶ ExplainerManager  — SHAP values (local and global)
    │
    ▼
OverbookingOptimizer — Poisson-Binomial PMF via dynamic programming
                       sweep 0…MAX_EXTRA_SWEEP extra bookings per group
                       maximize net_benefit subject to relocation_prob ≤ max_risk
    │
    ▼
Recommendations (arrival_date × room_type)
```

---

## Model Details

**Classifier:** XGBoost (`XGBClassifier`)

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 500 |
| `max_depth` | 12 |
| `learning_rate` | 0.03 |
| `scale_pos_weight` | 3 (handles class imbalance) |

**Benchmark performance on held-out test set:**

| Metric | Score |
|---|---|
| Accuracy | 0.79 |
| AUC | 0.81 |
| Precision | 0.67 |
| Recall | 0.48 |
| F1 | 0.56 |

**Overbooking math:** The optimizer uses the exact Poisson-Binomial distribution (computed via dynamic programming on individual Bernoulli probabilities) rather than a normal or Poisson approximation, giving precise relocation probability estimates even for small groups.

**Explainability:** SHAP values are computed using a kernel explainer wrapping `model.predict_proba()`. Features are grouped back to their original names (e.g., all one-hot columns for `hotel` are summed into a single `hotel` SHAP value) for readability.

---

## Deployment

### Docker (local)

```bash
# Build
make docker_build_local

# Run
make docker_run_local
# API available at http://localhost:8080/docs
```

### Google Cloud Run

```bash
# Build and push image to Artifact Registry
make docker_push

# Deploy to Cloud Run
make docker_deploy
```

The Cloud Run service is configured with 4 GiB of memory to accommodate model loading and SHAP computation.

---

## Testing

```bash
# Run the test suite
pytest tests/

# Or via Makefile
make test_structure
```

---

## Makefile Reference

| Command | Description |
|---|---|
| `make install_requirements` | Install Python dependencies |
| `make install` | Install package via pip |
| `make create_preprocessor` | Fit and save the sklearn preprocessor |
| `make run_api` | Start the FastAPI server on port 8000 |
| `make test_structure` | Run pytest |
| `make clean` | Remove build artifacts |
| `make docker_build_local` | Build Docker image locally |
| `make docker_run_local` | Run Docker container locally |
| `make docker_push` | Push image to GCP Artifact Registry |
| `make docker_deploy` | Deploy image to Google Cloud Run |
