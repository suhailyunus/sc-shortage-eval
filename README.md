<p align="center">
  <img src="reports/figures/readme_banner.png" alt="Supply Chain Stress Prediction" width="100%">
</p>

<p align="center">
  <strong>End-to-end machine learning pipeline for early detection of retail demand stress</strong>
</p>

## 🚀 Live Demo

<p align="center">

<a href="https://supply-chain-stress-prediction.streamlit.app">
<img src="https://img.shields.io/badge/🌐-Launch%20Web%20Application-success?style=for-the-badge">
</a>

<a href="https://supply-chain-stress-api.onrender.com/docs">
<img src="https://img.shields.io/badge/📘-Interactive%20API%20Docs-blue?style=for-the-badge">
</a>

<a href="https://github.com/suhailyunus/supply-chain-stress-prediction/actions">
<img src="https://img.shields.io/github/actions/workflow/status/suhailyunus/supply-chain-stress-prediction/ci.yml?branch=main&style=for-the-badge">
</a>

</p>

**Try it yourself**

1. Open the Streamlit application.
2. Upload `examples/sample_input.csv`.
3. Click **Run Prediction**.
4. Review the dashboard.
5. Download `predictions.csv`.



<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.13-blue">
  <img alt="XGBoost" src="https://img.shields.io/badge/Model-XGBoost-orange">
  <img alt="Validation" src="https://img.shields.io/badge/Validation-Chronological-success">
  <img alt="Explainability" src="https://img.shields.io/badge/Explainability-SHAP-purple">
  <img alt="Status" src="https://img.shields.io/badge/Status-Portfolio%20Ready-brightgreen">
</p>

> A production-oriented machine learning application demonstrating how historical retail demand signals can be transformed into an operational early-warning system through automated feature engineering, REST APIs, containerized deployment, and an interactive web interface.

---


## 📸 Application Preview

### Home Screen

<p align="center">
<img src="docs/images/homepage.png" width="100%">
</p>

### Prediction Results

<p align="center">
<img src="docs/images/app-preview.png" width="100%">
</p>

---

## Business Problem

Unexpected demand surges can create inventory pressure, stockouts, and lost sales. This project builds an **early-warning system** that ranks item-store observations by their probability of entering a high-demand state associated with potential supply stress.

The solution uses recent sales behavior, short-term volatility, calendar context, pricing, and location signals to produce a reusable risk score.

## Dataset

The project is built using a large-scale retail sales dataset containing historical product demand, pricing information, promotional calendar events, and multi-store geographic information.

These data sources provide a realistic environment for developing machine learning models that identify elevated demand conditions associated with potential supply chain stress.

The dataset is publicly available and widely used for benchmarking retail demand forecasting methods, making it suitable for demonstrating production-oriented machine learning workflows while maintaining reproducibility.

> **Target caveat:** The retail sales dataset used in this project does not contain verified inventory or stockout outcomes. To approximate supply stress, the target is defined as sales exceeding an item-specific 90th percentile demand threshold. The model therefore predicts a transparent high-demand proxy rather than confirmed stockout events.

## Results at a Glance

| Metric | Default Early-Warning Threshold |
|---|---:|
| Final model | XGBoost |
| Validation design | Chronological holdout |
| Accuracy | **0.67** |
| Stress precision | **0.14** |
| Stress recall | **0.61** |
| Stress F1 | **0.23** |
| Average Precision | **≈0.19** |

At the alternative F1-maximizing threshold (**≈0.64**), precision increases to roughly **0.21** and F1 to **0.26**, while recall falls to approximately **0.33**. The default threshold is retained because early detection is the principal business objective.

## Why This Project Stands Out

- **Production-style structure:** reusable logic lives in `src/`, not only in a notebook.
- **Leakage-safe temporal features:** lags and rolling statistics use prior observations only.
- **Item-store demand histories:** features are grouped by both item and store.
- **Chronological validation:** future observations are held out rather than randomly mixed.
- **Imbalance-aware modeling:** XGBoost uses training-period class weights.
- **Explainability:** SHAP shows the direction and magnitude of feature effects.
- **Operational thresholding:** the alert cutoff is tied to business trade-offs.
- **Repeatable inference:** saved model artifacts and feature metadata support future scoring.

## Architecture

<p align="center">
  <img src="reports/figures/pipeline_architecture.png" alt="Production ML Architecture" width="100%">
</p>

## Model Development Journey

<p align="center">
  <img src="reports/figures/model_development_timeline.png" alt="Model Development Timeline" width="100%">
</p>

## Evaluation Visuals

The `supply_stress_prediction_case_study.ipynb` notebook automatically generates the evaluation figures below during model validation.

### Precision–Recall Curve

<p align="center">
  <img src="reports/figures/precision_recall_curve.png" alt="Precision Recall Curve" width="72%">
</p>

### Confusion Matrix

<p align="center">
  <img src="reports/figures/confusion_matrix.png" alt="Confusion Matrix" width="65%">
</p>

### SHAP Summary

Save the final SHAP beeswarm as `reports/figures/shap_summary.png`, then it will render here:

<p align="center">
  <img src="reports/figures/shap_summary.png" alt="SHAP Summary" width="82%">
</p>

## Repository Structure

```text
.
├── data/raw/                                  # Retail sales, calendar, and pricing data
├── models/                                    # Generated model artifacts
├── notebooks/
│   ├── supply_stress_prediction_case_study.ipynb
│   └── archive/                               # Local drafts; ignored by Git
├── reports/figures/
│   ├── readme_banner.png
│   ├── pipeline_architecture.png
│   ├── model_development_timeline.png
│   ├── precision_recall_curve.png
│   ├── confusion_matrix.png
│   └── shap_summary.png
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── pipeline.py
├── project_notes.md
├── requirements.txt
└── README.md
```

## Feature Engineering

The final feature set includes:

- one-day and seven-day sales lags;
- seven-day rolling mean;
- seven-day rolling standard deviation;
- weekend and event-day indicators;
- state-specific SNAP indicators;
- current selling price and recent price movement;
- one-hot encoded state and store variables.

All demand features are isolated by `item_id` and `store_id`, and rolling features use `shift(1)` to prevent current-row leakage.

## Modeling Approach

1. Reshape retail sales history into an analytical long-format dataset.
2. Construct an item-relative high-demand target.
3. Enrich observations with calendar and price context.
4. Engineer temporal and geographic features.
5. Benchmark Logistic Regression and Random Forest.
6. Tune Random Forest and evaluate feature importance.
7. validate using a chronological holdout.
8. Benchmark and tune class-balanced XGBoost.
9. Analyze precision-recall trade-offs and thresholds.
10. Persist the model and run reusable inference.

## Installation

```bash
git clone <your-repository-url>
cd supply-chain-stress-prediction
python -m venv .venv
source .venv/bin/activate  # Windows Git Bash
pip install -r requirements.txt
```

Place the retail sales data files inside `data/raw/`:

```text
sales_train_validation.csv
calendar.csv
sell_prices.csv
```

## Train the Model

```python
from src.pipeline import run_training_pipeline

result = run_training_pipeline(
    data_dir="data/raw",
    max_items=100,
    models_dir="models",
)
```

## Run Inference

Inference requires recent historical observations because lag and rolling features cannot be calculated from an isolated row.

```python
from src.predict import predict_supply_stress

predictions = predict_supply_stress(
    recent_data,
    models_dir="models",
    threshold=0.50,
)
```

Example output:

| item_id | store_id | day_num | stress_probability | risk_label |
|---|---|---:|---:|---|
| HOBBIES_1_004 | CA_3 | 1898 | 0.941 | Stress Risk |
| HOBBIES_1_046 | CA_3 | 1913 | 0.935 | Stress Risk |
| HOBBIES_1_023 | TX_2 | 1913 | 0.931 | Stress Risk |

## Key Findings

- Selling price was consistently one of the strongest predictive signals.
- Weekends increased predicted stress risk.
- Recent demand level and volatility were operationally meaningful.
- Store-level variation was stronger than state-level variation.
- An unbalanced XGBoost model produced high accuracy but almost no stress detection.
- Class balancing improved stress recall to approximately 61%.
- Threshold changes altered business behavior without retraining the model.

## Engineering Lessons

- Accuracy can be misleading for rare-event classification.
- Time-aware validation is essential for future-facing decisions.
- Training and inference should share the same feature code.
- Feature schemas and operating thresholds should be saved with the model.
- A model can be most useful as a prioritized review queue rather than an autonomous decision-maker.

## Limitations

- The target is a proxy rather than a verified inventory outcome.
- The local workflow samples the first 100 items for computational efficiency.
- Inventory position, replenishment schedules, supplier lead times, weather, and logistics disruptions are not included.
- Probability calibration has not yet been performed.
- The alternative threshold was explored on the holdout and should be selected on a separate validation period in production.

## Future Work

- Replace the proxy target with verified inventory or stockout outcomes.
- Integrate inventory position, supplier lead times, and replenishment schedules.
- Deploy to Azure using managed container services.
- Add MLflow experiment tracking and model registry.
- Implement model drift monitoring and automated retraining.
- Introduce authentication and role-based access control.
- Support batch inference and scheduled scoring workflows.

## Notebook

The full analytical narrative is available in:

```text
notebooks/supply_stress_prediction_case_study.ipynb
```

The notebook documents the complete analytical workflow, from data preparation and feature engineering through model development, explainability, threshold selection, and production inference. It serves as the technical case study accompanying the modular Python implementation contained within the `src/` package.


## Acknowledgements

This project uses a publicly available retail sales forecasting dataset released for academic benchmarking. The engineering workflow, feature engineering strategy, model development, evaluation methodology, production pipeline, and documentation were developed independently as part of this portfolio project.

---

<p align="center">
  <strong>Built as a production-oriented machine learning portfolio project.</strong>
</p>

## Containerized Prediction API

The final model is exposed through a production-style FastAPI service and packaged as a hardened Docker container. The API loads the saved XGBoost model once at startup, reuses the same feature-engineering code as training, validates incoming history, and returns probability-based risk labels.

### API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness and model-load status |
| `/ready` | GET | Readiness check used by Docker |
| `/model-info` | GET | Model type, thresholds, and feature schema |
| `/predict` | POST | Score JSON historical observations |
| `/predict-file` | POST | Upload and score a CSV file |
| `/docs` | GET | Interactive Swagger documentation |

### Build and run with Docker Compose

```bash
docker compose up --build
```

Open the interactive API documentation:

```text
http://localhost:8000/docs
```

Check service readiness:

```bash
curl http://localhost:8000/ready
```

Submit the included example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @examples/sample_request.json
```

Stop the service:

```bash
docker compose down
```

### Container security and reproducibility

- Multi-stage image build separates dependency installation from runtime.
- The service runs as a non-root user.
- The container filesystem is read-only under Compose.
- Linux capabilities are dropped and privilege escalation is disabled.
- Model readiness is monitored through an HTTP health check.
- Runtime dependencies are isolated in `requirements-api.txt`.
- `.dockerignore` excludes notebooks, data, development caches, and other non-runtime files.

### Run API tests locally

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Download Business-Ready Predictions as CSV

The API includes a CSV-download endpoint for non-technical users:

```text
POST /predict-file-csv
```

From the Swagger interface at `http://localhost:8000/docs`:

1. Open **POST /predict-file-csv**.
2. Select **Try it out**.
3. Upload `examples/sample_input.csv`.
4. Leave the threshold blank to use the saved default of `0.50`.
5. Select **Execute**, then use the response **Download file** link.

The downloaded `predictions.csv` contains:

- item and store identifiers;
- model probability;
- binary stress prediction;
- operational label (`No Stress` or `Stress Risk`);
- business severity level (`Low`, `Moderate`, `High`, or `Critical`).

Risk bands are defined as:

| Probability | Business risk level |
|---:|---|
| `< 0.30` | Low |
| `0.30–0.59` | Moderate |
| `0.60–0.79` | High |
| `>= 0.80` | Critical |

## Web Interface

The project includes a Streamlit interface that sits in front of the FastAPI inference service. Users can upload a CSV, preview the input, run predictions, review business-friendly risk levels, inspect the highest-risk observations, and download `predictions.csv`.

Run both services with Docker Compose:

```bash
docker compose up --build
```

Open:

- Web application: `http://localhost:8501`
- FastAPI documentation: `http://localhost:8000/docs`

The frontend calls the API over the internal Docker network:

```text
Browser → Streamlit → FastAPI → feature engineering → XGBoost → predictions.csv
```

Use `examples/sample_input.csv` for a quick end-to-end test. Each item-store series needs at least eight chronological rows because the inference pipeline reconstructs lag and rolling features before scoring.

## Continuous Integration

The repository includes a GitHub Actions workflow at `.github/workflows/ci.yml`. It runs automatically for pushes and pull requests targeting `main`.

The workflow performs two stages:

1. **Python verification** — installs the development dependencies, compiles the Python modules, and runs the automated API tests.
2. **Container smoke test** — validates `compose.yaml`, builds both Docker images, starts the FastAPI and Streamlit services, and checks their health endpoints.

After pushing the workflow, open the repository's **Actions** tab to view the run. A successful run confirms that the tested code and both containerized services can start cleanly in a fresh environment.

> This workflow provides continuous integration. Continuous deployment will be added after selecting the public hosting target for the API and web interface.


---

## 🌐 Live Links

- **Web Application:** https://supply-chain-stress-prediction.streamlit.app/
- **Interactive API Documentation:** https://supply-chain-stress-api.onrender.com/docs
