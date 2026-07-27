# 📦 Supply Chain Stress Prediction

An end-to-end machine learning application that predicts retail
supply-chain stress from historical sales observations. The project
demonstrates how predictive analytics can be operationalized through a
production-style REST API, browser-based interface, containerized
deployment, and automated CI pipeline.

[![Live
Demo](https://img.shields.io/badge/🌐-Live%20Demo-success?style=for-the-badge)](https://supply-chain-stress-prediction.streamlit.app/)
[![API
Docs](https://img.shields.io/badge/📘-API%20Documentation-blue?style=for-the-badge)](https://supply-chain-stress-api.onrender.com/docs)
![CI](https://img.shields.io/github/actions/workflow/status/suhailyunus/supply-chain-stress-prediction/ci.yml?branch=main&style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge)

------------------------------------------------------------------------

# 🚀 Try the Application

🌐 **Live Web Application**

https://supply-chain-stress-prediction.streamlit.app/

📘 **Interactive API Documentation**

https://supply-chain-stress-api.onrender.com/docs

------------------------------------------------------------------------

# 📸 Application Preview

## Home Screen

![Home Screen](docs/images/homepage.png)

------------------------------------------------------------------------

## Prediction Results

![Prediction Results](docs/images/app-preview.png)

------------------------------------------------------------------------

# 💼 Business Problem

Retail organizations generate millions of historical sales observations,
making it difficult to identify emerging supply-chain risks before
shortages occur.

This project demonstrates how machine learning can transform historical
retail data into actionable, business-ready predictions.

Users can upload historical retail observations, score supply-chain
stress using a deployed REST API, visualize prediction results through a
web interface, and download business-friendly prediction reports.

------------------------------------------------------------------------

# 🏗 Architecture

``` text
                CSV Upload
                     │
                     ▼
          Streamlit Web Interface
                     │
               HTTPS Requests
                     │
                     ▼
             FastAPI REST API
                     │
                     ▼
      Feature Engineering Pipeline
                     │
                     ▼
         XGBoost Classification Model
                     │
                     ▼
     Business-Friendly Risk Predictions
                     │
                     ▼
          Downloadable Prediction CSV
```

------------------------------------------------------------------------

# ✨ Features

-   Upload historical retail data through a browser
-   Preview uploaded observations
-   Predict supply-chain stress using XGBoost
-   Business-friendly risk classifications
-   Interactive Streamlit dashboard
-   Download prediction reports as CSV
-   REST API for integration
-   Dockerized deployment
-   Automated testing with GitHub Actions

------------------------------------------------------------------------

# 🛠 Technology Stack

  Category           Technology
  ------------------ -------------------------
  Language           Python
  Machine Learning   XGBoost
  Data Processing    Pandas
  API                FastAPI
  Frontend           Streamlit
  Testing            Pytest
  Containerization   Docker & Docker Compose
  CI                 GitHub Actions

------------------------------------------------------------------------

# 📁 Repository Structure

``` text
api/            FastAPI inference service
frontend/       Streamlit web application
src/            Training and inference pipeline
models/         Trained ML model artifacts
tests/          Automated test suite
examples/       Sample input CSV files
configs/        Configuration files
paper_draft/    Research paper
reports/        Generated outputs
```

------------------------------------------------------------------------

# ▶ Running Locally

Clone the repository

``` bash
git clone https://github.com/suhailyunus/supply-chain-stress-prediction.git
```

Start the application

``` bash
docker compose up --build
```

Open:

**Web Application**

http://localhost:8501

**API Documentation**

http://localhost:8000/docs

------------------------------------------------------------------------

# ⚙ Engineering Decisions

-   **FastAPI** provides a lightweight, high-performance REST API for
    model serving.
-   **Streamlit** enables rapid development of an interactive user
    interface for business users.
-   **Docker** ensures reproducible environments across development and
    deployment.
-   **GitHub Actions** automatically validates the application on every
    push.
-   **Shared feature engineering** ensures inference uses the same
    transformations as model training.

------------------------------------------------------------------------

# 🚧 Roadmap

Future improvements include:

-   Azure deployment
-   MLflow experiment tracking
-   Model monitoring
-   Authentication
-   Scheduled model retraining
-   Batch inference support

------------------------------------------------------------------------

# 📜 License

This project was developed as a portfolio project demonstrating the
end-to-end lifecycle of a machine learning application, from feature
engineering and model development through deployment and continuous
integration.
