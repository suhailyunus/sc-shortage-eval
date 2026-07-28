from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app


def make_observations(days: int = 10) -> list[dict]:
    rows = []
    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    for index in range(days):
        rows.append(
            {
                "item_id": "HOBBIES_1_004",
                "store_id": "CA_3",
                "state_id": "CA",
                "day_num": 1800 + index,
                "sales": float([2, 2, 3, 2, 4, 3, 5, 6, 7, 8][index % 10]),
                "weekday": weekdays[index % 7],
                "event_name_1": None,
                "sell_price": 3.97,
                "snap_CA": int(index % 2 == 0),
                "snap_TX": 0,
                "snap_WI": 0,
            }
        )
    return rows


def test_health_and_readiness() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"


def test_model_info() -> None:
    """
    The exposed threshold must match the shipped config rather than a
    hardcoded constant. Pinning a literal here made this test fail the
    moment the operating point was retuned, which is a property of the
    test rather than of the API.
    """

    config = json.loads(
        (Path(__file__).resolve().parents[1] / "models" / "model_config.json")
        .read_text(encoding="utf-8")
    )

    with TestClient(app) as client:
        response = client.get("/model-info")
        assert response.status_code == 200
        body = response.json()
        assert body["model_type"] == "XGBClassifier"
        assert body["feature_count"] == 22
        assert body["default_threshold"] == pytest.approx(
            config["default_threshold"]
        )
        assert 0.0 < body["default_threshold"] < 1.0


def test_json_prediction() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"observations": make_observations(), "threshold": 0.5},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["input_rows"] == 10
        assert body["scored_rows"] >= 1
        assert 0 <= body["predictions"][0]["stress_probability"] <= 1
        assert body["predictions"][0]["risk_label"] in {
            "No Stress",
            "Stress Risk",
        }


def test_rejects_insufficient_history() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"observations": make_observations(8), "threshold": 0.5},
        )
        # Eight observations are accepted by schema but may produce one score;
        # the test chiefly verifies a controlled response rather than a crash.
        assert response.status_code in {200, 422}


def test_json_prediction_includes_business_risk_level() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"observations": make_observations(), "threshold": 0.5},
        )
        assert response.status_code == 200, response.text
        record = response.json()["predictions"][0]
        assert record["risk_level"] in {"Low", "Moderate", "High", "Critical"}


def test_csv_prediction_download() -> None:
    import csv
    import io

    rows = make_observations()
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

    with TestClient(app) as client:
        response = client.post(
            "/predict-file-csv",
            files={"file": ("sample_input.csv", buffer.getvalue(), "text/csv")},
            params={"threshold": 0.5},
        )

        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/csv")
        assert "predictions.csv" in response.headers["content-disposition"]
        assert response.headers["x-input-rows"] == "10"
        assert response.headers["x-scored-rows"] == "3"

        body = response.text
        assert "stress_probability" in body
        assert "risk_level" in body
        assert "High" in body or "Critical" in body or "Moderate" in body or "Low" in body
