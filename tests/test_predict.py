from __future__ import annotations

import json

import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.predict import load_model_artifacts, predict_supply_stress


def make_observations(days: int = 10) -> list[dict]:
    """Same fixture shape used in tests/test_api.py, kept in sync deliberately."""
    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    rows = []
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


# --- load_model_artifacts -------------------------------------------------


def test_load_model_artifacts_missing_directory_raises(tmp_path) -> None:
    empty_dir = tmp_path / "no_models_here"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_model_artifacts(models_dir=empty_dir)


def test_load_model_artifacts_returns_expected_types() -> None:
    model, feature_names, config = load_model_artifacts(models_dir="models")
    assert isinstance(model, XGBClassifier)
    assert isinstance(feature_names, list)
    assert all(isinstance(name, str) for name in feature_names)
    assert isinstance(config, dict)
    assert "default_threshold" in config


def test_load_model_artifacts_falls_back_to_pickle(tmp_path, monkeypatch) -> None:
    """If only the .pkl artifact exists, it should load without error."""
    import joblib
    import shutil

    fallback_dir = tmp_path / "pickle_only"
    fallback_dir.mkdir()

    # Reuse the real trained model, just re-saved as a pickle to exercise
    # the fallback branch without needing a second real model artifact.
    model, feature_names, config = load_model_artifacts(models_dir="models")
    joblib.dump(model, fallback_dir / "final_xgboost_supply_stress.pkl")
    shutil.copy("models/model_features.json", fallback_dir / "model_features.json")
    shutil.copy("models/model_config.json", fallback_dir / "model_config.json")

    loaded_model, loaded_features, loaded_config = load_model_artifacts(
        models_dir=fallback_dir
    )
    assert isinstance(loaded_model, XGBClassifier)
    assert loaded_features == feature_names
    assert loaded_config == config


# --- predict_supply_stress -------------------------------------------------


def test_predict_supply_stress_happy_path() -> None:
    data = pd.DataFrame(make_observations(days=10))
    results = predict_supply_stress(data, models_dir="models")

    assert not results.empty
    assert "stress_probability" in results.columns
    assert "stress_prediction" in results.columns
    assert "risk_label" in results.columns

    assert results["stress_probability"].between(0, 1).all()
    assert set(results["stress_prediction"].unique()).issubset({0, 1})
    assert set(results["risk_label"].unique()).issubset({"No Stress", "Stress Risk"})


def test_predict_supply_stress_respects_explicit_threshold() -> None:
    data = pd.DataFrame(make_observations(days=10))

    permissive = predict_supply_stress(data, models_dir="models", threshold=0.0)
    strict = predict_supply_stress(data, models_dir="models", threshold=1.0)

    # threshold=0.0 should flag everything as stress risk,
    # threshold=1.0 should flag nothing, given probabilities are in [0, 1).
    assert (permissive["stress_prediction"] == 1).all()
    assert (strict["stress_prediction"] == 0).all()


def test_predict_supply_stress_raises_on_insufficient_history() -> None:
    # Only 2 days of history: not enough for 7-day lag/rolling features.
    data = pd.DataFrame(make_observations(days=2))
    with pytest.raises(ValueError, match="enough history"):
        predict_supply_stress(data, models_dir="models")


def test_predict_supply_stress_uses_config_default_threshold_when_unset() -> None:
    data = pd.DataFrame(make_observations(days=10))
    _, _, config = load_model_artifacts(models_dir="models")

    default_result = predict_supply_stress(data, models_dir="models")
    explicit_result = predict_supply_stress(
        data, models_dir="models", threshold=float(config["default_threshold"])
    )

    pd.testing.assert_series_equal(
        default_result["stress_prediction"], explicit_result["stress_prediction"]
    )
