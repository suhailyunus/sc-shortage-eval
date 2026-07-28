from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.preprocess import find_split_day


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    day_num: pd.Series,
    *,
    train_fraction: float = 0.80,
    split_day: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split into a past training period and a future holdout period.

    The split is performed on ``day_num``, never on row position. The
    analytical table is sorted by ``item_id`` before ``day_num``, so a
    positional split would partition the data by item rather than by
    time and would leave train and test spanning the same calendar
    range. Every training observation is guaranteed to precede every
    test observation.

    Parameters
    ----------
    X, y
        Feature matrix and target, sharing an index.
    day_num
        Integer day index aligned to ``X``. Must share ``X``'s index.
    train_fraction
        Approximate share of observations assigned to training. Ignored
        when ``split_day`` is supplied.
    split_day
        Explicit last training day. Supply this to reuse an identical
        boundary across experiments.
    """

    if not X.index.equals(y.index):
        raise ValueError("X and y must share an identical index.")
    if not X.index.equals(day_num.index):
        raise ValueError("day_num must share an identical index with X.")

    cutoff = (
        int(split_day)
        if split_day is not None
        else find_split_day(day_num, train_fraction=train_fraction)
    )

    train_mask = day_num <= cutoff
    test_mask = ~train_mask

    if not train_mask.any():
        raise ValueError(f"No observations fall on or before day {cutoff}.")
    if not test_mask.any():
        raise ValueError(f"No observations fall after day {cutoff}.")

    return (
        X.loc[train_mask].copy(),
        X.loc[test_mask].copy(),
        y.loc[train_mask].copy(),
        y.loc[test_mask].copy(),
    )


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """Calculate the majority-to-minority class ratio."""

    negative_count = int((y == 0).sum())
    positive_count = int((y == 1).sum())

    if positive_count == 0:
        raise ValueError("The positive class is absent from the training data.")

    return negative_count / positive_count


def build_random_forest(**overrides: Any) -> RandomForestClassifier:
    """Create the tuned Random Forest benchmark."""

    params: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(overrides)
    return RandomForestClassifier(**params)


def build_xgboost(
    *,
    scale_pos_weight: float,
    **overrides: Any,
) -> XGBClassifier:
    """Create the selected class-balanced XGBoost model."""

    params: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.03,
        "min_child_weight": 3,
        "gamma": 0,
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(overrides)
    return XGBClassifier(**params)


def train_final_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """Fit the selected model using class imbalance from the training set."""

    model = build_xgboost(
        scale_pos_weight=calculate_scale_pos_weight(y_train)
    )
    model.fit(X_train, y_train)
    return model


def save_model_artifacts(
    model: Any,
    feature_names: list[str],
    output_dir: str | Path,
    *,
    default_threshold: float = 0.50,
    alternative_threshold: float | None = None,
) -> None:
    """Persist the model, feature schema, and deployment configuration."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        directory / "final_xgboost_supply_stress.pkl",
    )

    # Native XGBoost format is more portable across library versions
    # than Python pickle serialization and is preferred by the API.
    if hasattr(model, "save_model"):
        model.save_model(
            directory / "final_xgboost_supply_stress.ubj"
        )

    (directory / "model_features.json").write_text(
        json.dumps(feature_names, indent=2),
        encoding="utf-8",
    )

    config = {
        "model_type": type(model).__name__,
        "positive_class": 1,
        "positive_class_label": "Supply Stress",
        "default_threshold": default_threshold,
        "alternative_threshold": alternative_threshold,
    }
    (directory / "model_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
