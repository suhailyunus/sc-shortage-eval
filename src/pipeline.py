from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.features import prepare_model_input
from src.load_data import load_m5_data
from src.preprocess import build_analytical_table
from src.train import (
    chronological_split,
    save_model_artifacts,
    train_final_xgboost,
)


@dataclass
class TrainingResult:
    analytical_data: pd.DataFrame
    feature_data: pd.DataFrame
    model_features: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    model: object
    split_day: int


def run_training_pipeline(
    data_dir: str | Path,
    *,
    max_items: int | None = 100,
    train_fraction: float = 0.80,
    models_dir: str | Path | None = None,
    default_threshold: float = 0.80,
) -> TrainingResult:
    """Load raw files, engineer features, train XGBoost, and optionally save it.

    ``default_threshold`` is the score cutoff written into the deployment
    config. It is a stated operating choice rather than an optimum: any
    cutoff asserts a cost ratio between false alarms and missed events,
    and no such ratio is available for this dataset. The default favours
    a reviewable alert volume over maximum recall.
    """

    raw = load_m5_data(data_dir)
    analytical, split_day = build_analytical_table(
        raw.sales,
        raw.calendar,
        raw.prices,
        max_items=max_items,
        train_fraction=train_fraction,
    )

    feature_data, X, feature_names = prepare_model_input(analytical)
    y = feature_data["stress_event"]

    # The same day boundary that defined the stress threshold also
    # defines the holdout, so neither the label nor the split can be
    # informed by observations the model has not seen.
    X_train, X_test, y_train, y_test = chronological_split(
        X,
        y,
        feature_data["day_num"],
        split_day=split_day,
    )

    model = train_final_xgboost(X_train, y_train)

    if models_dir is not None:
        save_model_artifacts(
            model,
            feature_names,
            models_dir,
            default_threshold=default_threshold,
        )

    return TrainingResult(
        analytical_data=analytical,
        feature_data=feature_data,
        model_features=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model=model,
        split_day=split_day,
    )
