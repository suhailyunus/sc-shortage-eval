from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def build_synthetic_m5(tmp_path, n_items: int = 3, n_stores: int = 2, n_days: int = 120):
    """
    Write minimal but schema-correct M5 sales/calendar/prices CSVs to
    tmp_path, matching the raw wide-format contract that load_m5_data,
    reshape_sales_long, merge_calendar, and merge_prices all expect.

    Shared across test modules that need a real (synthetic) end-to-end
    dataset rather than mocks, so pipeline- and evaluation-level tests
    exercise the same realistic shapes.
    """
    rng = np.random.default_rng(0)

    item_ids = [f"HOBBIES_1_{i:03d}" for i in range(n_items)]
    store_ids = [f"CA_{s}" for s in range(1, n_stores + 1)]
    day_cols = [f"d_{d}" for d in range(1, n_days + 1)]

    sales_rows = []
    for item in item_ids:
        for store in store_ids:
            # Mostly low sales with occasional spikes so the 90th
            # percentile stress target has both classes represented.
            base = rng.poisson(3, size=n_days).astype(float)
            spikes = rng.random(n_days) < 0.1
            base[spikes] += rng.integers(8, 15, size=spikes.sum())
            row = {
                "id": f"{item}_{store}_validation",
                "item_id": item,
                "dept_id": "HOBBIES_1",
                "cat_id": "HOBBIES",
                "store_id": store,
                "state_id": store.split("_")[0],
            }
            row.update({col: val for col, val in zip(day_cols, base)})
            sales_rows.append(row)
    sales = pd.DataFrame(sales_rows)

    weekdays = [
        "Sunday", "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday",
    ]
    dates = pd.date_range("2015-01-04", periods=n_days, freq="D")
    calendar = pd.DataFrame(
        {
            "date": dates.astype(str),
            "wm_yr_wk": [11101 + (d // 7) for d in range(n_days)],
            "weekday": [weekdays[d % 7] for d in range(n_days)],
            "wday": [(d % 7) + 1 for d in range(n_days)],
            "month": dates.month,
            "year": dates.year,
            "d": day_cols,
            "event_name_1": [
                "SuperBowl" if d % 37 == 0 else None for d in range(n_days)
            ],
            "event_type_1": [
                "Sporting" if d % 37 == 0 else None for d in range(n_days)
            ],
            "event_name_2": [None] * n_days,
            "event_type_2": [None] * n_days,
            "snap_CA": (rng.random(n_days) < 0.3).astype(int),
            "snap_TX": (rng.random(n_days) < 0.3).astype(int),
            "snap_WI": (rng.random(n_days) < 0.3).astype(int),
        }
    )

    weeks = sorted(calendar["wm_yr_wk"].unique())
    price_rows = []
    for item in item_ids:
        for store in store_ids:
            base_price = round(rng.uniform(2.0, 8.0), 2)
            for week in weeks:
                price_rows.append(
                    {
                        "store_id": store,
                        "item_id": item,
                        "wm_yr_wk": week,
                        "sell_price": base_price,
                    }
                )
    prices = pd.DataFrame(price_rows)

    data_dir = tmp_path / "m5_synthetic"
    data_dir.mkdir()
    sales.to_csv(data_dir / "sales_train_validation.csv", index=False)
    calendar.to_csv(data_dir / "calendar.csv", index=False)
    prices.to_csv(data_dir / "sell_prices.csv", index=False)

    return data_dir


@pytest.fixture
def synthetic_m5_dir(tmp_path):
    """A ready-to-use directory of synthetic M5-shaped CSVs."""
    return build_synthetic_m5(tmp_path)


@pytest.fixture
def trained_pipeline_result(tmp_path):
    """A full TrainingResult from run_training_pipeline on synthetic data.

    Shared by evaluate.py tests that need a real fitted model, X_test,
    and y_test rather than hand-rolled mocks.
    """
    from src.pipeline import run_training_pipeline

    data_dir = build_synthetic_m5(tmp_path)
    return run_training_pipeline(data_dir, max_items=None, train_fraction=0.75)
