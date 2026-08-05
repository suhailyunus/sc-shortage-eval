"""
Polars port of features.py.

The one line that matters most for leakage-safety is preserved exactly:
rolling mean/std are computed on sales *shifted by 1* within each
item+store group, so day t's rolling features never see day t's own
sales value. Same as the pandas .transform(lambda s: s.shift(1).rolling(7)...).
"""

from __future__ import annotations

import polars as pl

BASE_MODEL_FEATURES = [
    "sales_lag_1",
    "sales_lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "is_event_day",
    "is_weekend",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "sell_price",
    "price_change_1",
]


def create_model_features(data: pl.DataFrame) -> pl.DataFrame:
    """
    Create demand, calendar, price, and location features.

    Lag and rolling calculations are isolated by item and store (via
    .over(group_keys), which requires the frame to be pre-sorted by
    day_num within each group) to prevent demand histories from
    different stores being mixed.
    """
    required = {
        "item_id", "store_id", "state_id", "day_num", "sales", "weekday",
        "event_name_1", "sell_price", "snap_CA", "snap_TX", "snap_WI",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    group_keys = ["item_id", "store_id"]
    df = data.sort(group_keys + ["day_num"])

    shifted_sales = pl.col("sales").shift(1).over(group_keys)

    df = df.with_columns(
        [
            pl.col("sales").shift(1).over(group_keys).alias("sales_lag_1"),
            pl.col("sales").shift(7).over(group_keys).alias("sales_lag_7"),
            shifted_sales.rolling_mean(window_size=7).over(group_keys).alias("rolling_mean_7"),
            shifted_sales.rolling_std(window_size=7).over(group_keys).alias("rolling_std_7"),
            pl.col("weekday").is_in(["Saturday", "Sunday"]).cast(pl.Int8).alias("is_weekend"),
            pl.col("event_name_1").is_not_null().cast(pl.Int8).alias("is_event_day"),
        ]
    )

    df = df.with_columns(pl.col("sell_price").shift(1).over(group_keys).alias("price_lag_1"))
    df = df.with_columns(
        [
            (pl.col("sell_price") - pl.col("price_lag_1")).alias("price_change_1"),
            (
                (pl.col("sell_price") - pl.col("price_lag_1")) / pl.col("price_lag_1")
            ).alias("price_pct_change_1"),
        ]
    )
    df = df.with_columns((pl.col("price_pct_change_1") < 0).cast(pl.Int8).alias("is_discount"))

    location_dummies = df.select(["state_id", "store_id"]).to_dummies(
        columns=["state_id", "store_id"], drop_first=True
    )
    location_dummies = location_dummies.cast(pl.Int8)

    return pl.concat([df, location_dummies], how="horizontal_extend")


def infer_model_features(featured_data: pl.DataFrame) -> list[str]:
    """Return the standard numeric features plus encoded locations."""
    location_columns = sorted(
        c for c in featured_data.columns if c.startswith("state_id_") or c.startswith("store_id_")
    )
    return BASE_MODEL_FEATURES + location_columns


def prepare_model_input(
    data: pl.DataFrame,
    expected_features: list[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """
    Engineer features, align the schema, and remove incomplete rows.

    Missing dummy columns are added as zeros so small inference batches
    remain compatible with the training schema.
    """
    featured = create_model_features(data)

    feature_names = (
        list(expected_features) if expected_features is not None else infer_model_features(featured)
    )

    for column in feature_names:
        if column not in featured.columns:
            featured = featured.with_columns(pl.lit(0).alias(column))

    ready = featured.drop_nulls(subset=feature_names)
    X_ready = ready.select(feature_names)

    return ready, X_ready, feature_names
