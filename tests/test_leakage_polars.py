"""
Leakage regression tests for the Polars port (src/polars_preprocess.py,
src/polars_features.py). Mirrors the checks already applied to the
pandas pipeline in test_leakage.py -- the port must satisfy the same
guarantees, not just produce plausible-looking output.
"""

import polars as pl
import pytest

from src.polars_preprocess import add_stress_target
from src.polars_features import create_model_features


def _toy_frame() -> pl.DataFrame:
    """
    Small synthetic item/store series with a known, hand-computable
    sales pattern, so rolling/lag values can be checked exactly rather
    than merely "looks reasonable".
    """
    days = list(range(1, 21))
    sales = [1, 2, 3, 4, 5, 6, 7, 100, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    return pl.DataFrame(
        {
            "item_id": ["ITEM_1"] * 20,
            "store_id": ["STORE_1"] * 20,
            "state_id": ["CA"] * 20,
            "day_num": days,
            "sales": sales,
            "weekday": ["Monday"] * 20,
            "event_name_1": [None] * 20,
            "sell_price": [5.0] * 20,
            "snap_CA": [0] * 20,
            "snap_TX": [0] * 20,
            "snap_WI": [0] * 20,
        }
    )


def test_rolling_features_exclude_current_day():
    """rolling_mean_7 / rolling_std_7 at day t must never include day t's
    own sales value -- this is the exact bug class that was originally
    fixed in the pandas pipeline (shift(1) before rolling)."""
    df = _toy_frame()
    featured = create_model_features(df)

    row = featured.filter(pl.col("day_num") == 15).row(0, named=True)
    # Manually compute the mean of days 8-14 (the 7 days strictly before day 15)
    prior_7 = df.filter((pl.col("day_num") >= 8) & (pl.col("day_num") <= 14))["sales"]
    expected_mean = prior_7.mean()

    assert row["rolling_mean_7"] == pytest.approx(expected_mean, abs=1e-6), (
        "rolling_mean_7 includes information not available before this day -- leakage."
    )

    # The spike at day 8 (sales=100) must be visible in the rolling window
    # starting day 9, but NOT bleed backward into day 8's own row.
    day_8_row = featured.filter(pl.col("day_num") == 8).row(0, named=True)
    day_9_row = featured.filter(pl.col("day_num") == 9).row(0, named=True)
    assert day_8_row["sales_lag_1"] != 100, "day 8's own spike leaked into its own lag_1 feature"
    assert day_9_row["sales_lag_1"] == 100, "day 9 should see day 8's spike via lag_1"


def test_stress_threshold_uses_train_period_only():
    """When threshold_cutoff_day is supplied, the stress threshold must
    be computed ONLY from on-or-before-cutoff history -- estimating it
    over the full series (including holdout) leaks future information
    into a label applied to the training period."""
    df = _toy_frame()
    long_format = df.with_columns(pl.lit("d_placeholder").alias("day"))

    cutoff_day = 10
    targeted = add_stress_target(
        long_format, quantile=0.90, grouping=("item_id",), threshold_cutoff_day=cutoff_day
    )

    train_only = df.filter(pl.col("day_num") <= cutoff_day)["sales"]
    expected_threshold = train_only.quantile(0.90)
    actual_threshold = targeted["stress_threshold"][0]

    assert actual_threshold == pytest.approx(expected_threshold, abs=1e-6), (
        "stress_threshold does not match a train-period-only quantile -- "
        "it may be leaking holdout-period sales into the label definition."
    )


def test_stress_threshold_none_cutoff_documented_as_leaky():
    """threshold_cutoff_day=None reproduces the original full-sample
    behaviour and is retained only for backward compatibility -- confirm
    it actually differs from the safe version, so nobody mistakes it for
    an equivalent, safe default."""
    df = _toy_frame()

    safe = add_stress_target(df, quantile=0.90, grouping=("item_id",), threshold_cutoff_day=10)
    unsafe = add_stress_target(df, quantile=0.90, grouping=("item_id",), threshold_cutoff_day=None)

    assert safe["stress_threshold"][0] != unsafe["stress_threshold"][0], (
        "Expected the train-only and full-sample thresholds to differ for this "
        "synthetic series (day 8's spike is outside the day<=10 window's tail "
        "influence at a 90th percentile) -- if they're equal, verify the test "
        "fixture still exercises the intended difference."
    )
