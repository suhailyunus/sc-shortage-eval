"""
Regression tests for temporal validity.

Each test here corresponds to a claim the project makes publicly. If one
of these fails, the reported evaluation metrics are not trustworthy and
the README claims must be withdrawn before the results are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import create_model_features
from src.preprocess import add_stress_target, find_split_day
from src.train import chronological_split

FEATURE_COLUMNS = [
    "sales_lag_1",
    "sales_lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "price_change_1",
]

WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def build_panel(
    items: tuple[str, ...] = ("ITEM_A", "ITEM_B"),
    stores: tuple[str, ...] = ("CA_1", "TX_1"),
    days: int = 40,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a small deterministic item-store panel."""

    rng = np.random.default_rng(seed)
    rows = []
    for item in items:
        for store in stores:
            for day in range(1, days + 1):
                rows.append(
                    {
                        "item_id": item,
                        "store_id": store,
                        "state_id": store.split("_")[0],
                        "day_num": day,
                        "sales": float(rng.integers(0, 20)),
                        "weekday": WEEKDAYS[(day - 1) % 7],
                        "event_name_1": None,
                        "sell_price": 3.0 + 0.1 * (day % 5),
                        "snap_CA": int(day % 2 == 0),
                        "snap_TX": int(day % 3 == 0),
                        "snap_WI": 0,
                    }
                )
    return pd.DataFrame(rows)


def features_by_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Index engineered features by item, store, and day for comparison."""

    return (
        create_model_features(frame)
        .set_index(["item_id", "store_id", "day_num"])[FEATURE_COLUMNS]
        .sort_index()
    )


# --------------------------------------------------------------------
# Feature-level leakage
# --------------------------------------------------------------------


def test_future_sales_cannot_change_past_features():
    """
    Perturbing sales after day D must leave every feature on or before
    day D bit-for-bit identical. This is the strongest available check
    that lag and rolling windows look strictly backwards.
    """

    cutoff = 25
    original = build_panel()
    perturbed = original.copy()

    future = perturbed["day_num"] > cutoff
    perturbed.loc[future, "sales"] = perturbed.loc[future, "sales"] * 1000 + 7
    perturbed.loc[future, "sell_price"] = 99.9

    base = features_by_key(original)
    altered = features_by_key(perturbed)

    past = base.index.get_level_values("day_num") <= cutoff
    pd.testing.assert_frame_equal(base.loc[past], altered.loc[past])


def test_same_day_sales_excluded_from_rolling_window():
    """
    The rolling mean on day D must average days D-7 through D-1. A window
    that included day D would make the target partially self-predicting.
    """

    frame = build_panel(items=("ITEM_A",), stores=("CA_1",), days=20)
    featured = create_model_features(frame).sort_values("day_num")

    sales = featured["sales"].to_numpy()
    for position in range(7, len(featured)):
        expected = sales[position - 7 : position].mean()
        assert featured["rolling_mean_7"].iloc[position] == pytest.approx(expected)


def test_lags_come_from_the_same_item_and_store():
    """
    Demand history must not bleed across series. Changing one store's
    sales must not move another store's features for the same item.
    """

    original = build_panel()
    perturbed = original.copy()
    other = perturbed["store_id"] == "TX_1"
    perturbed.loc[other, "sales"] = perturbed.loc[other, "sales"] * 500 + 3

    base = features_by_key(original).xs("CA_1", level="store_id")
    altered = features_by_key(perturbed).xs("CA_1", level="store_id")

    pd.testing.assert_frame_equal(base, altered)


def test_first_observation_of_each_series_has_no_history():
    """A series' first day cannot have a lag, and must be dropped later."""

    featured = create_model_features(build_panel())

    # groupby.first() skips nulls, which is exactly what must not be used
    # here; select the earliest row per series by position instead.
    earliest = featured.loc[
        featured.groupby(["item_id", "store_id"])["day_num"].idxmin()
    ]

    assert earliest["sales_lag_1"].isna().all()
    assert earliest["rolling_mean_7"].isna().all()


# --------------------------------------------------------------------
# Target-level leakage
# --------------------------------------------------------------------


def test_stress_threshold_ignores_the_holdout_period():
    """
    The stress threshold is part of the label definition. Estimating it
    over the full history would define training labels using sales that
    had not yet happened, and could not be reproduced at scoring time.
    """

    cutoff = 30
    original = build_panel()
    perturbed = original.copy()
    future = perturbed["day_num"] > cutoff
    perturbed.loc[future, "sales"] = 10_000.0

    base = add_stress_target(original, threshold_cutoff_day=cutoff)
    altered = add_stress_target(perturbed, threshold_cutoff_day=cutoff)

    pd.testing.assert_series_equal(
        base["stress_threshold"],
        altered["stress_threshold"],
    )


def test_training_labels_are_stable_under_future_change():
    """Labels inside the training period must not move when the future does."""

    cutoff = 30
    original = build_panel()
    perturbed = original.copy()
    future = perturbed["day_num"] > cutoff
    perturbed.loc[future, "sales"] = 10_000.0

    base = add_stress_target(original, threshold_cutoff_day=cutoff)
    altered = add_stress_target(perturbed, threshold_cutoff_day=cutoff)

    past = base["day_num"] <= cutoff
    assert base.loc[past, "stress_event"].equals(altered.loc[past, "stress_event"])


def test_full_sample_threshold_is_detectably_leaky():
    """
    Guards the guard. If this ever passes, the leak-free path above has
    stopped being distinguishable from the legacy behaviour and the test
    above would no longer prove anything.
    """

    original = build_panel()
    perturbed = original.copy()
    future = perturbed["day_num"] > 30
    perturbed.loc[future, "sales"] = 10_000.0

    leaky_base = add_stress_target(original, threshold_cutoff_day=None)
    leaky_altered = add_stress_target(perturbed, threshold_cutoff_day=None)

    assert not leaky_base["stress_threshold"].equals(
        leaky_altered["stress_threshold"]
    )


# --------------------------------------------------------------------
# Split-level leakage
# --------------------------------------------------------------------


def test_every_training_day_precedes_every_test_day():
    """
    The core temporal guarantee. A positional split on the analytical
    table partitions by item instead of by time, because the table is
    sorted by item_id before day_num.
    """

    panel = build_panel(items=("ITEM_A", "ITEM_B", "ITEM_C"), days=50)
    panel = panel.sort_values(["item_id", "store_id", "day_num"]).reset_index(
        drop=True
    )

    X = panel[["sales", "sell_price"]]
    y = pd.Series(0, index=panel.index)

    X_train, X_test, _, _ = chronological_split(
        X, y, panel["day_num"], train_fraction=0.8
    )

    train_days = panel.loc[X_train.index, "day_num"]
    test_days = panel.loc[X_test.index, "day_num"]

    assert train_days.max() < test_days.min()
    assert not set(train_days).intersection(test_days)


def test_holdout_contains_every_series():
    """
    A time split must hold out the future of all series, not the entirety
    of some series. Each item-store pair should appear on both sides.
    """

    panel = build_panel(items=("ITEM_A", "ITEM_B", "ITEM_C"), days=50)
    panel = panel.sort_values(["item_id", "store_id", "day_num"]).reset_index(
        drop=True
    )

    X = panel[["sales"]]
    y = pd.Series(0, index=panel.index)
    X_train, X_test, _, _ = chronological_split(
        X, y, panel["day_num"], train_fraction=0.8
    )

    keys = lambda index: set(  # noqa: E731
        map(tuple, panel.loc[index, ["item_id", "store_id"]].to_numpy())
    )

    assert keys(X_train.index) == keys(X_test.index)


def test_positional_split_would_have_failed_this_suite():
    """
    Documents the defect this module was written to prevent. The legacy
    positional split leaves train and test spanning the same days.
    """

    panel = build_panel(items=("ITEM_A", "ITEM_B", "ITEM_C"), days=50)
    panel = panel.sort_values(["item_id", "store_id", "day_num"]).reset_index(
        drop=True
    )

    cut = int(len(panel) * 0.8)
    train_days = set(panel.iloc[:cut]["day_num"])
    test_days = set(panel.iloc[cut:]["day_num"])

    assert train_days.intersection(test_days), (
        "The positional split no longer overlaps in time; if the table's "
        "sort order changed, revisit why this test exists."
    )


def test_split_day_is_reproducible():
    """An explicit split_day must override the fraction exactly."""

    panel = build_panel(days=50)
    X = panel[["sales"]]
    y = pd.Series(0, index=panel.index)

    X_train, _, _, _ = chronological_split(X, y, panel["day_num"], split_day=17)
    assert panel.loc[X_train.index, "day_num"].max() == 17


def test_split_rejects_misaligned_index():
    """Silent index misalignment would reintroduce leakage invisibly."""

    panel = build_panel(days=20)
    X = panel[["sales"]]
    y = pd.Series(0, index=panel.index)
    shifted = panel["day_num"].reset_index(drop=True) + 0
    shifted.index = shifted.index + 5

    with pytest.raises(ValueError):
        chronological_split(X, y, shifted)


def test_split_rejects_empty_holdout():
    panel = build_panel(days=20)
    X = panel[["sales"]]
    y = pd.Series(0, index=panel.index)

    with pytest.raises(ValueError):
        chronological_split(X, y, panel["day_num"], split_day=9999)


def test_find_split_day_respects_the_requested_fraction():
    panel = build_panel(days=100)
    cutoff = find_split_day(panel["day_num"], train_fraction=0.8)
    share = (panel["day_num"] <= cutoff).mean()

    assert 0.75 <= share <= 0.85
