"""
Regression test for a silent bug found during full-catalog validation:
one-hot encoding a categorical column (store_id, state_id) that's been
partitioned to a SINGLE category per chunk (e.g. by processing one
store at a time to fit memory) produces ZERO dummy columns -- no error,
no warning. A joint model trained across chunks with mismatched or
missing location columns is silently wrong, not obviously broken.

This test guards the fixed-category encoding approach in
scripts/build_full_catalog_chunks.py (create_model_features_fixed_location),
which encodes against the full known category list rather than whatever
happens to be present in a given chunk.
"""

import polars as pl
import pytest

from scripts.build_full_catalog_chunks import (
    create_model_features_fixed_location,
    STORE_DUMMY_COLUMNS,
    STATE_DUMMY_COLUMNS,
)


def _single_store_chunk(store_id: str, state_id: str, n_days: int = 20) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "item_id": ["ITEM_1"] * n_days,
            "day_num": list(range(1, n_days + 1)),
            "sales": [5] * n_days,
            "weekday": ["Monday"] * n_days,
            "event_name_1": [None] * n_days,
            "sell_price": [3.0] * n_days,
        }
    )


def test_naive_to_dummies_on_single_category_produces_zero_columns():
    """Confirms the FAILURE MODE this test suite guards against actually
    exists -- if Polars' to_dummies behavior ever changes, this should
    fail loudly rather than the fixed-location test silently becoming
    pointless."""
    single_category = pl.DataFrame({"store_id": ["CA_1"] * 10})
    naive_dummies = single_category.to_dummies(columns=["store_id"], drop_first=True)

    location_columns = [c for c in naive_dummies.columns if c.startswith("store_id_")]
    assert len(location_columns) == 0, (
        "Expected naive to_dummies on a single-category column with drop_first=True "
        "to produce zero columns (the documented failure mode) -- if this now "
        "produces columns, the fixed-location workaround may no longer be necessary, "
        "but should still be verified deliberately rather than assumed."
    )


@pytest.mark.parametrize(
    "store_id,state_id",
    [("CA_1", "CA"), ("TX_2", "TX"), ("WI_3", "WI")],
)
def test_fixed_location_encoding_produces_full_consistent_columns(store_id, state_id):
    """Every single-store chunk must produce the SAME full set of
    location dummy columns, regardless of which one store/state it
    contains -- otherwise concatenating chunks for joint training
    produces misaligned or missing columns."""
    chunk = _single_store_chunk(store_id, state_id)
    featured = create_model_features_fixed_location(chunk, store_id=store_id, state_id=state_id)

    for col in STORE_DUMMY_COLUMNS + STATE_DUMMY_COLUMNS:
        assert col in featured.columns, (
            f"Missing expected location column {col} when processing chunk "
            f"for store={store_id}, state={state_id}."
        )


def test_fixed_location_encoding_sets_correct_indicator():
    """The one dummy column matching this chunk's own store must be all
    1s; every other store's dummy column must be all 0s. (The reference
    category, e.g. CA_1, correctly has all dummies at 0 -- that's how
    one-hot-with-drop-first represents it, not a bug.)"""
    chunk = _single_store_chunk("TX_2", "TX")
    featured = create_model_features_fixed_location(chunk, store_id="TX_2", state_id="TX")

    own_column = "store_id_TX_2"
    assert (featured[own_column] == 1).all(), f"{own_column} should be all 1s for a TX_2-only chunk"

    other_columns = [c for c in STORE_DUMMY_COLUMNS if c != own_column]
    for col in other_columns:
        assert (featured[col] == 0).all(), f"{col} should be all 0s for a TX_2-only chunk"
