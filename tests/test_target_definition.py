"""
Regression test for the store-volume confound in the stress target
definition, flagged by external review (Gemini) and verified empirically
during the Validation at Scale work (see README.md).

Grouping the stress threshold by `item_id` alone pools sales across all
10 stores per item. High-volume stores then exceed that pooled 90th
percentile more often simply by selling more units -- a volume artifact,
not genuine demand stress. Verified: store-level stress-event rate
correlated with store-level average sales volume at Pearson r=0.85 under
the (item_id) grouping, and r=0.03 under the corrected (item_id,
store_id) grouping.

This test guards against that confound silently coming back -- e.g. if
someone "simplifies" the grouping back to item_id alone during a future
refactor.
"""

import numpy as np
import polars as pl
import pytest

from src.polars_preprocess import add_stress_target

# Absolute spread (max - min) ceiling for per-store stress rates.
# The real full-catalog validation measured spread ~2.6x (7.7%-19.9%)
# under the flawed grouping and ~1.2 percentage points (6.7%-8.1%)
# under the corrected one -- this threshold sits between those.
SPREAD_THRESHOLD = 0.03


def _synthetic_multi_store_frame(n_days: int = 500, n_items: int = 30) -> pl.DataFrame:
    """
    n_items items x 3 stores, where store A sells consistently more units
    than store B, which sells more than store C -- but each store's
    demand is otherwise "flat" (no genuine anomalies). Under a volume
    confound, the item-pooled threshold will flag store A far more often
    than store C even though nothing unusual is happening at store A
    specifically.

    Uses enough items x days that per-store stress rates converge to a
    stable estimate -- a correlation computed over only 3 store-level
    points (one per store) is otherwise dominated by sampling noise
    rather than the effect being tested for.
    """
    rng = np.random.default_rng(0)
    store_base_volume = {"STORE_A": 50, "STORE_B": 20, "STORE_C": 5}

    rows = []
    for item_idx in range(n_items):
        item_id = f"ITEM_{item_idx}"
        for store_id, base in store_base_volume.items():
            # Small, proportionate noise -- no genuine spikes, just
            # different baseline volume per store.
            daily_sales = rng.poisson(lam=base, size=n_days)
            for day_num, sales in enumerate(daily_sales, start=1):
                rows.append(
                    {"item_id": item_id, "store_id": store_id, "day_num": day_num, "sales": int(sales)}
                )
    return pl.DataFrame(rows)


def test_item_only_grouping_shows_volume_confound():
    """Sanity check on the test fixture itself: confirm the OLD (flawed)
    grouping does produce the confound in this synthetic data (a wide
    spread in per-store stress rates purely from volume differences), so
    the fixture is actually exercising the failure mode it's meant to
    catch."""
    df = _synthetic_multi_store_frame()
    cutoff_day = int(df["day_num"].max() * 0.8)

    targeted = add_stress_target(df, quantile=0.90, grouping=("item_id",), threshold_cutoff_day=cutoff_day)

    rates = targeted.group_by("store_id").agg(pl.col("stress_event").mean().alias("rate"))["rate"]
    spread = rates.max() - rates.min()

    assert spread > SPREAD_THRESHOLD, (
        f"Test fixture did not reproduce the volume confound under item-only "
        f"grouping (spread={spread:.3f}) -- fixture may need stronger volume "
        f"separation between stores."
    )


def test_item_store_grouping_removes_volume_confound():
    """The actual regression test: with the CORRECTED (item_id, store_id)
    grouping, per-store stress rates must be tightly clustered -- NOT
    spread out in proportion to store sales volume."""
    df = _synthetic_multi_store_frame()
    cutoff_day = int(df["day_num"].max() * 0.8)

    targeted = add_stress_target(
        df, quantile=0.90, grouping=("item_id", "store_id"), threshold_cutoff_day=cutoff_day
    )

    rates = targeted.group_by("store_id").agg(pl.col("stress_event").mean().alias("rate"))["rate"]
    spread = rates.max() - rates.min()

    assert spread < SPREAD_THRESHOLD, (
        f"Per-store stress rates spread by {spread:.3f} under (item_id, store_id) "
        f"grouping -- the volume confound may have regressed. Expected spread < "
        f"{SPREAD_THRESHOLD}."
    )
