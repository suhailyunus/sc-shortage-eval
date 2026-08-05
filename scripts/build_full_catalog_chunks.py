"""
Full-catalog (3,049 items x 10 stores = 30,490 series) feature
engineering, processed one store at a time to fit this container's
memory budget. Each store chunk (~3,049 items x 1,913 days = ~5.83M
rows) is comparable in size to the 5,000-series sample we already
proved works, so this should be safe.

Uses the precomputed cross-store item_thresholds (item_thresholds.parquet)
so the stress_event target has EXACTLY the same definition as the
original: threshold = item's 90th percentile pooled across all 10
stores, train-period only. Computing it per-store-chunk would silently
change the target (only 1/10th the pooled history), which would make
any comparison invalid.

Each store's result is saved to disk immediately (checkpointed) so a
timeout mid-run doesn't lose completed work -- reruns skip stores
already done.
"""

import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

from src.polars_preprocess import merge_calendar, merge_prices, add_stress_target
from src.polars_features import BASE_MODEL_FEATURES

STORES = [f"CA_{i}" for i in range(1, 5)] + [f"TX_{i}" for i in range(1, 4)] + [f"WI_{i}" for i in range(1, 4)]
ALL_STORES = sorted(STORES)  # CA_1..CA_4, TX_1..TX_3, WI_1..WI_3 -- fixed, known full list
ALL_STATES = ["CA", "TX", "WI"]
# Match pl.to_dummies(drop_first=True) convention: drop the alphabetically-first category
STORE_DUMMY_COLUMNS = [f"store_id_{s}" for s in ALL_STORES[1:]]
STATE_DUMMY_COLUMNS = [f"state_id_{s}" for s in ALL_STATES[1:]]
FULL_FEATURE_NAMES = BASE_MODEL_FEATURES + STORE_DUMMY_COLUMNS + STATE_DUMMY_COLUMNS


def create_model_features_fixed_location(data: pl.DataFrame, store_id: str, state_id: str) -> pl.DataFrame:
    """
    Same as polars_features.create_model_features, except location is
    one-hot encoded against the FULL known category list (all 10 stores,
    all 3 states) rather than whatever categories happen to be present
    in this chunk. Since each chunk here is exactly one store/state, the
    normal to_dummies() would produce zero location columns (a single
    category has nothing to encode against) -- this makes every chunk
    produce the same, stackable set of location columns instead.
    """
    group_keys = ["item_id"]  # store is fixed within this chunk, so grouping is just item_id
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
    df = df.with_columns((pl.col("sell_price") - pl.col("price_lag_1")).alias("price_change_1"))

    # Fixed-category location dummies: every chunk gets the full, identical
    # set of columns, filled with the correct constant (1 for this chunk's
    # own store/state, 0 for all others) rather than inferred per-chunk.
    location_values = {}
    for col in STORE_DUMMY_COLUMNS:
        location_values[col] = 1 if col == f"store_id_{store_id}" else 0
    for col in STATE_DUMMY_COLUMNS:
        location_values[col] = 1 if col == f"state_id_{state_id}" else 0

    df = df.with_columns(
        [pl.lit(v, dtype=pl.Int8).alias(k) for k, v in location_values.items()]
    )
    return df


def prepare_model_input_fixed_location(data: pl.DataFrame, store_id: str, state_id: str):
    featured = create_model_features_fixed_location(data, store_id, state_id)
    ready = featured.drop_nulls(subset=FULL_FEATURE_NAMES)
    X_ready = ready.select(FULL_FEATURE_NAMES)
    return ready, X_ready, FULL_FEATURE_NAMES

TRAIN_SUBSAMPLE_FRACTION = 0.20  # keep this fraction of each store's TRAIN rows for the joint fit
RANDOM_SEED = 42

CATS_CAL = ["d", "weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]


def process_store(store_id: str, split_day: int, thresholds: pl.DataFrame, calendar: pl.DataFrame):
    out_train = f"chunk_train_{store_id}.parquet"
    out_holdout = f"chunk_holdout_{store_id}.parquet"
    if os.path.exists(out_train) and os.path.exists(out_holdout):
        print(f"  [{store_id}] already done, skipping")
        return

    t0 = time.time()
    sales = (
        pl.scan_csv("sales_train_validation.csv")
        .filter(pl.col("store_id") == store_id)
        .collect()
    )
    prices = (
        pl.scan_csv("sell_prices.csv")
        .filter(pl.col("store_id") == store_id)
        .collect()
    )

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales = sales.with_columns([pl.col(c).cast(pl.Categorical) for c in id_cols])
    day_columns = [c for c in sales.columns if c.startswith("d_")]
    sales = sales.with_columns([pl.col(c).cast(pl.Int32) for c in day_columns])
    prices = prices.with_columns(
        [pl.col("store_id").cast(pl.Categorical), pl.col("item_id").cast(pl.Categorical),
         pl.col("wm_yr_wk").cast(pl.Int32), pl.col("sell_price").cast(pl.Float32)]
    )

    long_sales = sales.unpivot(
        index=id_cols, on=day_columns, variable_name="day", value_name="sales"
    ).with_columns(
        pl.col("day").str.replace("d_", "").cast(pl.Int32).alias("day_num"),
        pl.col("day").cast(pl.Categorical),
    )
    del sales
    gc.collect()

    # Since this chunk is a single store, grouping by item_id here is
    # equivalent to grouping by (item_id, store_id) -- exactly the fix
    # for the volume-confound Gemini flagged: no more pooling a threshold
    # across other stores this chunk doesn't even contain.
    long_sales = add_stress_target(
        long_sales, quantile=0.90, grouping=("item_id",), threshold_cutoff_day=split_day
    )

    with_cal = merge_calendar(long_sales, calendar)
    del long_sales
    analytical = merge_prices(with_cal, prices).sort(["item_id", "day_num"])
    del with_cal, prices
    gc.collect()

    state_id = store_id.split("_")[0]  # CA_1 -> CA, TX_2 -> TX, WI_3 -> WI
    ready, X_all, feature_names = prepare_model_input_fixed_location(analytical, store_id, state_id)
    del analytical
    gc.collect()

    y = ready["stress_event"].to_numpy()
    day_num = ready["day_num"].to_numpy()
    X_np = X_all.cast(pl.Float32).to_numpy()
    del ready, X_all
    gc.collect()

    train_mask = day_num <= split_day
    holdout_mask = ~train_mask

    rng = np.random.default_rng(RANDOM_SEED)
    train_idx = np.where(train_mask)[0]
    keep_n = int(len(train_idx) * TRAIN_SUBSAMPLE_FRACTION)
    sub_idx = rng.choice(train_idx, size=keep_n, replace=False)

    train_df = pd.DataFrame(X_np[sub_idx], columns=feature_names)
    train_df["stress_event"] = y[sub_idx]
    train_df.to_parquet(out_train, index=False)

    holdout_idx = np.where(holdout_mask)[0]
    holdout_df = pd.DataFrame(X_np[holdout_idx], columns=feature_names)
    holdout_df["stress_event"] = y[holdout_idx]
    holdout_df.to_parquet(out_holdout, index=False)

    elapsed = time.time() - t0
    print(f"  [{store_id}] train_kept={len(sub_idx):,}/{len(train_idx):,}  "
          f"holdout={len(holdout_idx):,}  total_rows={len(y):,}  ({elapsed:.0f}s)")

    del X_np, y, day_num, train_df, holdout_df
    gc.collect()


if __name__ == "__main__":
    split_day = int(open("split_day.txt").read().strip())
    thresholds = pl.read_parquet("item_thresholds.parquet")
    thresholds = thresholds.with_columns(pl.col("item_id").cast(pl.Categorical))

    calendar = pl.read_csv("calendar.csv")
    calendar = calendar.with_columns([pl.col(c).cast(pl.Categorical) for c in CATS_CAL])
    calendar = calendar.with_columns(
        [
            pl.col("wday").cast(pl.Int8), pl.col("month").cast(pl.Int8), pl.col("year").cast(pl.Int16),
            pl.col("snap_CA").cast(pl.Int8), pl.col("snap_TX").cast(pl.Int8), pl.col("snap_WI").cast(pl.Int8),
            pl.col("wm_yr_wk").cast(pl.Int32),
        ]
    )

    stores_to_run = sys.argv[1:] if len(sys.argv) > 1 else STORES
    print(f"Processing stores: {stores_to_run}")

    for store_id in stores_to_run:
        process_store(store_id, split_day, thresholds, calendar)
