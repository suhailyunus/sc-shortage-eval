"""
Validate the supply-chain-stress model at scale, beyond the original
100-item development sample.

Two scopes are supported:

  python scripts/validate_at_scale.py --scope 5000 --seed 42
      Trains and evaluates on a stratified 5,000-series sample (500 items
      x 10 stores), stratified by department and demand class (Syntetos-
      Boylan ADI/CV^2 classification), with intermittent/lumpy items
      oversampled 1.5x relative to their natural catalog share.

  python scripts/validate_at_scale.py --scope full
      Trains on a 20% row-subsample of the training period and evaluates
      on the COMPLETE, non-subsampled 30,490-series holdout. Requires
      `scripts/build_full_catalog_chunks.py` to have been run first
      (produces chunk_train_*.parquet / chunk_holdout_*.parquet).

Both scopes use the CORRECTED stress-target definition -- grouping by
(item_id, store_id), not item_id alone. Grouping by item_id alone pools
sales across all 10 stores per item, which lets high-volume stores
exceed the pooled 90th percentile more often simply by selling more,
independent of genuine demand stress (verified: store-level stress rate
correlated with store sales volume at Pearson r=0.85 under the old
grouping; r=0.03 under this corrected one). See README.md ->
"Validation at Scale" for the full writeup, including comparison numbers
against the original, uncorrected 100-item run.

Cost assumptions ($20/missed event, $12/false alarm, $10/mitigated true
positive) match scripts/report_business_impact.py and are not re-derived
here; see that script / the README for the sourcing rationale.
"""

from __future__ import annotations

import argparse
import gc
import glob

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score

from src.polars_preprocess import (
    reshape_sales_long, find_split_day, add_stress_target, merge_calendar, merge_prices,
)
from src.polars_features import prepare_model_input
from scripts.build_stratified_sample import build_stratified_item_sample, build_item_catalog

FN_COST, FP_COST, TP_COST = 20, 12, 10

CATS_SALES = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CATS_CAL = ["d", "weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
CATS_PRICES = ["store_id", "item_id"]


def _tighten_dtypes(sales: pl.DataFrame, calendar: pl.DataFrame, prices: pl.DataFrame):
    """Categorical + integer downcasting; see README Engineering Lessons
    for why this is required, not optional, at this row count."""
    sales = sales.with_columns([pl.col(c).cast(pl.Categorical) for c in CATS_SALES])
    day_columns = [c for c in sales.columns if c.startswith("d_")]
    sales = sales.with_columns([pl.col(c).cast(pl.Int32) for c in day_columns])

    calendar = calendar.with_columns([pl.col(c).cast(pl.Categorical) for c in CATS_CAL])
    calendar = calendar.with_columns(
        [
            pl.col("wday").cast(pl.Int8), pl.col("month").cast(pl.Int8), pl.col("year").cast(pl.Int16),
            pl.col("snap_CA").cast(pl.Int8), pl.col("snap_TX").cast(pl.Int8), pl.col("snap_WI").cast(pl.Int8),
            pl.col("wm_yr_wk").cast(pl.Int32),
        ]
    )
    prices = prices.with_columns([pl.col(c).cast(pl.Categorical) for c in CATS_PRICES])
    prices = prices.with_columns([pl.col("wm_yr_wk").cast(pl.Int32), pl.col("sell_price").cast(pl.Float32)])
    return sales, calendar, prices


def run_5000_series(seed: int, data_dir: str = "."):
    sales_full_pd = pd.read_csv(f"{data_dir}/sales_train_validation.csv")
    catalog = build_item_catalog(sales_full_pd)
    del sales_full_pd
    gc.collect()

    sample = build_stratified_item_sample(catalog=catalog, random_seed=seed)
    sample_items = sample["item_id"].tolist()

    calendar = pl.read_csv(f"{data_dir}/calendar.csv")
    sales = pl.scan_csv(f"{data_dir}/sales_train_validation.csv").filter(pl.col("item_id").is_in(sample_items)).collect()
    prices = pl.scan_csv(f"{data_dir}/sell_prices.csv").filter(pl.col("item_id").is_in(sample_items)).collect()
    sales, calendar, prices = _tighten_dtypes(sales, calendar, prices)

    long_sales = reshape_sales_long(sales).with_columns(pl.col("day").cast(pl.Categorical))
    del sales
    split_day = find_split_day(long_sales["day_num"], train_fraction=0.80)

    # CORRECTED grouping: (item_id, store_id), not item_id alone.
    targeted = add_stress_target(
        long_sales, quantile=0.90, grouping=("item_id", "store_id"), threshold_cutoff_day=split_day
    )
    del long_sales
    with_cal = merge_calendar(targeted, calendar)
    del targeted
    analytical = merge_prices(with_cal, prices).sort(["item_id", "store_id", "day_num"])
    del with_cal, prices
    gc.collect()

    ready, X_all, feature_names = prepare_model_input(analytical)
    del analytical
    gc.collect()

    y_all = ready["stress_event"].to_numpy()
    day_num = ready["day_num"].to_numpy()
    X_np = X_all.cast(pl.Float32).to_numpy()
    del ready, X_all
    gc.collect()

    train_mask = day_num <= split_day
    X_train, y_train = X_np[train_mask], y_all[train_mask]
    X_holdout, y_holdout = X_np[~train_mask], y_all[~train_mask]

    return _fit_and_report(X_train, y_train, X_holdout, y_holdout, label=f"5,000-series (seed={seed})")


def run_full_catalog(chunk_dir: str = "."):
    train_files = sorted(glob.glob(f"{chunk_dir}/chunk_train_*.parquet"))
    holdout_files = sorted(glob.glob(f"{chunk_dir}/chunk_holdout_*.parquet"))
    if not train_files or not holdout_files:
        raise FileNotFoundError(
            "No chunk_train_*/chunk_holdout_* parquet files found. "
            "Run scripts/build_full_catalog_chunks.py first."
        )

    train_df = pd.concat([pd.read_parquet(f) for f in train_files], ignore_index=True)
    # Exclude _day_num: chronological metadata added by
    # build_full_catalog_chunks.py for scripts/train_calibrate_full_catalog.py's
    # chronological split, not a model feature. Letting it into the
    # feature set would hand the model a literal, high-signal timestamp.
    feature_cols = [c for c in train_df.columns if c not in ("stress_event", "_day_num")]
    X_train = train_df[feature_cols].to_numpy(dtype="float32")
    y_train = train_df["stress_event"].to_numpy()
    del train_df
    gc.collect()

    model = _fit(X_train, y_train)
    del X_train, y_train
    gc.collect()

    all_probs, all_y = [], []
    for f in holdout_files:
        df = pd.read_parquet(f)
        probs = model.predict_proba(df[feature_cols].to_numpy(dtype="float32"))[:, 1]
        all_probs.append(probs)
        all_y.append(df["stress_event"].to_numpy())
        del df
        gc.collect()

    probs = np.concatenate(all_probs)
    y_holdout = np.concatenate(all_y)
    return _report(probs, y_holdout, label="30,490-series (full catalog)")


def _fit(X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        tree_method="hist", n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def _fit_and_report(X_train, y_train, X_holdout, y_holdout, label: str):
    model = _fit(X_train, y_train)
    probs = model.predict_proba(X_holdout)[:, 1]
    return _report(probs, y_holdout, label=label)


def _report(probs, y_holdout, label: str):
    total_positives = int(y_holdout.sum())
    do_nothing_cost = total_positives * FN_COST

    print(f"\n=== {label} ===")
    print(f"Holdout: {len(y_holdout):,} rows, base rate={y_holdout.mean():.4f}")

    preds_80 = (probs >= 0.80).astype(int)
    tp = int(((preds_80 == 1) & (y_holdout == 1)).sum())
    fp = int(((preds_80 == 1) & (y_holdout == 0)).sum())
    fn = int(((preds_80 == 0) & (y_holdout == 1)).sum())
    net_80 = do_nothing_cost - (tp * TP_COST + fp * FP_COST + fn * FN_COST)
    print(f"@0.80: precision={precision_score(y_holdout, preds_80, zero_division=0):.3f} "
          f"recall={recall_score(y_holdout, preds_80, zero_division=0):.3f} net=${net_80:+,}")

    best_net, best_threshold = None, None
    for threshold in np.arange(0.50, 0.99, 0.01):
        preds = (probs >= threshold).astype(int)
        tp_s = int(((preds == 1) & (y_holdout == 1)).sum())
        fp_s = int(((preds == 1) & (y_holdout == 0)).sum())
        fn_s = int(((preds == 0) & (y_holdout == 1)).sum())
        net_s = do_nothing_cost - (tp_s * TP_COST + fp_s * FP_COST + fn_s * FN_COST)
        if best_net is None or net_s > best_net:
            best_net, best_threshold = net_s, threshold

    print(f"ROI-optimal: threshold={best_threshold:.2f} net=${best_net:+,.0f}")
    return {"net_80": net_80, "roi_threshold": best_threshold, "roi_net": best_net}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["5000", "full"], required=True)
    parser.add_argument("--seed", type=int, default=42, help="Only used for --scope 5000")
    parser.add_argument("--data-dir", default=".")
    args = parser.parse_args()

    if args.scope == "5000":
        run_5000_series(seed=args.seed, data_dir=args.data_dir)
    else:
        run_full_catalog(chunk_dir=args.data_dir)
