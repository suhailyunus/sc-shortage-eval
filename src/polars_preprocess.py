"""
Polars port of preprocess.py.

Semantics are intentionally identical to the pandas version:
- item selection now takes an explicit list of item_ids (from the
  stratified sample) rather than "first N by appearance order"
- the train/test day-boundary split logic is unchanged
- the stress-target leakage guard (threshold_cutoff_day) is unchanged:
  quantiles are computed only from on-or-before-cutoff history, and that
  fixed value is broadcast onto the full series, exactly as before
"""

from __future__ import annotations

import polars as pl

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def select_items(sales: pl.DataFrame, *, item_ids: list[str] | None = None) -> pl.DataFrame:
    """Restrict to an explicit set of item_ids (e.g. the stratified sample)."""
    if item_ids is None:
        return sales
    return sales.filter(pl.col("item_id").is_in(item_ids))


def reshape_sales_long(sales: pl.DataFrame) -> pl.DataFrame:
    """Convert daily wide sales columns (d_1, d_2, ...) to long format."""
    missing = [c for c in ID_COLUMNS if c not in sales.columns]
    if missing:
        raise ValueError(f"Sales data is missing ID columns: {missing}")

    day_columns = [c for c in sales.columns if c.startswith("d_")]
    if not day_columns:
        raise ValueError("No daily columns matching 'd_*' were found.")

    long_data = sales.unpivot(
        index=ID_COLUMNS,
        on=day_columns,
        variable_name="day",
        value_name="sales",
    )

    long_data = long_data.with_columns(
        pl.col("day").str.replace("d_", "").cast(pl.Int32).alias("day_num")
    )

    return long_data


def find_split_day(day_num: pl.Series, *, train_fraction: float = 0.80) -> int:
    """Return the last day belonging to the training period (day-boundary exact)."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    ordered = day_num.sort()
    cutoff_position = int(len(ordered) * train_fraction)
    return int(ordered[cutoff_position])


def add_stress_target(
    data: pl.DataFrame,
    *,
    quantile: float = 0.90,
    grouping: tuple[str, ...] = ("item_id",),
    threshold_cutoff_day: int | None = None,
) -> pl.DataFrame:
    """
    Create an item-relative high-demand stress proxy.

    Same leakage guard as the pandas version: when threshold_cutoff_day
    is given, quantiles are estimated using only on-or-before-cutoff
    history, and that fixed value labels the entire series (train and
    holdout alike). Groups with no history on/before the cutoff fall
    back to the global training-period quantile.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between 0 and 1.")

    group_columns = list(grouping)

    if threshold_cutoff_day is None:
        return data.with_columns(
            pl.col("sales").quantile(quantile).over(group_columns).alias("stress_threshold")
        ).with_columns(
            (pl.col("sales") > pl.col("stress_threshold")).cast(pl.Int8).alias("stress_event")
        )

    if "day_num" not in data.columns:
        raise ValueError("day_num is required when threshold_cutoff_day is supplied.")

    history = data.filter(pl.col("day_num") <= threshold_cutoff_day)
    if history.height == 0:
        raise ValueError(f"No observations fall on or before day {threshold_cutoff_day}.")

    group_thresholds = history.group_by(group_columns).agg(
        pl.col("sales").quantile(quantile).alias("stress_threshold")
    )
    fallback = float(history["sales"].quantile(quantile))

    df = data.join(group_thresholds, on=group_columns, how="left")
    df = df.with_columns(pl.col("stress_threshold").fill_null(fallback))
    df = df.with_columns(
        (pl.col("sales") > pl.col("stress_threshold")).cast(pl.Int8).alias("stress_event")
    )
    return df


def merge_calendar(sales_long: pl.DataFrame, calendar: pl.DataFrame) -> pl.DataFrame:
    """Attach calendar, event, week, and SNAP context."""
    calendar_columns = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    missing = [c for c in calendar_columns if c not in calendar.columns]
    if missing:
        raise ValueError(f"Calendar data is missing columns: {missing}")

    return sales_long.join(
        calendar.select(calendar_columns),
        left_on="day",
        right_on="d",
        how="left",
    )


def merge_prices(data: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """Attach weekly item-store selling prices."""
    price_columns = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    missing = [c for c in price_columns if c not in prices.columns]
    if missing:
        raise ValueError(f"Price data is missing columns: {missing}")

    return data.join(
        prices.select(price_columns),
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )


def build_analytical_table(
    sales: pl.DataFrame,
    calendar: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    item_ids: list[str] | None,
    stress_quantile: float = 0.90,
    train_fraction: float = 0.80,
) -> tuple[pl.DataFrame, int]:
    """
    Create the merged long-format analytical table.

    Returns the table together with the last day of the training period,
    same contract as the pandas version.
    """
    selected_sales = select_items(sales, item_ids=item_ids)
    long_sales = reshape_sales_long(selected_sales)

    split_day = find_split_day(long_sales["day_num"], train_fraction=train_fraction)

    targeted = add_stress_target(
        long_sales,
        quantile=stress_quantile,
        grouping=("item_id",),
        threshold_cutoff_day=split_day,
    )
    with_calendar = merge_calendar(targeted, calendar)
    with_prices = merge_prices(with_calendar, prices)

    analytical = with_prices.sort(["item_id", "store_id", "day_num"])

    return analytical, split_day
