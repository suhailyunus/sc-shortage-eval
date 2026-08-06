from __future__ import annotations

import pandas as pd

ID_COLUMNS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]


def select_items(
    sales: pd.DataFrame,
    *,
    max_items: int | None = 100,
) -> pd.DataFrame:
    """Optionally restrict development work to the first N unique items."""

    if max_items is None:
        return sales.copy()

    items = sales["item_id"].drop_duplicates().head(max_items)
    return sales[sales["item_id"].isin(items)].copy()


def reshape_sales_long(sales: pd.DataFrame) -> pd.DataFrame:
    """Convert daily wide sales columns (d_1, d_2, ...) to long format."""

    missing = [column for column in ID_COLUMNS if column not in sales.columns]
    if missing:
        raise ValueError(f"Sales data is missing ID columns: {missing}")

    day_columns = [
        column
        for column in sales.columns
        if column.startswith("d_")
    ]
    if not day_columns:
        raise ValueError("No daily columns matching 'd_*' were found.")

    long_data = sales.melt(
        id_vars=ID_COLUMNS,
        value_vars=day_columns,
        var_name="day",
        value_name="sales",
    )

    long_data["day_num"] = (
        long_data["day"]
        .str.replace("d_", "", regex=False)
        .astype("int32")
    )

    return long_data


def find_split_day(
    day_num: pd.Series,
    *,
    train_fraction: float = 0.80,
) -> int:
    """
    Return the last day belonging to the training period.

    The cutoff is chosen so that approximately ``train_fraction`` of all
    observations fall on or before it. Because a single day contains many
    item-store rows, the realized fraction is only approximate; the day
    boundary itself is exact, which is what temporal validity requires.
    """

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    ordered_days = day_num.sort_values()
    cutoff_position = int(len(ordered_days) * train_fraction)
    return int(ordered_days.iloc[cutoff_position])


def add_stress_target(
    data: pd.DataFrame,
    *,
    quantile: float = 0.90,
    grouping: tuple[str, ...] = ("item_id",),
    threshold_cutoff_day: int | None = None,
) -> pd.DataFrame:
    """
    Create an item-relative high-demand stress proxy.

    The M5 data does not contain direct inventory or stockout labels.
    This target therefore identifies unusually high sales relative to
    the selected grouping's own historical distribution.

    Parameters
    ----------
    threshold_cutoff_day
        When supplied, each group's quantile is estimated using only
        observations on or before this day, and that fixed value then
        labels the entire series. This matters: estimating the quantile
        over the full history would define the training labels using
        sales that had not yet occurred, and no such threshold could be
        computed at scoring time in production. Groups with no history
        on or before the cutoff fall back to the global training-period
        quantile.

        Passing ``None`` reproduces the original full-sample behaviour
        and is retained only for backward compatibility. It leaks future
        information into the label and should not be used for any result
        that will be reported.
    """

    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between 0 and 1.")

    df = data.copy()
    group_columns = list(grouping)

    if threshold_cutoff_day is None:
        thresholds = df.groupby(group_columns)["sales"].transform(
            lambda values: values.quantile(quantile)
        )
    else:
        if "day_num" not in df.columns:
            raise ValueError(
                "day_num is required when threshold_cutoff_day is supplied."
            )

        history = df.loc[df["day_num"] <= threshold_cutoff_day]
        if history.empty:
            raise ValueError(
                f"No observations fall on or before day {threshold_cutoff_day}."
            )

        group_thresholds = (
            history.groupby(group_columns)["sales"].quantile(quantile)
        )
        fallback = float(history["sales"].quantile(quantile))

        thresholds = (
            df[group_columns]
            .merge(
                group_thresholds.rename("stress_threshold").reset_index(),
                on=group_columns,
                how="left",
            )["stress_threshold"]
            .fillna(fallback)
        )
        thresholds.index = df.index

    df["stress_threshold"] = thresholds
    df["stress_event"] = (df["sales"] > df["stress_threshold"]).astype("int8")

    return df


def merge_calendar(
    sales_long: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Attach calendar, event, week, and SNAP context."""

    calendar_columns = [
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]

    missing = [
        column for column in calendar_columns if column not in calendar.columns
    ]
    if missing:
        raise ValueError(f"Calendar data is missing columns: {missing}")

    return sales_long.merge(
        calendar[calendar_columns],
        left_on="day",
        right_on="d",
        how="left",
        validate="many_to_one",
    )


def merge_prices(
    data: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Attach weekly item-store selling prices."""

    price_columns = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    missing = [column for column in price_columns if column not in prices.columns]
    if missing:
        raise ValueError(f"Price data is missing columns: {missing}")

    return data.merge(
        prices[price_columns],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )


def build_analytical_table(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    max_items: int | None = 100,
    stress_quantile: float = 0.90,
    train_fraction: float = 0.80,
    stress_grouping: tuple[str, ...] = ("item_id", "store_id"),
) -> tuple[pd.DataFrame, int]:
    """
    Create the merged long-format analytical table.

    Returns the table together with the last day of the training period.
    The stress threshold is estimated from that period only, and callers
    must reuse the same boundary when splitting so that no label or
    feature is informed by the holdout.

    stress_grouping defaults to (item_id, store_id), NOT item_id alone.
    Grouping by item_id alone pools sales across all 10 stores per item,
    which lets high-volume stores exceed the pooled 90th percentile more
    often simply by selling more -- a volume artifact, not genuine demand
    stress. Verified: store-level stress rate correlated with store sales
    volume at Pearson r=0.85 under the item-only grouping, r=0.03 under
    this corrected one. See paper_draft/case_study.md, "Scaling past 100
    items -- and a critique that held up," for the full writeup. Pass
    grouping=("item_id",) explicitly only to reproduce the original,
    confounded behaviour for historical comparison -- never for a result
    that will be reported or deployed.
    """

    selected_sales = select_items(sales, max_items=max_items)
    long_sales = reshape_sales_long(selected_sales)

    split_day = find_split_day(
        long_sales["day_num"],
        train_fraction=train_fraction,
    )

    targeted = add_stress_target(
        long_sales,
        quantile=stress_quantile,
        grouping=stress_grouping,
        threshold_cutoff_day=split_day,
    )
    with_calendar = merge_calendar(targeted, calendar)
    with_prices = merge_prices(with_calendar, prices)

    analytical = with_prices.sort_values(
        ["item_id", "store_id", "day_num"]
    ).reset_index(drop=True)

    return analytical, split_day
