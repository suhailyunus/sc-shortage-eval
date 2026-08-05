"""
Precompute per-item stress thresholds for the FULL catalog.

The original add_stress_target(grouping=("item_id",)) pools sales across
ALL stores for a given item to compute its 90th-percentile threshold.
If we process store-by-store to fit in memory, we must replicate that
exact pooling first -- otherwise a per-store chunk would only see 1/10
of an item's history and compute a different (wrong) threshold, silently
changing the target definition versus the original methodology.

This runs once on the wide-format data (cheap: ~30,490 rows), before any
store chunking, and writes item_thresholds.parquet for reuse.
"""
import numpy as np
import pandas as pd

QUANTILE = 0.90


def compute_split_day(day_columns: list[str], train_fraction: float = 0.80) -> int:
    """Same day-boundary logic as find_split_day, but derived directly
    from the day column count (every series shares the same day range,
    so this is exact and doesn't require materializing long format)."""
    day_nums = sorted(int(c.replace("d_", "")) for c in day_columns)
    cutoff_position = int(len(day_nums) * train_fraction)
    return day_nums[cutoff_position]


def build_item_thresholds(sales_wide: pd.DataFrame, split_day: int) -> pd.DataFrame:
    """
    For each item, pool sales across all 10 stores for days <= split_day,
    and compute the 90th-percentile threshold on that pooled distribution.
    """
    day_columns = [c for c in sales_wide.columns if c.startswith("d_")]
    train_day_columns = [c for c in day_columns if int(c.replace("d_", "")) <= split_day]

    records = []
    fallback_values = []
    for item_id, group in sales_wide.groupby("item_id"):
        pooled = group[train_day_columns].to_numpy(dtype="float64").ravel()
        threshold = float(np.quantile(pooled, QUANTILE))
        records.append((item_id, threshold))
        fallback_values.append(pooled)

    fallback = float(np.quantile(np.concatenate(fallback_values), QUANTILE))

    df = pd.DataFrame(records, columns=["item_id", "stress_threshold"])
    return df, fallback


if __name__ == "__main__":
    sales = pd.read_csv("sales_train_validation.csv")
    day_columns = [c for c in sales.columns if c.startswith("d_")]
    split_day = compute_split_day(day_columns)
    print(f"split_day={split_day}")

    thresholds, fallback = build_item_thresholds(sales, split_day)
    print(f"Computed thresholds for {len(thresholds)} items, fallback={fallback}")
    thresholds.to_parquet("item_thresholds.parquet", index=False)
    with open("split_day.txt", "w") as f:
        f.write(str(split_day))
    print("Saved item_thresholds.parquet and split_day.txt")
