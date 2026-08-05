"""
Stratified item sampling for the 5,000-series validation sample.

Sampling unit: item_id. Every item spans all 10 stores in M5, so
sampling 500 items yields exactly 500 x 10 = 5,000 series with zero
store-representation bias.

Stratification: dept_id (7 depts, which also implies cat_id) x demand
class (Syntetos-Boylan: smooth / erratic / intermittent / lumpy),
computed from each item's total demand aggregated across all 10 stores.
Intermittent and lumpy strata are oversampled by a fixed multiplier
relative to their natural catalog share, since that's the hardest and
most business-relevant segment for this model.

This is a *sample selection* step, not feature engineering: using full
historical demand to classify and choose which items to study does not
leak into the model, because the classification is never used as a
feature or label, and it's the same use of full history any analyst
would use to decide "what should I even bother modeling."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 42
TARGET_ITEMS = 500
INTERMITTENT_OVERSAMPLE = 1.5  # multiplier applied to intermittent + lumpy strata weight

ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49


def classify_demand(sales_wide: pd.DataFrame, day_columns: list[str]) -> pd.DataFrame:
    """
    Aggregate demand across stores per item, then classify each item's
    demand pattern using Average demand Interval (ADI) and squared
    Coefficient of Variation (CV^2) of non-zero demand sizes.

    Returns one row per item_id with: mean_daily_demand, adi, cv2, demand_class.
    """
    item_totals = sales_wide.groupby("item_id")[day_columns].sum()
    values = item_totals.to_numpy(dtype="float64")
    n_periods = values.shape[1]

    records = []
    for item_id, row in zip(item_totals.index, values):
        nonzero_mask = row > 0
        n_nonzero = int(nonzero_mask.sum())

        if n_nonzero == 0:
            adi = np.inf
            cv2 = 0.0
            mean_demand = 0.0
        else:
            adi = n_periods / n_nonzero
            demand_sizes = row[nonzero_mask]
            mean_size = demand_sizes.mean()
            std_size = demand_sizes.std(ddof=0)
            cv2 = (std_size / mean_size) ** 2 if mean_size > 0 else 0.0
            mean_demand = row.mean()

        if adi < ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
            demand_class = "smooth"
        elif adi >= ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
            demand_class = "intermittent"
        elif adi < ADI_THRESHOLD and cv2 >= CV2_THRESHOLD:
            demand_class = "erratic"
        else:
            demand_class = "lumpy"

        records.append((item_id, mean_demand, adi, cv2, demand_class))

    return pd.DataFrame(
        records, columns=["item_id", "mean_daily_demand", "adi", "cv2", "demand_class"]
    )


def build_item_catalog(sales_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Compute item-level metadata + demand classification once. This is
    the expensive, seed-independent step \u2014 reuse the result across
    multiple stratified draws rather than recomputing per seed.
    """
    day_columns = [c for c in sales_wide.columns if c.startswith("d_")]
    item_meta = sales_wide[["item_id", "dept_id", "cat_id"]].drop_duplicates()
    demand_stats = classify_demand(sales_wide, day_columns)
    catalog = item_meta.merge(demand_stats, on="item_id", how="left")
    catalog["stratum"] = catalog["dept_id"] + "__" + catalog["demand_class"]
    return catalog


def build_stratified_item_sample(
    sales_wide: pd.DataFrame | None = None,
    *,
    catalog: pd.DataFrame | None = None,
    target_items: int = TARGET_ITEMS,
    oversample_multiplier: float = INTERMITTENT_OVERSAMPLE,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Return a DataFrame of exactly `target_items` sampled item_ids with
    their stratum metadata, using proportional-with-oversample allocation.

    Pass a precomputed `catalog` (from `build_item_catalog`) to skip
    re-running demand classification when only the random seed changes
    across repeated calls \u2014 classification is seed-independent and
    is the expensive part of this function.
    """
    if catalog is None:
        if sales_wide is None:
            raise ValueError("Provide either sales_wide or a precomputed catalog.")
        catalog = build_item_catalog(sales_wide)

    stratum_counts = catalog.groupby("stratum").size().rename("catalog_count")
    stratum_weight = stratum_counts / stratum_counts.sum()

    is_oversampled = stratum_weight.index.str.endswith(("intermittent", "lumpy"))
    adjusted_weight = stratum_weight.copy()
    adjusted_weight[is_oversampled] = adjusted_weight[is_oversampled] * oversample_multiplier
    adjusted_weight = adjusted_weight / adjusted_weight.sum()

    raw_targets = adjusted_weight * target_items
    target_per_stratum = np.floor(raw_targets).astype(int)

    # Distribute leftover items (from flooring) to the strata with the
    # largest fractional remainder, so the total lands exactly on target_items.
    remainder = target_items - target_per_stratum.sum()
    fractional = (raw_targets - target_per_stratum).sort_values(ascending=False)
    for stratum_name in fractional.index[:remainder]:
        target_per_stratum[stratum_name] += 1

    # Cap each stratum's target at its actual catalog count (can't oversample
    # beyond the number of items that exist in that stratum).
    target_per_stratum = target_per_stratum.combine(stratum_counts, min)

    rng = np.random.default_rng(random_seed)
    sampled_frames = []
    for stratum_name, n_take in target_per_stratum.items():
        if n_take <= 0:
            continue
        pool = catalog.loc[catalog["stratum"] == stratum_name]
        chosen_idx = rng.choice(pool.index.to_numpy(), size=int(n_take), replace=False)
        sampled_frames.append(pool.loc[chosen_idx])

    sample = pd.concat(sampled_frames, ignore_index=True)

    # If capping caused a shortfall vs target_items, top up randomly from
    # whatever's left in the catalog (rare, but keep the total exact).
    shortfall = target_items - len(sample)
    if shortfall > 0:
        remaining_pool = catalog.loc[~catalog["item_id"].isin(sample["item_id"])]
        top_up_idx = rng.choice(remaining_pool.index.to_numpy(), size=shortfall, replace=False)
        sample = pd.concat([sample, remaining_pool.loc[top_up_idx]], ignore_index=True)

    return sample.sort_values(["dept_id", "demand_class", "item_id"]).reset_index(drop=True)


if __name__ == "__main__":
    sales = pd.read_csv("sales_train_validation.csv")

    sample = build_stratified_item_sample(sales)

    print(f"Sampled {len(sample)} items -> {len(sample) * 10} series (10 stores each)\n")

    print("Stratum breakdown (sampled vs. full catalog):")
    catalog_share = (
        sales[["item_id", "dept_id"]]
        .drop_duplicates()
        .groupby("dept_id")
        .size()
        .rename("catalog_items")
    )
    sample_share = sample.groupby(["dept_id", "demand_class"]).size().rename("sampled_items")
    print(sample_share.unstack(fill_value=0))
    print()
    print("Demand class totals in sample:")
    print(sample["demand_class"].value_counts())
    print()
    print("Demand class totals in full catalog (for comparison):")
    full_classes = classify_demand(sales, [c for c in sales.columns if c.startswith("d_")])
    print(full_classes["demand_class"].value_counts())

    sample[["item_id", "dept_id", "cat_id", "demand_class", "mean_daily_demand", "adi", "cv2"]].to_csv(
        "stratified_item_sample.csv", index=False
    )
    print("\nSaved item list to stratified_item_sample.csv")
