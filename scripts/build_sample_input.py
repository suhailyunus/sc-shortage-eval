"""
Build a demonstration input file that exercises every risk band.

The committed sample scores three rows, all in one band, and raises no
alert. That is a poor showcase for a public demo: a reader learns
nothing about what the model distinguishes.

This script scores real holdout rows, then selects item-store series
whose predictions span Low through Critical, keeping enough leading
history for each series that the seven-day features can be rebuilt at
inference time.

Run from the repository root:

    python scripts/build_sample_input.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_training_pipeline  # noqa: E402

# Columns the API expects in an uploaded file.
INPUT_COLUMNS = [
    "item_id",
    "store_id",
    "state_id",
    "day_num",
    "sales",
    "weekday",
    "event_name_1",
    "sell_price",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]

# Feature warm-up: seven prior days, plus the scored day itself.
HISTORY_ROWS = 8


def band(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "Critical" if probability >= threshold + (1 - threshold) / 2 else "High"
    return "Moderate" if probability >= threshold / 2 else "Low"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="examples/sample_input.csv")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--days-per-series", type=int, default=14)
    args = parser.parse_args()

    print("Scoring holdout rows to locate representative series...\n")
    result = run_training_pipeline(args.data_dir, max_items=100)

    scored = result.feature_data.loc[result.X_test.index].copy()
    scored["probability"] = result.model.predict_proba(result.X_test)[:, 1]
    scored["band"] = [band(p, args.threshold) for p in scored["probability"]]

    # Pick one series per band, preferring the clearest example of each.
    chosen: list[tuple[str, str, int]] = []
    for target, ascending in (
        ("Critical", False),
        ("High", False),
        ("Moderate", True),
        ("Low", True),
    ):
        candidates = scored[scored["band"] == target]
        if candidates.empty:
            print(f"  no holdout rows landed in {target}; skipping")
            continue

        row = candidates.sort_values("probability", ascending=ascending).iloc[0]
        key = (row["item_id"], row["store_id"], int(row["day_num"]))
        if key[:2] in {(i, s) for i, s, _ in chosen}:
            continue
        chosen.append(key)
        print(
            f"  {target:9s} <- {row['item_id']} / {row['store_id']} "
            f"day {int(row['day_num'])}  (p={row['probability']:.3f})"
        )

    if not chosen:
        raise SystemExit("No series could be selected.")

    analytical = result.analytical_data
    frames = []
    for item_id, store_id, day_num in chosen:
        series = analytical[
            (analytical["item_id"] == item_id)
            & (analytical["store_id"] == store_id)
            & (analytical["day_num"] <= day_num)
        ].sort_values("day_num")

        window = max(HISTORY_ROWS, args.days_per_series)
        frames.append(series.tail(window))

    sample = pd.concat(frames, ignore_index=True)

    missing = [c for c in INPUT_COLUMNS if c not in sample.columns]
    if missing:
        raise SystemExit(f"Analytical table is missing expected columns: {missing}")

    sample = sample[INPUT_COLUMNS].sort_values(
        ["item_id", "store_id", "day_num"]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False)

    scorable = len(sample) - (HISTORY_ROWS - 1) * len(chosen)
    print(f"\nWrote {output}")
    print(f"  rows              : {len(sample)}")
    print(f"  series            : {len(chosen)}")
    print(f"  expected scorable : ~{scorable}")
    print("\nUpload this through /predict-file-csv to confirm the spread.")


if __name__ == "__main__":
    main()
