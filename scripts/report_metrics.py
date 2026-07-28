"""
Retrain on the corrected split and print every number the README needs.

Run from the repository root:

    python scripts/report_metrics.py

Optionally point at a different data directory:

    python scripts/report_metrics.py --data-dir data/raw --max-items 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_training_pipeline  # noqa: E402


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of true positives among the k highest-scored observations."""

    if k > len(scores):
        return float("nan")
    top = np.argsort(scores)[::-1][:k]
    return float(y_true[top].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    args = parser.parse_args()

    print("Training on the corrected chronological split...\n")
    result = run_training_pipeline(
        args.data_dir,
        max_items=args.max_items,
        train_fraction=args.train_fraction,
        models_dir=None,  # nothing is overwritten until we agree on the numbers
    )

    feature_data = result.feature_data
    y_test = result.y_test.to_numpy()
    scores = result.model.predict_proba(result.X_test)[:, 1]

    train_days = feature_data.loc[result.X_train.index, "day_num"]
    test_days = feature_data.loc[result.X_test.index, "day_num"]

    print("=" * 62)
    print("SPLIT")
    print("=" * 62)
    print(f"last training day     : {result.split_day}")
    print(f"train days            : {train_days.min()} -> {train_days.max()}")
    print(f"test  days            : {test_days.min()} -> {test_days.max()}")
    print(f"day overlap           : {len(set(train_days) & set(test_days))}")
    print(f"train rows            : {len(result.X_train):,}")
    print(f"test  rows            : {len(result.X_test):,}")

    base_rate = float(y_test.mean())
    print()
    print("=" * 62)
    print("BASE RATE  (the number the old README never stated)")
    print("=" * 62)
    print(f"train positive rate   : {result.y_train.mean():.4f}")
    print(f"test  positive rate   : {base_rate:.4f}")
    print(f"all-negative accuracy : {1 - base_rate:.4f}   <- the bar to beat")

    print()
    print("=" * 62)
    print("RANKING QUALITY")
    print("=" * 62)
    average_precision = average_precision_score(y_test, scores)
    print(f"average precision     : {average_precision:.4f}")
    print(f"random baseline (AP)  : {base_rate:.4f}")
    print(f"lift over random      : {average_precision / base_rate:.2f}x")

    print()
    print("precision@k  (the metric that matches a review queue)")
    for k in (50, 100, 250, 500, 1000):
        value = precision_at_k(y_test, scores, k)
        if not np.isnan(value):
            print(f"  P@{k:<5} = {value:.4f}   ({value / base_rate:.2f}x base rate)")

    print()
    print("=" * 62)
    print("PERSISTENCE BASELINE  (does the model beat 'yesterday was high'?)")
    print("=" * 62)
    test_rows = feature_data.loc[result.X_test.index]
    persistence = (
        test_rows["sales_lag_1"] > test_rows["stress_threshold"]
    ).astype(int).to_numpy()

    print(classification_report(y_test, persistence, zero_division=0, digits=4))
    print("confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, persistence))

    print()
    print("=" * 62)
    print("MODEL @ threshold 0.50")
    print("=" * 62)
    predictions = (scores >= 0.50).astype(int)
    print(classification_report(y_test, predictions, zero_division=0, digits=4))
    print("confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, predictions))

    print()
    print("=" * 62)
    print("THRESHOLD SWEEP")
    print("=" * 62)
    rows = []
    for threshold in np.arange(0.10, 0.95, 0.05):
        predicted = (scores >= threshold).astype(int)
        flagged = int(predicted.sum())
        true_positives = int(((predicted == 1) & (y_test == 1)).sum())
        precision = true_positives / flagged if flagged else 0.0
        recall = true_positives / int(y_test.sum()) if y_test.sum() else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "flagged": flagged,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print("Done. No model artifacts were overwritten.")


if __name__ == "__main__":
    main()
