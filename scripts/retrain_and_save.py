"""
Retrain on the corrected chronological split and write the deployment
artifacts that the API loads at startup.

This OVERWRITES models/. The previous artifacts were trained against a
leaking label and a holdout that partitioned by item rather than by
time, so they are not worth preserving, but the script takes a backup
anyway.

Run from the repository root:

    python scripts/retrain_and_save.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitoring import build_feature_reference, save_feature_reference  # noqa: E402
from src.pipeline import run_training_pipeline  # noqa: E402


def _log_to_mlflow(args, result, precision: float, recall: float, average_precision: float) -> None:
    """Log this run's params and metrics to MLflow, if it's installed.

    This is intentionally optional and best-effort: MLflow is a reporting
    aid, not a training dependency, so a missing package or an
    unreachable tracking server must not fail the actual retraining run.
    """

    try:
        import mlflow
    except ImportError:
        print("\n--mlflow was passed but the mlflow package is not installed; skipping.")
        return

    try:
        mlflow.set_experiment("supply-chain-stress-prediction")
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "max_items": args.max_items,
                    "threshold": args.threshold,
                    "n_train_rows": len(result.X_train),
                    "n_test_rows": len(result.X_test),
                    "split_day": result.split_day,
                    "model_type": type(result.model).__name__,
                }
            )
            mlflow.log_metrics(
                {
                    "average_precision": average_precision,
                    "precision_at_threshold": precision,
                    "recall_at_threshold": recall,
                }
            )
    except Exception as exc:  # pragma: no cover - depends on external tracking server
        print(f"\nMLflow logging failed ({exc}); the saved model artifacts are unaffected.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up the existing models directory.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log params and metrics for this run to MLflow (requires the mlflow package).",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)

    if models_dir.exists() and not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = models_dir.with_name(f"{models_dir.name}_backup_{stamp}")
        shutil.copytree(models_dir, backup)
        print(f"Backed up existing artifacts to {backup}/\n")

    print(f"Retraining, writing to {models_dir}/ at threshold {args.threshold}...\n")

    result = run_training_pipeline(
        args.data_dir,
        max_items=args.max_items,
        models_dir=models_dir,
        default_threshold=args.threshold,
    )

    y_test = result.y_test.to_numpy()
    scores = result.model.predict_proba(result.X_test)[:, 1]
    predictions = (scores >= args.threshold).astype(int)

    flagged = int(predictions.sum())
    true_positives = int(((predictions == 1) & (y_test == 1)).sum())
    precision = true_positives / flagged if flagged else 0.0
    recall = true_positives / int(y_test.sum()) if y_test.sum() else 0.0

    test_days = result.feature_data.loc[result.X_test.index, "day_num"]
    n_days = int(test_days.nunique())

    print("=" * 62)
    print(f"PERFORMANCE AT THE SAVED THRESHOLD ({args.threshold})")
    print("=" * 62)
    print(f"test base rate        : {y_test.mean():.4f}")
    print(f"average precision     : {average_precision_score(y_test, scores):.4f}")
    print(f"alerts raised         : {flagged:,}  (~{flagged / n_days:.0f} per day)")
    print(f"precision             : {precision:.4f}")
    print(f"recall                : {recall:.4f}")
    print(f"events missed         : {int(y_test.sum()) - true_positives:,}")

    print()
    print("=" * 62)
    print("ARTIFACTS WRITTEN")
    print("=" * 62)
    for path in sorted(models_dir.glob("*")):
        if path.is_file():
            print(f"  {path.name:45s} {path.stat().st_size / 1024:8.1f} KB")

    config_path = models_dir / "model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"\n{config_path}:")
    print(json.dumps(config, indent=2))

    if not np.isclose(config["default_threshold"], args.threshold):
        raise SystemExit(
            f"Saved threshold {config['default_threshold']} does not match "
            f"the requested {args.threshold}."
        )

    print("\nSaved threshold matches the requested value.")

    reference_path = models_dir / "feature_reference_stats.json"
    references = build_feature_reference(result.X_train, list(result.X_train.columns))
    save_feature_reference(references, reference_path)
    print(f"\nSaved drift-monitoring reference distribution to {reference_path}")

    if args.mlflow:
        _log_to_mlflow(args, result, precision, recall, average_precision_score(y_test, scores))

    print("\nNext: restart the API locally and confirm it loads the new model.")


if __name__ == "__main__":
    main()
