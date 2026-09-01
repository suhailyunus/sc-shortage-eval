"""Train, calibrate, and evaluate XGBoost on full-catalog chunk files.

This script requires chunks produced by ``build_full_catalog_chunks.py``
after that builder was updated to retain ``_day_num`` in both train and
holdout parquet files.  The chronological flow is:

    train -> calibration/threshold selection -> final evaluation

Run from the repository root::

    python scripts/train_calibrate_full_catalog.py --chunk-dir .
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FN_COST = 20
FP_COST = 12
TP_REVIEW_COST = 10
CALIBRATION_FRACTION_OF_HOLDOUT = 0.40
METADATA_COLUMNS = {"stress_event", "_day_num"}


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _frozen_isotonic(model):
    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    except ImportError:  # scikit-learn before 1.6
        return CalibratedClassifierCV(model, method="isotonic", cv="prefit")


def _fit_xgboost(X_train: np.ndarray, y_train: np.ndarray, n_jobs: int):
    scale_pos_weight = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
    model = xgboost.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def _load_training(files: list[str], feature_columns: list[str]):
    frames = []
    for file_name in files:
        frame = pd.read_parquet(file_name, columns=feature_columns + ["stress_event"])
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    X = combined[feature_columns].to_numpy(dtype="float32")
    y = combined["stress_event"].to_numpy(dtype="int8")
    del combined, frames
    gc.collect()
    return X, y


def _holdout_cutoff(files: list[str]) -> int:
    days = set()
    for file_name in files:
        values = pd.read_parquet(file_name, columns=["_day_num"])["_day_num"].unique()
        days.update(int(value) for value in values)
    ordered = np.array(sorted(days))
    return int(np.quantile(ordered, CALIBRATION_FRACTION_OF_HOLDOUT))


def _load_period(
    files: list[str], feature_columns: list[str], cutoff: int, calibration: bool
):
    X_parts, y_parts = [], []
    for file_name in files:
        frame = pd.read_parquet(
            file_name, columns=feature_columns + ["stress_event", "_day_num"]
        )
        mask = frame["_day_num"] <= cutoff
        if not calibration:
            mask = ~mask
        selected = frame.loc[mask]
        X_parts.append(selected[feature_columns].to_numpy(dtype="float32"))
        y_parts.append(selected["stress_event"].to_numpy(dtype="int8"))
        del frame, selected
        gc.collect()
    return np.concatenate(X_parts), np.concatenate(y_parts)


def _probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    prevalence = float(y_true.mean())
    reference = prevalence * (1.0 - prevalence)
    brier = float(brier_score_loss(y_true, probabilities))
    return {
        "prevalence": prevalence,
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": brier,
        "prevalence_only_brier": reference,
        "brier_skill_score": float(1.0 - brier / reference),
    }


def _select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    """Select the best attainable single-threshold policy, including no action."""
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_p = probabilities[order]
    sorted_y = y_true[order].astype(np.int8, copy=False)
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1 - sorted_y)
    group_ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(sorted_p) - 1]
    nets = (FN_COST - TP_REVIEW_COST) * tp[group_ends] - FP_COST * fp[group_ends]
    best = int(np.argmax(nets))
    if int(nets[best]) <= 0:
        return {"threshold": None, "incremental_net": 0, "alerts": 0}
    endpoint = int(group_ends[best])
    return {
        "threshold": float(sorted_p[endpoint]),
        "incremental_net": int(nets[best]),
        "alerts": endpoint + 1,
    }


def _evaluate_policy(
    y_true: np.ndarray, probabilities: np.ndarray, policy: dict
) -> dict:
    threshold = policy["threshold"]
    if threshold is None:
        return {
            "threshold": None,
            "incremental_net": 0,
            "alerts": 0,
            "true_positives": 0,
            "false_positives": 0,
        }
    predicted = probabilities >= threshold
    tp = int(np.sum(predicted & (y_true == 1)))
    fp = int(np.sum(predicted & (y_true == 0)))
    return {
        "threshold": float(threshold),
        "incremental_net": int((FN_COST - TP_REVIEW_COST) * tp - FP_COST * fp),
        "alerts": int(predicted.sum()),
        "true_positives": tp,
        "false_positives": fp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "paper_draft" / "generated"
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    train_files = sorted(glob.glob(str(args.chunk_dir / "chunk_train_*.parquet")))
    holdout_files = sorted(glob.glob(str(args.chunk_dir / "chunk_holdout_*.parquet")))
    if not train_files or not holdout_files:
        raise FileNotFoundError(
            "Missing chunk_train_*.parquet or chunk_holdout_*.parquet files. "
            "Run scripts/build_full_catalog_chunks.py first."
        )
    if len(train_files) != len(holdout_files):
        raise ValueError("Training and holdout chunk counts do not match.")

    schema = pd.read_parquet(train_files[0]).columns.tolist()
    feature_columns = [c for c in schema if c not in METADATA_COLUMNS and not c.startswith("_")]
    holdout_schema = pd.read_parquet(holdout_files[0]).columns
    if "_day_num" not in holdout_schema:
        raise ValueError(
            "Holdout chunks do not contain _day_num. Update and rerun "
            "scripts/build_full_catalog_chunks.py before calibration."
        )

    print(f"Loading {len(train_files)} training chunks...")
    X_train, y_train = _load_training(train_files, feature_columns)
    print(f"Training rows: {len(y_train):,}; features: {len(feature_columns)}")
    model = _fit_xgboost(X_train, y_train, args.n_jobs)
    del X_train, y_train
    gc.collect()

    cutoff = _holdout_cutoff(holdout_files)
    print(f"Calibration cutoff day: {cutoff}")
    X_calibration, y_calibration = _load_period(
        holdout_files, feature_columns, cutoff, calibration=True
    )
    print(f"Calibration rows: {len(y_calibration):,}")
    calibrator = _frozen_isotonic(model)
    calibrator.fit(X_calibration, y_calibration)

    raw_calibration = model.predict_proba(X_calibration)[:, 1]
    isotonic_calibration = calibrator.predict_proba(X_calibration)[:, 1]
    raw_policy = _select_threshold(y_calibration, raw_calibration)
    isotonic_policy = _select_threshold(y_calibration, isotonic_calibration)
    del X_calibration, raw_calibration, isotonic_calibration
    gc.collect()

    X_final, y_final = _load_period(
        holdout_files, feature_columns, cutoff, calibration=False
    )
    print(f"Final-evaluation rows: {len(y_final):,}")
    raw_final = model.predict_proba(X_final)[:, 1]
    isotonic_final = calibrator.predict_proba(X_final)[:, 1]
    del X_final
    gc.collect()

    results = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "feature_columns": feature_columns,
        "calibration_cutoff_day": cutoff,
        "rows": {
            "calibration": len(y_calibration),
            "final_evaluation": len(y_final),
        },
        "final_evaluation": {
            "raw": _probability_metrics(y_final, raw_final),
            "isotonic": _probability_metrics(y_final, isotonic_final),
        },
        "policy_selected_on_calibration_and_evaluated_on_final": {
            "raw": _evaluate_policy(y_final, raw_final, raw_policy),
            "isotonic": _evaluate_policy(y_final, isotonic_final, isotonic_policy),
        },
        "selection_period_policies": {"raw": raw_policy, "isotonic": isotonic_policy},
        "retrospective_final_oracle": {
            "raw": _select_threshold(y_final, raw_final),
            "isotonic": _select_threshold(y_final, isotonic_final),
        },
        "cost_assumptions": {
            "missed_event": FN_COST,
            "false_alert": FP_COST,
            "reviewed_true_positive": TP_REVIEW_COST,
            "incremental_formula": "(20-10)*TP - 12*FP",
        },
    }

    fraction_raw, mean_raw = calibration_curve(
        y_final, raw_final, n_bins=10, strategy="quantile"
    )
    fraction_iso, mean_iso = calibration_curve(
        y_final, isotonic_final, n_bins=10, strategy="quantile"
    )
    figure, axis = plt.subplots(figsize=(5.4, 4.4))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.plot(mean_raw, fraction_raw, marker="o", label="Raw XGBoost")
    axis.plot(mean_iso, fraction_iso, marker="o", label="Isotonic")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed event rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_dir / "calibration_full_catalog.pdf", bbox_inches="tight")
    plt.close(figure)
    _write_json(args.output_dir / "calibration_full_catalog.json", results)

    raw_metrics = results["final_evaluation"]["raw"]
    iso_metrics = results["final_evaluation"]["isotonic"]
    print(
        f"Brier: {raw_metrics['brier_score']:.6f} -> "
        f"{iso_metrics['brier_score']:.6f}; "
        f"isotonic BSS={iso_metrics['brier_skill_score']:.6f}"
    )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
