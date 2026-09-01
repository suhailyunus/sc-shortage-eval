"""Reproduce the paper's proxy-label confound and small calibration audits.

Run from the repository root::

    python scripts/paper_audit.py --data-dir data/raw

Outputs are written to ``paper_draft/generated`` by default.  The script
uses the complete M5 sales catalog for the confound audit and the project's
standard 100-item development sample for the calibration audit.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from scipy.stats import norm, pearsonr
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import prepare_model_input
from src.load_data import load_m5_data
from src.preprocess import build_analytical_table
from src.train import chronological_split, train_final_xgboost

TRAIN_FRACTION = 0.80
CALIBRATION_FRACTION_OF_HOLDOUT = 0.40
FN_COST = 20
FP_COST = 12
TP_REVIEW_COST = 10


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


def _fisher_interval(r: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 3:
        return float("nan"), float("nan")
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    half_width = norm.ppf(0.5 + confidence / 2) / np.sqrt(n - 3)
    return float(np.tanh(z - half_width)), float(np.tanh(z + half_width))


def audit_volume_confound(data_dir: Path, output_dir: Path) -> dict:
    """Compare item-only and item-store percentile proxy definitions."""
    sales_path = data_dir / "sales_train_validation.csv"
    sales = pd.read_csv(sales_path)
    day_columns = [c for c in sales.columns if c.startswith("d_")]
    day_columns.sort(key=lambda c: int(c.split("_")[1]))
    split_day = int(len(day_columns) * TRAIN_FRACTION)
    train_columns = day_columns[:split_day]
    holdout_columns = day_columns[split_day:]

    train_values = sales[train_columns].to_numpy()
    holdout_values = sales[holdout_columns].to_numpy()
    all_volume = sales[day_columns].mean(axis=1).to_numpy()

    item_only_threshold = np.empty(len(sales), dtype=float)
    for indices in sales.groupby("item_id", sort=False).indices.values():
        item_only_threshold[indices] = np.quantile(train_values[indices].ravel(), 0.90)
    item_store_threshold = np.quantile(train_values, 0.90, axis=1)

    item_only_rate = (holdout_values > item_only_threshold[:, None]).mean(axis=1)
    item_store_rate = (holdout_values > item_store_threshold[:, None]).mean(axis=1)

    series = pd.DataFrame(
        {
            "item_id": sales["item_id"],
            "store_id": sales["store_id"],
            "mean_daily_sales": all_volume,
            "item_only_rate": item_only_rate,
            "item_store_rate": item_store_rate,
        }
    )
    stores = series.groupby("store_id", sort=True)[
        ["mean_daily_sales", "item_only_rate", "item_store_rate"]
    ].mean()

    results: dict[str, object] = {
        "series_count": int(len(series)),
        "store_count": int(len(stores)),
        "split_day": split_day,
        "definitions": {},
    }
    for label, column in (
        ("item_only", "item_only_rate"),
        ("item_store", "item_store_rate"),
    ):
        store_test = pearsonr(stores["mean_daily_sales"], stores[column])
        low, high = _fisher_interval(float(store_test.statistic), len(stores))
        volume_residual = series["mean_daily_sales"] - series.groupby("item_id")[
            "mean_daily_sales"
        ].transform("mean")
        rate_residual = series[column] - series.groupby("item_id")[column].transform("mean")
        within_test = pearsonr(volume_residual, rate_residual)
        results["definitions"][label] = {
            "overall_holdout_rate": float(series[column].mean()),
            "store_rate_min": float(stores[column].min()),
            "store_rate_max": float(stores[column].max()),
            "store_level_pearson_r": float(store_test.statistic),
            "store_level_p_value_two_sided": float(store_test.pvalue),
            "store_level_fisher_95_ci": [low, high],
            "within_item_residual_pearson_r": float(within_test.statistic),
            "within_item_residual_p_value_two_sided": float(within_test.pvalue),
        }

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharex=True)
    for axis, column, title in (
        (axes[0], "item_only_rate", "Item-only threshold"),
        (axes[1], "item_store_rate", "Item-store threshold"),
    ):
        axis.scatter(stores["mean_daily_sales"], stores[column], color="#2b6cb0", s=45)
        for store_id, row in stores.iterrows():
            axis.annotate(
                store_id,
                (row["mean_daily_sales"], row[column]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set_title(title)
        axis.set_xlabel("Mean daily unit sales")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Holdout stress-event rate")
    figure.tight_layout()
    figure.savefig(output_dir / "store_volume_confound.pdf", bbox_inches="tight")
    plt.close(figure)

    _write_json(output_dir / "confound_audit.json", results)
    return results


def _frozen_isotonic(model):
    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    except ImportError:  # scikit-learn before 1.6
        return CalibratedClassifierCV(model, method="isotonic", cv="prefit")


def _metrics(y_true: np.ndarray | pd.Series, probabilities: np.ndarray) -> dict:
    prevalence = float(np.mean(y_true))
    reference = prevalence * (1.0 - prevalence)
    brier = float(brier_score_loss(y_true, probabilities))
    return {
        "prevalence": prevalence,
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": brier,
        "prevalence_only_brier": reference,
        "brier_skill_score": float(1.0 - brier / reference),
    }


def _best_incremental_threshold(y_true, probabilities) -> dict:
    """Maximize (FN cost - TP review cost)*TP - FP cost*FP.

    Only endpoints of tied score groups are candidates.  A no-alert policy
    with zero incremental value is included, so a negative policy is never
    reported as optimal.
    """
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probabilities, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    sorted_p, sorted_y = p[order], y[order]
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1 - sorted_y)
    group_ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(sorted_p) - 1]
    nets = (FN_COST - TP_REVIEW_COST) * tp[group_ends] - FP_COST * fp[group_ends]
    best = int(np.argmax(nets))
    if float(nets[best]) <= 0:
        return {"threshold": None, "incremental_net": 0, "alerts": 0}
    endpoint = int(group_ends[best])
    return {
        "threshold": float(sorted_p[endpoint]),
        "incremental_net": int(nets[best]),
        "alerts": endpoint + 1,
    }


def audit_small_calibration(data_dir: Path, output_dir: Path, max_items: int = 100) -> dict:
    raw = load_m5_data(data_dir)
    analytical, split_day = build_analytical_table(
        raw.sales, raw.calendar, raw.prices, max_items=max_items
    )
    feature_data, X, feature_names = prepare_model_input(analytical)
    y = feature_data["stress_event"]
    day_num = feature_data["day_num"]
    X_train, X_holdout, y_train, y_holdout = chronological_split(
        X, y, day_num, split_day=split_day
    )
    holdout_days = day_num.loc[X_holdout.index]
    calibration_cutoff = int(
        np.quantile(np.sort(holdout_days.unique()), CALIBRATION_FRACTION_OF_HOLDOUT)
    )
    calibration_mask = holdout_days <= calibration_cutoff
    X_calibration = X_holdout.loc[calibration_mask]
    y_calibration = y_holdout.loc[calibration_mask]
    X_final = X_holdout.loc[~calibration_mask]
    y_final = y_holdout.loc[~calibration_mask]

    model = train_final_xgboost(X_train, y_train)
    calibrator = _frozen_isotonic(model)
    calibrator.fit(X_calibration, y_calibration)

    raw_calibration = model.predict_proba(X_calibration)[:, 1]
    isotonic_calibration = calibrator.predict_proba(X_calibration)[:, 1]
    raw_final = model.predict_proba(X_final)[:, 1]
    isotonic_final = calibrator.predict_proba(X_final)[:, 1]

    logistic = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1_000, random_state=42)
    )
    logistic.fit(X_train, y_train)
    logistic_final = logistic.predict_proba(X_final)[:, 1]

    raw_policy = _best_incremental_threshold(y_calibration, raw_calibration)
    isotonic_policy = _best_incremental_threshold(y_calibration, isotonic_calibration)

    result = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "max_items": max_items,
        "feature_count": len(feature_names),
        "split_day": split_day,
        "calibration_cutoff_day": calibration_cutoff,
        "rows": {
            "train": len(X_train),
            "calibration": len(X_calibration),
            "final_evaluation": len(X_final),
        },
        "final_evaluation": {
            "raw_xgboost": _metrics(y_final, raw_final),
            "isotonic_xgboost": _metrics(y_final, isotonic_final),
            "unweighted_logistic_baseline": _metrics(y_final, logistic_final),
        },
        "thresholds_selected_on_calibration": {
            "raw": raw_policy,
            "isotonic": isotonic_policy,
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
    figure.savefig(output_dir / "calibration_100_item.pdf", bbox_inches="tight")
    plt.close(figure)

    _write_json(output_dir / "calibration_100_item.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "paper_draft" / "generated"
    )
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--skip-confound", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_confound:
        result = audit_volume_confound(args.data_dir, args.output_dir)
        old = result["definitions"]["item_only"]
        new = result["definitions"]["item_store"]
        print(
            "Confound audit: "
            f"store r={old['store_level_pearson_r']:.6f} -> "
            f"{new['store_level_pearson_r']:.6f}"
        )
    if not args.skip_calibration:
        result = audit_small_calibration(args.data_dir, args.output_dir, args.max_items)
        final = result["final_evaluation"]
        print(
            "Calibration audit: "
            f"Brier {final['raw_xgboost']['brier_score']:.6f} -> "
            f"{final['isotonic_xgboost']['brier_score']:.6f}"
        )


if __name__ == "__main__":
    main()
