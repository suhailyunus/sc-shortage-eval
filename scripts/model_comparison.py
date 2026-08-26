"""
Model comparison: is XGBoost actually the best choice, or just the
first one tried? Flagged as open in project_notes.md since the earliest
exploratory phase of this project.

Fairness rules, so this comparison actually means something:
- Same data, same features, same leakage-safe stress target as the
  shipped model.
- Same chronological train/holdout split.
- Same class-imbalance approach: scale_pos_weight computed identically
  for all three models (each library's own parameter name for it).
- Same evaluation: average precision, Brier score (raw/uncalibrated,
  matching how the shipped model was first evaluated before the
  calibration work), and business-impact ROI-optimal threshold under
  the same cost assumptions used everywhere else in this project.
- Comparable model complexity: ~200 trees, similar depth, across all
  three -- not tuning one harder than the others.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score
from xgboost import XGBClassifier

from src.features import prepare_model_input
from src.load_data import load_m5_data
from src.preprocess import build_analytical_table
from src.train import calculate_scale_pos_weight, chronological_split

FN_COST, FP_COST, TP_COST = 20, 12, 10
REPORTS_DIR = Path("reports/figures")


def business_impact_sweep(y_true, probs):
    do_nothing_cost = int(y_true.sum()) * FN_COST
    best_net, best_threshold = None, None
    for threshold in np.arange(0.02, 0.99, 0.01):
        preds = (probs >= threshold).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        net = do_nothing_cost - (tp * TP_COST + fp * FP_COST + fn * FN_COST)
        if best_net is None or net > best_net:
            best_net, best_threshold = net, threshold
    return best_threshold, best_net


def evaluate(name, model, X_holdout, y_holdout):
    probs = model.predict_proba(X_holdout)[:, 1]
    ap = average_precision_score(y_holdout, probs)
    brier = brier_score_loss(y_holdout, probs)

    preds_at_50 = (probs >= 0.50).astype(int)
    prec_50 = precision_score(y_holdout, preds_at_50, zero_division=0)
    rec_50 = recall_score(y_holdout, preds_at_50, zero_division=0)

    roi_threshold, roi_net = business_impact_sweep(y_holdout, probs)

    print(f"\n--- {name} ---")
    print(f"  Average precision:     {ap:.4f}")
    print(f"  Brier score (raw):     {brier:.4f}")
    print(f"  Precision/recall @0.50: {prec_50:.3f} / {rec_50:.3f}")
    print(f"  ROI-optimal:           threshold={roi_threshold:.2f}, net=${roi_net:+,.0f}")

    return {
        "average_precision": round(ap, 4),
        "brier_score_raw": round(brier, 4),
        "precision_at_050": round(prec_50, 4),
        "recall_at_050": round(rec_50, 4),
        "roi_optimal_threshold": round(float(roi_threshold), 2),
        "roi_optimal_net": roi_net,
    }


def main():
    print("Loading data and building features (max_items=100, matching the shipped model)...")
    raw = load_m5_data("data/raw")
    analytical, split_day = build_analytical_table(raw.sales, raw.calendar, raw.prices, max_items=100)
    feature_data, X, feature_names = prepare_model_input(analytical)
    y = feature_data["stress_event"]

    X_train, X_holdout, y_train, y_holdout = chronological_split(
        X, y, feature_data["day_num"], split_day=split_day
    )
    print(f"train={len(X_train):,}  holdout={len(X_holdout):,}\n")

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    print(f"scale_pos_weight (shared across all three models): {scale_pos_weight:.2f}")

    results = {}

    # --- XGBoost: the actual production model, exact same hyperparameters ---
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.03,
        min_child_weight=3, gamma=0, subsample=0.9, colsample_bytree=1.0,
        scale_pos_weight=scale_pos_weight, objective="binary:logistic",
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    results["xgboost"] = evaluate("XGBoost (production baseline)", xgb_model, X_holdout, y_holdout)

    # --- LightGBM: comparable complexity, same imbalance handling ---
    lgbm_model = LGBMClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.03,
        min_child_weight=3, subsample=0.9, colsample_bytree=1.0,
        scale_pos_weight=scale_pos_weight, objective="binary", random_state=42,
        n_jobs=-1, verbose=-1,
    )
    lgbm_model.fit(X_train, y_train)
    results["lightgbm"] = evaluate("LightGBM", lgbm_model, X_holdout, y_holdout)

    # --- CatBoost: comparable complexity, same imbalance handling ---
    cat_model = CatBoostClassifier(
        iterations=200, depth=8, learning_rate=0.03,
        subsample=0.9, colsample_bylevel=1.0,
        scale_pos_weight=scale_pos_weight, loss_function="Logloss",
        random_state=42, verbose=False,
    )
    cat_model.fit(X_train, y_train)
    results["catboost"] = evaluate("CatBoost", cat_model, X_holdout, y_holdout)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<12}{'Avg Precision':<16}{'Brier (raw)':<14}{'ROI-optimal net':<18}")
    for name, r in results.items():
        print(f"{name:<12}{r['average_precision']:<16}{r['brier_score_raw']:<14}${r['roi_optimal_net']:<17,}")

    best_by_ap = max(results, key=lambda k: results[k]["average_precision"])
    best_by_net = max(results, key=lambda k: results[k]["roi_optimal_net"])
    print(f"\nBest by average precision: {best_by_ap}")
    print(f"Best by ROI-optimal net:   {best_by_net}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {REPORTS_DIR / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
