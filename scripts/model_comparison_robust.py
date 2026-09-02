"""
Robustness check: is the conclusion in this paper model-family-independent?

Rerun of the original model_comparison.py, fixed to match the rigor
established elsewhere in this paper rather than the paper's own earlier,
looser standard:

  1. Same three-way chronological split as Section "Probability
     Calibration" (train / calibration / final_eval), not a plain 80/20
     holdout -- so the reported Brier score is directly comparable to
     Table "Final-evaluation probability quality" instead of coming
     from a differently-composed holdout.
  2. The corrected item-store label (already the default in
     src/preprocess.py -- unchanged from the rest of the paper).
  3. Complexity matched by EARLY STOPPING on the calibration period
     (a held-out validation set the model never trains on), not by
     forcing identical max_depth/n_estimators across three libraries
     with different regularization behavior.
  4. Three seeds (varying each model's own random_state / bagging
     randomness -- subsample/colsample are set below 1.0 specifically
     so random_state actually produces different models across seeds;
     leaving them at their 1.0 defaults was tried first and produced
     identical results across seeds, a real bug caught by inspecting
     the output rather than assuming the seed loop worked).
  5. Proper Brier Skill Score against the prevalence-only reference
     p(1-p), not the incorrect "0.25" reference used in the original
     (now-removed) model comparison section.
  6. Oracle (threshold selected AND evaluated on final_eval -- upper
     bound diagnostic) reported separately from ex-ante (threshold
     selected on the calibration period, evaluated once on final_eval),
     matching the same separation used throughout Section
     "Probability Calibration".
  7. A logistic regression baseline recomputed under this exact split,
     rather than citing the single-split number from the calibration
     section, so this table is internally self-consistent.
  8. BSS is reported on ISOTONIC-CALIBRATED scores, not raw scores --
     raw class-weighted scores from every boosted-tree model here have
     deeply negative raw-score BSS despite good ranking (AP), an
     expected consequence of severe raw overconfidence (consistent with
     this paper's own Table "Final-evaluation probability quality"
     raw-vs-reference Brier gap), not a defect specific to this script.

Run from the repository root:
    python scripts/model_comparison_robust.py <seed>
    python scripts/model_comparison_robust.py  # runs all 3 seeds + aggregates
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import prepare_model_input
from src.load_data import load_m5_data
from src.preprocess import build_analytical_table
from src.train import calculate_scale_pos_weight, chronological_split

FN_COST, FP_COST, TP_COST = 20, 12, 10
SEEDS = [42, 7, 123]
OUT_DIR = Path("paper_draft/generated")


def net_at_threshold(y_true, probs, threshold):
    if threshold is None:
        return 0  # no-alert policy: net is 0 by definition, nothing to compute
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    return (FN_COST - TP_COST) * tp - FP_COST * fp


def best_threshold_and_net(y_true, probs):
    """
    Select the best attainable single-threshold policy, including the
    always-available no-alert option (net=0).

    Adopted from scripts/paper_audit.py and
    scripts/train_calibrate_full_catalog.py's identical implementation,
    for full methodological consistency across every table in the paper
    -- an earlier version of this function used a coarse 0.01-step grid
    search (np.arange(0.02, 0.99, 0.01)) with no explicit no-alert
    floor, which could report an impossible negative "oracle" value
    when a model's highest-scored observation was itself a false
    positive and no grid point was high enough to exclude it. Testing
    every achievable distinct threshold (via tied-score group
    endpoints) rather than a fixed grid is also strictly more precise:
    it cannot miss a better threshold that happened to fall between
    grid points.
    """
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probs, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    sorted_p, sorted_y = p[order], y[order]
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1 - sorted_y)
    group_ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(sorted_p) - 1]
    nets = (FN_COST - TP_COST) * tp[group_ends] - FP_COST * fp[group_ends]
    best = int(np.argmax(nets))
    if float(nets[best]) <= 0:
        return None, 0  # no-alert policy: always achievable, net=0 by definition
    endpoint = int(group_ends[best])
    return float(sorted_p[endpoint]), int(nets[best])


def brier_skill_score(y_true, probs):
    prevalence = float(y_true.mean())
    reference_bs = prevalence * (1 - prevalence)
    model_bs = brier_score_loss(y_true, probs)
    bss = 1 - model_bs / reference_bs if reference_bs > 0 else float("nan")
    return prevalence, reference_bs, model_bs, bss


def fit_with_early_stopping(model_name, X_train, y_train, X_calib, y_calib, scale_pos_weight, seed):
    if model_name == "xgboost":
        model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, objective="binary:logistic",
            eval_metric="logloss", random_state=seed, n_jobs=-1,
            early_stopping_rounds=20,
        )
        model.fit(X_train, y_train, eval_set=[(X_calib, y_calib)], verbose=False)
        return model

    if model_name == "lightgbm":
        import lightgbm as lgb
        model = LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, objective="binary",
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        model.fit(
            X_train, y_train, eval_set=[(X_calib, y_calib)],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        return model

    if model_name == "catboost":
        model = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bylevel=0.8, bootstrap_type="Bernoulli",
            scale_pos_weight=scale_pos_weight, loss_function="Logloss",
            random_state=seed, verbose=False, early_stopping_rounds=20,
        )
        model.fit(X_train, y_train, eval_set=(X_calib, y_calib))
        return model

    raise ValueError(model_name)


def evaluate_model(model, X_calib, y_calib, X_eval, y_eval):
    """
    Mirrors this paper's own Section "Probability Calibration" exactly:
    report raw Brier for reference, then fit an isotonic calibrator on
    the calibration period and report Brier Skill Score on the
    CALIBRATED scores.
    """
    probs_calib_raw = model.predict_proba(X_calib)[:, 1]
    probs_eval_raw = model.predict_proba(X_eval)[:, 1]

    ap = average_precision_score(y_eval, probs_eval_raw)
    prevalence, ref_bs, raw_bs, _ = brier_skill_score(y_eval, probs_eval_raw)

    frozen = FrozenEstimator(model)
    calibrator = CalibratedClassifierCV(frozen, method="isotonic")
    calibrator.fit(X_calib, y_calib)
    probs_eval_calibrated = calibrator.predict_proba(X_eval)[:, 1]
    _, _, calibrated_bs, calibrated_bss = brier_skill_score(y_eval, probs_eval_calibrated)

    ex_ante_threshold, _ = best_threshold_and_net(y_calib, probs_calib_raw)
    ex_ante_net = net_at_threshold(y_eval, probs_eval_raw, ex_ante_threshold)
    oracle_threshold, oracle_net = best_threshold_and_net(y_eval, probs_eval_raw)

    return {
        "average_precision": round(float(ap), 4),
        "prevalence": round(prevalence, 4),
        "reference_brier": round(ref_bs, 4),
        "raw_brier": round(raw_bs, 4),
        "calibrated_brier": round(calibrated_bs, 4),
        "calibrated_bss": round(calibrated_bss, 4),
        "ex_ante_threshold": round(ex_ante_threshold, 4) if ex_ante_threshold is not None else None,
        "ex_ante_net": ex_ante_net,
        "oracle_threshold": round(oracle_threshold, 4) if oracle_threshold is not None else None,
        "oracle_net": oracle_net,
    }


def run_one_seed(seed, X_train, y_train, X_calib, y_calib, X_eval, y_eval, scale_pos_weight):
    results = {}

    for model_name in ["xgboost", "lightgbm", "catboost"]:
        model = fit_with_early_stopping(model_name, X_train, y_train, X_calib, y_calib, scale_pos_weight, seed)
        results[model_name] = evaluate_model(model, X_calib, y_calib, X_eval, y_eval)

    logreg_pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
    ])
    logreg_pipeline.fit(X_train, y_train)
    results["logistic_regression"] = evaluate_model(logreg_pipeline, X_calib, y_calib, X_eval, y_eval)

    return results


def main():
    import sys

    print("Loading data and building features (100-item / 1,000-series scale, corrected label)...")
    raw = load_m5_data("data/raw")
    analytical, split_day = build_analytical_table(raw.sales, raw.calendar, raw.prices, max_items=100)
    feature_data, X, feature_names = prepare_model_input(analytical)
    y = feature_data["stress_event"]
    day_num = feature_data["day_num"]

    X_train, X_holdout, y_train, y_holdout = chronological_split(X, y, day_num, split_day=split_day)
    holdout_days = day_num.loc[X_holdout.index]

    calib_cutoff = int(np.quantile(holdout_days.unique(), 0.40))
    calib_mask = holdout_days <= calib_cutoff
    X_calib, y_calib = X_holdout.loc[calib_mask], y_holdout.loc[calib_mask]
    X_eval, y_eval = X_holdout.loc[~calib_mask], y_holdout.loc[~calib_mask]

    print(f"train={len(X_train):,}  calibration={len(X_calib):,}  final_eval={len(X_eval):,}")

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    print(f"scale_pos_weight: {scale_pos_weight:.2f}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    requested_seeds = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else SEEDS

    for seed in requested_seeds:
        checkpoint_path = OUT_DIR / f"model_comparison_seed_{seed}.json"
        if checkpoint_path.exists():
            print(f"Seed {seed}: already done, skipping")
            continue
        print(f"=== Seed {seed} ===")
        result = run_one_seed(seed, X_train, y_train, X_calib, y_calib, X_eval, y_eval, scale_pos_weight)
        for model_name, r in result.items():
            print(f"  {model_name:<20} AP={r['average_precision']:.4f}  "
                  f"BSS={r['calibrated_bss']:.4f}  "
                  f"ex-ante net=${r['ex_ante_net']:+,}  oracle net=${r['oracle_net']:+,}")
        with open(checkpoint_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved {checkpoint_path}\n")

    if not all((OUT_DIR / f"model_comparison_seed_{s}.json").exists() for s in SEEDS):
        print("Not all seeds complete yet -- run remaining seeds before aggregating.")
        return

    all_seed_results = {}
    for seed in SEEDS:
        with open(OUT_DIR / f"model_comparison_seed_{seed}.json") as f:
            all_seed_results[seed] = json.load(f)

    print("\n" + "=" * 78)
    print("AGGREGATE ACROSS 3 SEEDS (mean, range)")
    print("=" * 78)
    summary = {}
    for model_name in ["xgboost", "lightgbm", "catboost", "logistic_regression"]:
        aps = [all_seed_results[s][model_name]["average_precision"] for s in SEEDS]
        bsss = [all_seed_results[s][model_name]["calibrated_bss"] for s in SEEDS]
        ex_ante_nets = [all_seed_results[s][model_name]["ex_ante_net"] for s in SEEDS]
        oracle_nets = [all_seed_results[s][model_name]["oracle_net"] for s in SEEDS]

        summary[model_name] = {
            "ap_mean": round(float(np.mean(aps)), 4),
            "ap_range": [round(float(min(aps)), 4), round(float(max(aps)), 4)],
            "bss_mean": round(float(np.mean(bsss)), 4),
            "bss_range": [round(float(min(bsss)), 4), round(float(max(bsss)), 4)],
            "ex_ante_net_mean": round(float(np.mean(ex_ante_nets)), 0),
            "ex_ante_net_range": [min(ex_ante_nets), max(ex_ante_nets)],
            "oracle_net_mean": round(float(np.mean(oracle_nets)), 0),
            "oracle_net_range": [min(oracle_nets), max(oracle_nets)],
        }
        s = summary[model_name]
        print(f"{model_name:<20} AP={s['ap_mean']:.4f} {s['ap_range']}  "
              f"BSS={s['bss_mean']:.4f} {s['bss_range']}  "
              f"ex-ante net=${s['ex_ante_net_mean']:+,.0f} {s['ex_ante_net_range']}  "
              f"oracle net=${s['oracle_net_mean']:+,.0f} {s['oracle_net_range']}")

    with open(OUT_DIR / "model_comparison_robust.json", "w") as f:
        json.dump({"per_seed": all_seed_results, "summary": summary}, f, indent=2)
    print(f"\nSaved {OUT_DIR / 'model_comparison_robust.json'}")


if __name__ == "__main__":
    main()
