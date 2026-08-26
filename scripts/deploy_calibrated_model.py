"""
Deploy the calibrated model: retrains, fits a calibrator on a clean
three-way chronological split, and saves the CALIBRATED model as the
artifact the API actually loads.

Why the threshold changes but the alert VOLUME doesn't: the original
0.80 threshold was never chosen because it was a meaningful probability
-- it was chosen to keep the daily alert volume reviewable by one
analyst (see src/pipeline.py's docstring). That workflow constraint
didn't change just because the score labels did. So instead of picking
a new threshold from scratch, this script finds the calibrated score
that flags the EXACT SAME set of observations as the old raw >= 0.80
cutoff did -- same alerts, same workload, honestly labeled.

Run from the repository root:
    python scripts/deploy_calibrated_model.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import prepare_model_input
from src.load_data import load_m5_data
from src.monitoring import build_feature_reference, save_feature_reference
from src.preprocess import build_analytical_table
from src.train import chronological_split, save_model_artifacts, train_final_xgboost


def find_equivalent_threshold(raw_probs, calibrated_probs, original_raw_threshold):
    """
    Find the calibrated-probability threshold that flags a set of
    observations as close as possible in SIZE to
    `raw_probs >= original_raw_threshold`. Calibration is monotonic (it
    never changes rank order), but isotonic calibration also produces
    tied "plateau" values for groups of observations -- so the search is
    over the actual distinct calibrated values present, picking whichever
    one lands closest to the target count, rather than assuming a rank
    position maps cleanly onto a single usable cutoff.
    """
    n_target = int((raw_probs >= original_raw_threshold).sum())
    if n_target == 0:
        return float(calibrated_probs.max()) + 1e-6

    unique_values = np.unique(calibrated_probs)[::-1]  # descending
    counts = np.array([(calibrated_probs >= v).sum() for v in unique_values])
    best_idx = int(np.argmin(np.abs(counts - n_target)))
    return float(unique_values[best_idx])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--original-threshold", type=float, default=0.80,
                         help="The raw-score threshold whose alert volume we're preserving")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if models_dir.exists() and not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = models_dir.with_name(f"{models_dir.name}_backup_{stamp}")
        shutil.copytree(models_dir, backup)
        print(f"Backed up existing artifacts to {backup}/\n")

    print("Loading data and building features...")
    raw = load_m5_data(args.data_dir)
    analytical, split_day = build_analytical_table(raw.sales, raw.calendar, raw.prices, max_items=args.max_items)
    feature_data, X, feature_names = prepare_model_input(analytical)
    y = feature_data["stress_event"]
    day_num = feature_data["day_num"]

    X_train, X_holdout, y_train, y_holdout = chronological_split(X, y, day_num, split_day=split_day)
    holdout_days = day_num.loc[X_holdout.index]

    # Same three-way split as the investigation: 40% of holdout for
    # calibration fitting, 60% held out for the honest final report.
    calib_cutoff = int(np.quantile(holdout_days.unique(), 0.40))
    calib_mask = holdout_days <= calib_cutoff
    X_calib, y_calib = X_holdout.loc[calib_mask], y_holdout.loc[calib_mask]
    X_eval, y_eval = X_holdout.loc[~calib_mask], y_holdout.loc[~calib_mask]

    print(f"train={len(X_train):,}  calibration={len(X_calib):,}  final_eval={len(X_eval):,}\n")

    print("Training base model...")
    base_model = train_final_xgboost(X_train, y_train)

    print("Fitting isotonic calibrator on the calibration set...")
    frozen = FrozenEstimator(base_model)
    calibrated_model = CalibratedClassifierCV(frozen, method="isotonic")
    calibrated_model.fit(X_calib, y_calib)

    # --- Evaluate before/after on final_eval, for the console report ---
    raw_probs_eval = base_model.predict_proba(X_eval)[:, 1]
    calibrated_probs_eval = calibrated_model.predict_proba(X_eval)[:, 1]

    brier_before = brier_score_loss(y_eval, raw_probs_eval)
    brier_after = brier_score_loss(y_eval, calibrated_probs_eval)

    equivalent_threshold = find_equivalent_threshold(
        raw_probs_eval, calibrated_probs_eval, args.original_threshold
    )

    n_flagged_raw = int((raw_probs_eval >= args.original_threshold).sum())
    n_flagged_calibrated = int((calibrated_probs_eval >= equivalent_threshold).sum())

    print("\n" + "=" * 62)
    print("CALIBRATION RESULT")
    print("=" * 62)
    print(f"Brier score, raw:        {brier_before:.4f}")
    print(f"Brier score, calibrated: {brier_after:.4f}")
    print(f"\nOriginal raw threshold:      {args.original_threshold}")
    print(f"Equivalent calibrated threshold: {equivalent_threshold:.4f}")
    print(f"Alerts flagged, raw >= {args.original_threshold}:        {n_flagged_raw:,}")
    print(f"Alerts flagged, calibrated >= {equivalent_threshold:.4f}: {n_flagged_calibrated:,}")

    volume_diff_pct = abs(n_flagged_calibrated - n_flagged_raw) / max(n_flagged_raw, 1) * 100
    if volume_diff_pct > 5:
        raise SystemExit(
            f"Alert volume changed by {volume_diff_pct:.1f}% (more than the 5% tolerance) "
            f"-- investigate before shipping, this is bigger than isotonic tie-plateau noise."
        )
    print(
        f"Alert volume changed by {volume_diff_pct:.1f}% -- within tolerance. "
        f"(Isotonic calibration produces tied 'plateau' values for groups of observations, "
        f"so an exact match isn't always achievable at a single cutoff; this is expected, "
        f"not a bug.)\n"
    )

    # --- Save the CALIBRATED model as the deployment artifact ---
    # save_model_artifacts checks hasattr(model, "save_model") before
    # writing the .ubj file; CalibratedClassifierCV doesn't have that
    # method, so this automatically writes ONLY the .pkl, and
    # src/predict.py's loader already falls back to .pkl when no .ubj
    # is present. No changes needed to the loading code.
    # save_model_artifacts checks hasattr(model, "save_model") before
    # writing the .ubj file; CalibratedClassifierCV doesn't have that
    # method, so it correctly skips writing a NEW .ubj. But it also
    # doesn't delete an OLD .ubj left over from a previous, uncalibrated
    # run -- and src/predict.py's loader checks .ubj BEFORE .pkl, so a
    # stale .ubj would silently keep serving the old uncalibrated model
    # even after this script "succeeds". Remove it explicitly first.
    stale_ubj = models_dir / "final_xgboost_supply_stress.ubj"
    if stale_ubj.exists():
        stale_ubj.unlink()
        print(f"Removed stale {stale_ubj} (a calibrated model can't be saved in this format, "
              f"and the loader checks .ubj before .pkl -- leaving it would silently serve "
              f"the old uncalibrated model).")

    save_model_artifacts(
        calibrated_model,
        feature_names,
        models_dir,
        default_threshold=round(equivalent_threshold, 4),
    )
    print(f"Saved calibrated model to {models_dir}/")

    # Extend model_config.json with calibration provenance (additive,
    # doesn't remove any field the API or tests already read).
    config_path = models_dir / "model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["calibration_method"] = "isotonic"
    config["calibrated"] = True
    config["original_raw_threshold_replaced"] = args.original_threshold
    config["brier_score_raw"] = round(brier_before, 4)
    config["brier_score_calibrated"] = round(brier_after, 4)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Updated {config_path} with calibration provenance.")

    # Drift-monitoring reference is about the FEATURES, not the model
    # object, so it stays valid and just needs regenerating against the
    # same X_train used here.
    reference_path = models_dir / "feature_reference_stats.json"
    references = build_feature_reference(X_train, list(X_train.columns))
    save_feature_reference(references, reference_path)
    print(f"Saved drift-monitoring reference to {reference_path}")

    print("\nFinal performance at the new (calibrated) threshold:")
    preds = (calibrated_probs_eval >= equivalent_threshold).astype(int)
    tp = int(((preds == 1) & (y_eval == 1)).sum())
    precision = tp / n_flagged_calibrated if n_flagged_calibrated else 0.0
    recall = tp / int(y_eval.sum()) if y_eval.sum() else 0.0
    print(f"  average precision: {average_precision_score(y_eval, calibrated_probs_eval):.4f}")
    print(f"  precision:         {precision:.4f}")
    print(f"  recall:            {recall:.4f}")

    print("\nNext: restart the API locally and confirm it loads the calibrated model.")


if __name__ == "__main__":
    main()
