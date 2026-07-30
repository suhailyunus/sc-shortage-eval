"""
Translate holdout performance into dollar terms under stated cost assumptions.

The cost figures below are PLACEHOLDERS, not researched numbers - this
project has no access to real supply-chain cost data. Replace them with
figures you can defend before using this output anywhere. Everything
downstream (README, API, resume bullets) should say "under these cost
assumptions," never "the model saves $X."

Run from the repository root:

    python scripts/report_business_impact.py \
        --cost-fn 2500 --cost-fp 150 --cost-tp 300

Writes artifacts/business_impact.json, which api/main.py serves as a
static (not live) reference via GET /business-impact.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.business_impact import CostAssumptions, compare_thresholds, evaluate_business_impact  # noqa: E402
from src.pipeline import run_training_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument(
        "--cost-fn",
        type=float,
        required=True,
        help="Dollar cost of a missed stress event (false negative). No default: state your own.",
    )
    parser.add_argument(
        "--cost-fp",
        type=float,
        required=True,
        help="Dollar cost of a false alarm (false positive). No default: state your own.",
    )
    parser.add_argument(
        "--cost-tp",
        type=float,
        required=True,
        help="Dollar cost of mitigating a correctly caught event (true positive).",
    )
    parser.add_argument("--output", default="artifacts/business_impact.json")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Also sweep thresholds 0.05-0.95 and report the ROI-optimal "
            "point. The shipped threshold is chosen for precision/recall, "
            "not dollar ROI, and the two do not necessarily agree."
        ),
    )
    args = parser.parse_args()

    print("Retraining on the corrected chronological split...\n")
    result = run_training_pipeline(
        args.data_dir,
        max_items=args.max_items,
        models_dir=None,
    )

    scores = result.model.predict_proba(result.X_test)[:, 1]
    costs = CostAssumptions(
        cost_false_negative=args.cost_fn,
        cost_false_positive=args.cost_fp,
        cost_true_positive_mitigation=args.cost_tp,
    )
    impact = evaluate_business_impact(
        result.y_test, scores, threshold=args.threshold, costs=costs
    )

    print("=" * 62)
    print(f"BUSINESS IMPACT @ threshold {args.threshold}  (holdout, one evaluation run)")
    print("=" * 62)
    print(f"cost assumptions      : FN=${args.cost_fn:,.0f}  FP=${args.cost_fp:,.0f}  TP=${args.cost_tp:,.0f}")
    print(f"confusion matrix      : TP={impact.true_positives} FP={impact.false_positives} "
          f"TN={impact.true_negatives} FN={impact.false_negatives}")
    print(f"do-nothing cost       : ${impact.do_nothing_cost:,.0f}")
    print(f"model cost            : ${impact.total_cost:,.0f}")
    print(f"net savings vs nothing: ${impact.net_savings_vs_do_nothing:,.0f}")

    sweep_summary = None
    if args.sweep:
        import numpy as np

        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2).tolist()
        sweep = compare_thresholds(result.y_test, scores, thresholds=thresholds, costs=costs)
        best_row = sweep.loc[sweep["net_savings_vs_do_nothing"].idxmax()]

        print("\n" + "=" * 62)
        print("THRESHOLD SWEEP (ROI-optimal point may differ from the shipped")
        print("precision/recall-optimal threshold - that is expected, not a bug)")
        print("=" * 62)
        print(
            f"shipped threshold {args.threshold}: net savings = "
            f"${impact.net_savings_vs_do_nothing:,.0f}"
        )
        print(
            f"best threshold {best_row.threshold}: net savings = "
            f"${best_row.net_savings_vs_do_nothing:,.0f}  "
            f"(TP={best_row.true_positives:.0f} FP={best_row.false_positives:.0f} "
            f"FN={best_row.false_negatives:.0f})"
        )
        sweep_summary = {
            "shipped_threshold": args.threshold,
            "shipped_net_savings": impact.net_savings_vs_do_nothing,
            "best_threshold": float(best_row.threshold),
            "best_net_savings": float(best_row.net_savings_vs_do_nothing),
            "best_true_positives": int(best_row.true_positives),
            "best_false_positives": int(best_row.false_positives),
            "best_false_negatives": int(best_row.false_negatives),
        }

    payload = {
        "cost_assumptions": asdict(costs),
        "evaluated_on": "holdout set (see README for split definition)",
        "n_holdout_observations": len(result.y_test),
        **{k: v for k, v in asdict(impact).items()},
    }
    if sweep_summary is not None:
        payload["threshold_sweep"] = sweep_summary

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {output_path}")
    print(
        "\nReminder: these dollar figures depend entirely on --cost-fn/--cost-fp/--cost-tp. "
        "Re-run with your own numbers before quoting this anywhere."
    )


if __name__ == "__main__":
    main()
