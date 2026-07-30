from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostAssumptions:
    """
    Dollar costs assigned to each confusion-matrix outcome.

    These are not measured costs - this project has no verified financial
    data. They are inputs the caller must supply and be prepared to defend,
    analogous to the threshold choice in ``src/train.py``: any number here
    implicitly asserts a cost ratio between a false alarm and a missed
    event, and that ratio is a business decision, not a modeling result.
    """

    cost_false_negative: float
    cost_false_positive: float
    cost_true_positive_mitigation: float

    def __post_init__(self) -> None:
        for name in (
            "cost_false_negative",
            "cost_false_positive",
            "cost_true_positive_mitigation",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")


@dataclass(frozen=True)
class BusinessImpactResult:
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_cost: float
    cost_per_observation: float
    # Cost of doing nothing: every positive becomes a missed event (FN),
    # every negative costs nothing. This is the honest reference point,
    # not "cost of the previous model" or some other moving target.
    do_nothing_cost: float
    net_savings_vs_do_nothing: float


def evaluate_business_impact(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    costs: CostAssumptions,
) -> BusinessImpactResult:
    """
    Translate a confusion matrix at a given threshold into dollar terms.

    This is an evaluation-time calculation over labeled holdout data. It
    is not a live production metric: outcomes (did a shipment actually
    stress out?) are not known at prediction time, so this cannot be
    computed from a single day of unlabeled scoring traffic. Rerun it
    whenever the model is retrained or the cost assumptions change, the
    same way ``scripts/report_metrics.py`` is rerun.
    """

    y = np.asarray(y_true).astype(int)
    predictions = (np.asarray(probabilities) >= threshold).astype(int)

    if y.shape != predictions.shape:
        raise ValueError("y_true and probabilities must have the same length.")

    true_positives = int(((predictions == 1) & (y == 1)).sum())
    false_positives = int(((predictions == 1) & (y == 0)).sum())
    true_negatives = int(((predictions == 0) & (y == 0)).sum())
    false_negatives = int(((predictions == 0) & (y == 1)).sum())

    total_cost = (
        false_negatives * costs.cost_false_negative
        + false_positives * costs.cost_false_positive
        + true_positives * costs.cost_true_positive_mitigation
    )
    do_nothing_cost = int(y.sum()) * costs.cost_false_negative

    n = len(y)
    return BusinessImpactResult(
        threshold=float(threshold),
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        total_cost=float(total_cost),
        cost_per_observation=float(total_cost / n) if n else 0.0,
        do_nothing_cost=float(do_nothing_cost),
        net_savings_vs_do_nothing=float(do_nothing_cost - total_cost),
    )


def compare_thresholds(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: list[float],
    costs: CostAssumptions,
) -> pd.DataFrame:
    """Sweep thresholds under one fixed set of cost assumptions.

    Mirrors the threshold sweep in ``scripts/report_metrics.py`` but in
    dollar terms instead of precision/recall, so the two tables can sit
    side by side in the README without contradicting each other.
    """

    rows = [
        evaluate_business_impact(
            y_true, probabilities, threshold=t, costs=costs
        ).__dict__
        for t in thresholds
    ]
    return pd.DataFrame(rows)
