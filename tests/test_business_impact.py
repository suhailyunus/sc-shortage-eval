"""
Tests for src/business_impact.py.

These check arithmetic correctness of the cost translation, not whether
any particular dollar figure is "right" - the cost assumptions are
inputs, not claims this module makes.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.business_impact import CostAssumptions, evaluate_business_impact


def test_negative_costs_are_rejected():
    with pytest.raises(ValueError):
        CostAssumptions(
            cost_false_negative=-1.0,
            cost_false_positive=10.0,
            cost_true_positive_mitigation=5.0,
        )


def test_mismatched_lengths_are_rejected():
    costs = CostAssumptions(100.0, 10.0, 20.0)
    with pytest.raises(ValueError):
        evaluate_business_impact(
            np.array([0, 1, 1]), np.array([0.1, 0.9]), threshold=0.5, costs=costs
        )


def test_confusion_matrix_counts_are_correct():
    y_true = np.array([0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.9, 0.4, 0.2])
    # Predictions at 0.5: [0, 1, 1, 0, 0]
    # TN: idx0 (y=0,pred=0), FP: idx1 (y=0,pred=1)
    # TP: idx2 (y=1,pred=1), FN: idx3, idx4 (y=1,pred=0)
    costs = CostAssumptions(
        cost_false_negative=1.0,
        cost_false_positive=1.0,
        cost_true_positive_mitigation=1.0,
    )
    result = evaluate_business_impact(y_true, probabilities, threshold=0.5, costs=costs)

    assert result.true_negatives == 1
    assert result.false_positives == 1
    assert result.true_positives == 1
    assert result.false_negatives == 2


def test_total_cost_matches_hand_calculation():
    y_true = np.array([0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.9, 0.4, 0.2])
    costs = CostAssumptions(
        cost_false_negative=2500.0,
        cost_false_positive=150.0,
        cost_true_positive_mitigation=300.0,
    )
    result = evaluate_business_impact(y_true, probabilities, threshold=0.5, costs=costs)

    # 2 FN * 2500 + 1 FP * 150 + 1 TP * 300 = 5450
    assert result.total_cost == pytest.approx(5450.0)


def test_do_nothing_cost_is_all_positives_as_false_negatives():
    y_true = np.array([0, 1, 1, 1])
    probabilities = np.array([0.9, 0.9, 0.9, 0.9])  # everything flagged
    costs = CostAssumptions(
        cost_false_negative=2500.0,
        cost_false_positive=150.0,
        cost_true_positive_mitigation=300.0,
    )
    result = evaluate_business_impact(y_true, probabilities, threshold=0.5, costs=costs)

    # Doing nothing means all 3 real events become missed events.
    assert result.do_nothing_cost == pytest.approx(3 * 2500.0)


def test_net_savings_can_be_negative_for_a_bad_threshold():
    """A threshold so low it flags everything can cost more than doing nothing
    if false-positive volume is large enough - this must not be hidden."""

    y_true = np.array([0, 0, 0, 0, 1])
    probabilities = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    costs = CostAssumptions(
        cost_false_negative=100.0,
        cost_false_positive=1000.0,
        cost_true_positive_mitigation=0.0,
    )
    result = evaluate_business_impact(y_true, probabilities, threshold=0.5, costs=costs)

    assert result.net_savings_vs_do_nothing < 0
