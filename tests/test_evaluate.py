from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend so plt.show() doesn't block tests

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    ThresholdResult,
    classification_metrics,
    evaluate_probabilities,
    find_best_f1_threshold,
    plot_confusion_matrix,
    plot_precision_recall,
    print_classification_metrics,
)


# --- classification_metrics / print_classification_metrics ----------------


def test_classification_metrics_returns_expected_keys() -> None:
    y_true = pd.Series([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])

    report = classification_metrics(y_true, y_pred)

    assert isinstance(report, dict)
    assert "accuracy" in report
    assert "1" in report  # per-class metrics keyed by label
    assert "macro avg" in report


def test_classification_metrics_handles_zero_division_gracefully() -> None:
    # No positive predictions at all: precision for class 1 is undefined
    # and should be reported as 0, not raise.
    y_true = pd.Series([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])

    report = classification_metrics(y_true, y_pred)
    assert report["1"]["precision"] == 0.0


def test_print_classification_metrics_prints_report(capsys) -> None:
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    print_classification_metrics(y_true, y_pred)

    captured = capsys.readouterr()
    assert "precision" in captured.out
    assert "recall" in captured.out


# --- find_best_f1_threshold -------------------------------------------------


def test_find_best_f1_threshold_returns_threshold_result() -> None:
    y_true = pd.Series([0, 0, 1, 1, 1, 0, 1])
    probabilities = np.array([0.1, 0.2, 0.9, 0.8, 0.6, 0.3, 0.55])

    result = find_best_f1_threshold(y_true, probabilities)

    assert isinstance(result, ThresholdResult)
    assert 0.0 <= result.threshold <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1 <= 1.0


def test_find_best_f1_threshold_perfect_separation_gives_f1_of_one() -> None:
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.01, 0.05, 0.1, 0.9, 0.95, 0.99])

    result = find_best_f1_threshold(y_true, probabilities)

    assert result.f1 == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)


# --- evaluate_probabilities -------------------------------------------------


def test_evaluate_probabilities_structure_and_threshold() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.2, 0.4, 0.6, 0.8])

    result = evaluate_probabilities(y_true, probabilities, threshold=0.5)

    assert result["threshold"] == 0.5
    assert 0.0 <= result["average_precision"] <= 1.0
    assert "classification_report" in result
    assert isinstance(result["classification_report"], dict)


def test_evaluate_probabilities_default_threshold_is_half() -> None:
    y_true = pd.Series([0, 1])
    probabilities = np.array([0.3, 0.7])

    result = evaluate_probabilities(y_true, probabilities)

    assert result["threshold"] == 0.50


# --- plot_confusion_matrix / plot_precision_recall -------------------------
# These use a real fitted model from the shared pipeline fixture rather
# than a mock, so the plotting calls exercise the actual sklearn/XGBoost
# estimator interface (predict, classes_) that ConfusionMatrixDisplay
# relies on.


def test_plot_confusion_matrix_runs_and_saves(trained_pipeline_result, tmp_path) -> None:
    result = trained_pipeline_result
    output_path = tmp_path / "confusion.png"

    plot_confusion_matrix(
        result.model,
        result.X_test,
        result.y_test,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_confusion_matrix_runs_without_saving_when_no_path(
    trained_pipeline_result,
) -> None:
    result = trained_pipeline_result
    # Should not raise even though no output_path is given.
    plot_confusion_matrix(result.model, result.X_test, result.y_test)


def test_plot_precision_recall_runs_and_saves(trained_pipeline_result, tmp_path) -> None:
    result = trained_pipeline_result
    probabilities = result.model.predict_proba(result.X_test)[:, 1]
    output_path = tmp_path / "pr_curve.png"

    plot_precision_recall(
        result.y_test,
        probabilities,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_precision_recall_creates_parent_directories(
    trained_pipeline_result, tmp_path
) -> None:
    result = trained_pipeline_result
    probabilities = result.model.predict_proba(result.X_test)[:, 1]
    nested_path = tmp_path / "nested" / "dir" / "pr_curve.png"

    plot_precision_recall(result.y_test, probabilities, output_path=nested_path)

    assert nested_path.exists()
