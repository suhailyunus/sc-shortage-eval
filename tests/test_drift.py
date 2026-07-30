"""
Tests for src/monitoring.py.

Each test corresponds to a failure mode a drift check is supposed to
catch: no reaction to sampling noise, a clear reaction to an actual
shift, and graceful handling of missing or thin data rather than a
crash in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring import (
    build_feature_reference,
    compute_drift_report,
    load_feature_reference,
    save_feature_reference,
)

FEATURES = ["sales_lag_1", "rolling_mean_7"]


def _reference_frame(seed: int = 0, n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "sales_lag_1": rng.normal(loc=5.0, scale=2.0, size=n),
            "rolling_mean_7": rng.normal(loc=5.0, scale=1.0, size=n),
        }
    )


def test_identical_distribution_is_not_flagged():
    reference_data = _reference_frame(seed=0)
    references = build_feature_reference(reference_data, FEATURES)

    current = _reference_frame(seed=1)  # same distribution, different draw
    report = compute_drift_report(references, current)

    assert all(not r.drifted for r in report)


def test_a_real_shift_is_flagged():
    reference_data = _reference_frame(seed=0)
    references = build_feature_reference(reference_data, FEATURES)

    shifted = _reference_frame(seed=2)
    shifted["sales_lag_1"] = shifted["sales_lag_1"] + 8.0  # large mean shift

    report = compute_drift_report(references, shifted)
    result = {r.feature: r for r in report}

    assert result["sales_lag_1"].drifted is True
    assert result["rolling_mean_7"].drifted is False


def test_missing_feature_is_reported_not_crashed():
    reference_data = _reference_frame(seed=0)
    references = build_feature_reference(reference_data, FEATURES)

    current = _reference_frame(seed=1).drop(columns=["sales_lag_1"])
    report = compute_drift_report(references, current)
    result = {r.feature: r for r in report}

    assert result["sales_lag_1"].drifted is False
    assert "absent" in result["sales_lag_1"].reason


def test_too_few_rows_is_not_evaluated():
    reference_data = _reference_frame(seed=0)
    references = build_feature_reference(reference_data, FEATURES)

    tiny_batch = _reference_frame(seed=3, n=5)
    report = compute_drift_report(references, tiny_batch, min_rows=30)
    result = {r.feature: r for r in report}

    assert result["sales_lag_1"].drifted is False
    assert "fewer than" in result["sales_lag_1"].reason


def test_reference_survives_a_save_load_roundtrip(tmp_path):
    reference_data = _reference_frame(seed=0)
    references = build_feature_reference(reference_data, FEATURES)

    path = tmp_path / "feature_reference_stats.json"
    save_feature_reference(references, path)
    reloaded = load_feature_reference(path)

    assert set(reloaded.keys()) == set(references.keys())
    for feature in FEATURES:
        assert reloaded[feature].sample == references[feature].sample


def test_sample_size_is_bounded():
    reference_data = _reference_frame(seed=0, n=9000)
    references = build_feature_reference(reference_data, FEATURES, sample_size=1000)

    assert len(references["sales_lag_1"].sample) == 1000
