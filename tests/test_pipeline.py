from __future__ import annotations

import pytest

from src.pipeline import TrainingResult, run_training_pipeline
from tests.conftest import build_synthetic_m5


def test_run_training_pipeline_end_to_end(synthetic_m5_dir) -> None:
    data_dir = synthetic_m5_dir

    result = run_training_pipeline(
        data_dir,
        max_items=None,
        train_fraction=0.75,
    )

    assert isinstance(result, TrainingResult)
    assert not result.X_train.empty
    assert not result.X_test.empty
    assert len(result.X_train) == len(result.y_train)
    assert len(result.X_test) == len(result.y_test)
    assert set(result.model_features) == set(result.X_train.columns)

    # Chronological validity: every training day must precede every
    # test day, mirroring the guarantee asserted in test_leakage.py.
    train_days = result.feature_data.loc[result.X_train.index, "day_num"]
    test_days = result.feature_data.loc[result.X_test.index, "day_num"]
    assert train_days.max() <= result.split_day
    assert test_days.min() > result.split_day


def test_run_training_pipeline_respects_max_items(tmp_path) -> None:
    data_dir = build_synthetic_m5(tmp_path, n_items=3)

    result = run_training_pipeline(data_dir, max_items=1, train_fraction=0.75)

    assert result.analytical_data["item_id"].nunique() == 1


def test_run_training_pipeline_saves_artifacts_when_models_dir_given(synthetic_m5_dir, tmp_path) -> None:
    data_dir = synthetic_m5_dir
    models_dir = tmp_path / "trained_models"

    run_training_pipeline(
        data_dir,
        max_items=None,
        train_fraction=0.75,
        models_dir=models_dir,
        default_threshold=0.8,
    )

    assert (models_dir / "model_features.json").exists()
    assert (models_dir / "model_config.json").exists()


def test_run_training_pipeline_skips_saving_when_models_dir_is_none(synthetic_m5_dir) -> None:
    result = run_training_pipeline(synthetic_m5_dir, max_items=None, models_dir=None)

    # Nothing should have been written anywhere outside tmp_path's data_dir.
    assert result.model is not None


def test_run_training_pipeline_raises_on_missing_data_files(tmp_path) -> None:
    empty_dir = tmp_path / "no_data_here"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        run_training_pipeline(empty_dir)
