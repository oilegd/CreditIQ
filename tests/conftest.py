"""Shared SYNTHETIC fixtures. No test downloads or reads the Lending Club dataset."""

from __future__ import annotations

import pandas as pd
import pytest

from creditiq.config import Config, load_config
from creditiq.data.pipeline import PreparedData, prepare_dataset
from creditiq.data.synthetic import make_synthetic_loans


@pytest.fixture(scope="session")
def cfg() -> Config:
    # Small LightGBM budget and few bootstrap reps keep the suite fast.
    return load_config().with_overrides(
        {
            "model": {
                "lightgbm": {
                    "n_estimators": 150,
                    "early_stopping_rounds": 20,
                    "param_grid": [{"num_leaves": 15, "min_child_samples": 50, "reg_lambda": 1.0}],
                },
                "logistic_regression": {"C_grid": [0.1], "class_weight_grid": [None]},
            },
            "evaluation": {"bootstrap_reps": 20, "random_split_benchmark": False},
        }
    )


@pytest.fixture(scope="session")
def raw_synthetic() -> pd.DataFrame:
    return make_synthetic_loans(n=4000, seed=7)


@pytest.fixture(scope="session")
def prepared(raw_synthetic: pd.DataFrame, cfg: Config) -> PreparedData:
    return prepare_dataset(raw_synthetic, cfg)


@pytest.fixture()
def raw_copy(raw_synthetic: pd.DataFrame) -> pd.DataFrame:
    return raw_synthetic.copy()


@pytest.fixture(scope="session")
def trained(prepared: PreparedData, cfg: Config, tmp_path_factory: pytest.TempPathFactory):
    """Train once on synthetic data and save artifacts to a temp models dir."""
    from creditiq.models.train import save_training_result, train_all

    models_dir = tmp_path_factory.mktemp("models")
    result = train_all(prepared.data, cfg, "applicant")
    save_training_result(result, models_dir, cfg)
    return result, models_dir
