"""Shared fixtures: a smaller deterministic environment keeps the suite fast."""
from __future__ import annotations

import pytest

from src.config import DataSettings
from src.data.synthetic_data import generate_environment

TEST_SETTINGS = DataSettings(seed=7, n_dealers=150, n_weeks=72, n_institutional=120)


@pytest.fixture(scope="session")
def env():
    return generate_environment(TEST_SETTINGS)


@pytest.fixture(scope="session")
def demand(env):
    from src.models.demand_forecasting import train_demand_model
    return train_demand_model(env)


@pytest.fixture(scope="session")
def credit(env):
    from src.models.credit_risk import train_credit_model
    return train_credit_model(env)


@pytest.fixture(scope="session")
def eta(env):
    from src.models.eta_prediction import train_eta_model
    return train_eta_model(env)
