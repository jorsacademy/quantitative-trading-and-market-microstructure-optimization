from dataclasses import replace

import numpy as np
import pytest

from optimal_trade_execution.impact_model import (
    fit_impact_model,
    generate_synthetic_impact_data,
)
from optimal_trade_execution.stochastic_execution import (
    default_stochastic_problem,
    generate_scenarios,
    scenario_costs,
    solve_stochastic,
)


@pytest.fixture(scope="module")
def problem():
    return replace(
        default_stochastic_problem(),
        scenarios=12,
    )


@pytest.fixture(scope="module")
def impact_model():
    return fit_impact_model(
        generate_synthetic_impact_data(
            observations=600,
            seed=5,
        )
    )


@pytest.fixture(scope="module")
def result(problem, impact_model):
    return solve_stochastic(
        problem,
        impact_model=impact_model,
    )


def test_impact_calibration_is_nonnegative_and_accurate(impact_model):
    assert impact_model.intercept >= 0.0
    assert (impact_model.coefficients >= 0.0).all()
    assert impact_model.train_rmse_bps < 0.20


def test_stochastic_scenarios_are_reproducible(problem):
    a_alpha, a_vol = generate_scenarios(problem)
    b_alpha, b_vol = generate_scenarios(problem)

    assert a_alpha.equals(b_alpha)
    assert a_vol.equals(b_vol)


def test_stochastic_execution_completes_parent_order(problem, result):
    assert np.isclose(
        result.schedule.sum(),
        problem.base.total_quantity,
        atol=1e-4,
    )
    assert np.isclose(
        result.remaining_inventory.iloc[-1],
        0.0,
        atol=1e-4,
    )


def test_stochastic_execution_respects_participation_caps(problem, result):
    caps = (
        problem.base.maximum_participation
        * problem.base.market_volume
    )
    assert (result.schedule <= caps + 1e-4).all()
    assert (result.schedule >= -1e-8).all()


def test_stochastic_execution_has_valid_tail_metrics(result):
    assert result.expected_cost <= result.cvar_cost + 1e-8
    assert result.var_cost <= result.cvar_cost + 1e-8
    assert np.isfinite(result.objective_value)


def test_scenario_costs_match_reported_result(
    problem,
    impact_model,
    result,
):
    alpha, vol = generate_scenarios(problem)
    costs = scenario_costs(
        result.schedule,
        problem,
        impact_model,
        alpha,
        vol,
    )
    assert np.isclose(
        costs.mean(),
        result.expected_cost,
        atol=1e-6,
    )
