import numpy as np

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


def test_impact_calibration_is_nonnegative_and_accurate():
    data = generate_synthetic_impact_data(
        observations=600,
        seed=5,
    )
    model = fit_impact_model(data)

    assert model.intercept >= 0.0
    assert (model.coefficients >= 0.0).all()
    assert model.train_rmse_bps < 0.20


def test_stochastic_scenarios_are_reproducible():
    p = default_stochastic_problem()
    a_alpha, a_vol = generate_scenarios(p)
    b_alpha, b_vol = generate_scenarios(p)

    assert a_alpha.equals(b_alpha)
    assert a_vol.equals(b_vol)


def test_stochastic_execution_completes_parent_order():
    p = default_stochastic_problem()
    r = solve_stochastic(p)

    assert np.isclose(
        r.schedule.sum(),
        p.base.total_quantity,
        atol=1e-4,
    )
    assert np.isclose(
        r.remaining_inventory.iloc[-1],
        0.0,
        atol=1e-4,
    )


def test_stochastic_execution_respects_participation_caps():
    p = default_stochastic_problem()
    r = solve_stochastic(p)

    caps = (
        p.base.maximum_participation
        * p.base.market_volume
    )
    assert (r.schedule <= caps + 1e-4).all()
    assert (r.schedule >= -1e-8).all()


def test_stochastic_execution_has_valid_tail_metrics():
    p = default_stochastic_problem()
    r = solve_stochastic(p)

    assert r.expected_cost <= r.cvar_cost + 1e-8
    assert r.var_cost <= r.cvar_cost + 1e-8
    assert np.isfinite(r.objective_value)


def test_scenario_costs_match_reported_result():
    p = default_stochastic_problem()
    model = fit_impact_model()
    alpha, vol = generate_scenarios(p)
    r = solve_stochastic(p, impact_model=model)

    costs = scenario_costs(
        r.schedule,
        p,
        model,
        alpha,
        vol,
    )
    assert np.isclose(
        costs.mean(),
        r.expected_cost,
        atol=1e-6,
    )
