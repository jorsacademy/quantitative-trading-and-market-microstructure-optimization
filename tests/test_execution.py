import numpy as np

from trading_optimization.execution import (
    default_problem,
    objective_components,
    solve,
    twap_schedule,
    vwap_schedule,
)


def test_execution_completes_parent_order():
    p = default_problem()
    r = solve(p)

    assert np.isclose(r.schedule.sum(), p.total_quantity)
    assert np.isclose(r.remaining_inventory.iloc[-1], 0.0, atol=1e-5)


def test_execution_respects_participation_caps():
    p = default_problem()
    r = solve(p)

    caps = p.maximum_participation * p.market_volume
    assert (r.schedule <= caps + 1e-5).all()
    assert (r.schedule >= -1e-8).all()


def test_execution_not_worse_than_standard_benchmarks():
    p = default_problem()
    r = solve(p)

    assert r.objective_value <= r.twap_objective + 1e-4
    assert r.objective_value <= r.vwap_objective + 1e-4


def test_benchmark_schedules_complete_order():
    p = default_problem()

    assert np.isclose(twap_schedule(p).sum(), p.total_quantity)
    assert np.isclose(vwap_schedule(p).sum(), p.total_quantity)


def test_reported_objective_matches_components():
    p = default_problem()
    r = solve(p)
    impact, risk, total = objective_components(r.schedule, p)

    assert np.isclose(r.impact_cost, impact)
    assert np.isclose(r.risk_cost, risk)
    assert np.isclose(r.objective_value, total)
