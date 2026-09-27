import numpy as np

from trading_optimization.market_making import (
    default_problem,
    parse_action,
    simulate_policy,
    solve,
)


def test_market_making_value_function_is_finite():
    p = default_problem()
    r = solve(p)

    assert np.isfinite(r.value_function.to_numpy()).all()
    assert np.isfinite(r.initial_value)


def test_boundary_inventory_disables_risk_increasing_side():
    p = default_problem()
    r = solve(p)

    for t in range(p.horizon):
        bid, _ask = parse_action(
            r.policy.loc[t, p.maximum_inventory]
        )
        _bid, ask = parse_action(
            r.policy.loc[t, -p.maximum_inventory]
        )

        assert bid == 0
        assert ask == 0


def test_policy_actions_keep_inventory_feasible():
    p = default_problem()
    r = solve(p)

    for inventory in r.policy.columns:
        for t in r.policy.index:
            bid, ask = parse_action(
                r.policy.loc[t, inventory]
            )

            if inventory == p.maximum_inventory:
                assert bid == 0
            if inventory == -p.maximum_inventory:
                assert ask == 0


def test_simulation_respects_inventory_bounds():
    p = default_problem()
    r = solve(p)
    simulation = simulate_policy(
        r,
        p,
        replications=200,
        seed=19,
    )

    assert (
        simulation["terminal_inventory"].abs()
        <= p.maximum_inventory
    ).all()
    assert (simulation["fills"] >= 0).all()
