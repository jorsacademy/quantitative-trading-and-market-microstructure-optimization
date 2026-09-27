import numpy as np

from trading_optimization.limit_order_placement import (
    ACTIONS,
    QUEUE_STATES,
    default_problem,
    solve,
)


def test_limit_order_policy_defined_for_every_state():
    p = default_problem()
    r = solve(p)

    expected_states = (
        p.horizon
        * len(QUEUE_STATES)
        * 3
    )
    policy = r.policy[
        r.policy.index.get_level_values("time") < p.horizon
    ]

    assert len(policy) == expected_states
    assert set(policy.unique()).issubset(set(ACTIONS))


def test_limit_order_value_function_is_finite():
    r = solve(default_problem())

    assert np.isfinite(r.value_function.to_numpy()).all()
    assert np.isfinite(r.initial_value)


def test_deadline_policy_crosses_market():
    p = default_problem()
    r = solve(p)

    deadline = r.policy[
        r.policy.index.get_level_values("time") == p.horizon
    ]
    assert set(deadline.unique()) == {"market"}


def test_initial_value_not_worse_than_immediate_market():
    p = default_problem()
    r = solve(p)

    assert r.initial_value <= p.market_order_cost + 1e-9
