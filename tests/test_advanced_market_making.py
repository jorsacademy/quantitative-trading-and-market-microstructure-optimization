import numpy as np

from market_making_inventory_control.avellaneda_stoikov import (
    default_problem as default_as_problem,
    solve_avellaneda_stoikov,
)
from market_making_inventory_control.multi_asset import (
    default_multi_asset_problem,
    solve_multi_asset,
)
from market_making_inventory_control.rl_benchmark import (
    evaluate_policy_exact,
    train_q_learning,
)


def test_avellaneda_stoikov_inventory_skews_quotes():
    p = default_as_problem()
    r = solve_avellaneda_stoikov(p)

    t = 0
    positive = r.quote_surface.loc[(t, 3)]
    negative = r.quote_surface.loc[(t, -3)]

    # Long inventory: bid moves farther away, ask moves closer.
    assert positive["bid_offset"] > positive["ask_offset"]

    # Short inventory: ask moves farther away, bid moves closer.
    assert negative["ask_offset"] > negative["bid_offset"]


def test_avellaneda_stoikov_spread_is_finite_positive():
    r = solve_avellaneda_stoikov(default_as_problem())

    assert np.isfinite(r.half_spread.to_numpy()).all()
    assert (r.half_spread > 0).all()


def test_multi_asset_value_function_and_policy_are_finite():
    r = solve_multi_asset(default_multi_asset_problem())

    assert np.isfinite(r.value_function.to_numpy()).all()
    assert np.isfinite(r.initial_value)
    assert (r.policy != "").all()


def test_q_learning_policy_is_not_better_than_exact_dp():
    r = train_q_learning(
        episodes=4_000,
        seed=9,
    )

    assert np.isfinite(r.exact_policy_value)
    assert np.isfinite(r.optimal_dp_value)
    assert r.exact_policy_value <= r.optimal_dp_value + 1e-8
    assert r.value_gap >= -1e-8


def test_q_learning_policy_exact_evaluator_is_consistent():
    r = train_q_learning(
        episodes=2_000,
        seed=11,
    )

    value = evaluate_policy_exact(r.policy)
    assert np.isclose(
        value,
        r.exact_policy_value,
        atol=1e-10,
    )
