import numpy as np
import pytest

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


@pytest.fixture(scope="module")
def as_result():
    return solve_avellaneda_stoikov(
        default_as_problem()
    )


@pytest.fixture(scope="module")
def multi_result():
    return solve_multi_asset(
        default_multi_asset_problem()
    )


@pytest.fixture(scope="module")
def rl_result():
    return train_q_learning(
        episodes=4_000,
        seed=9,
    )


def test_avellaneda_stoikov_inventory_skews_quotes(as_result):
    t = 0
    positive = as_result.quote_surface.loc[(t, 3)]
    negative = as_result.quote_surface.loc[(t, -3)]

    assert positive["bid_offset"] > positive["ask_offset"]
    assert negative["ask_offset"] > negative["bid_offset"]


def test_avellaneda_stoikov_spread_is_finite_positive(as_result):
    assert np.isfinite(as_result.half_spread.to_numpy()).all()
    assert (as_result.half_spread > 0).all()


def test_multi_asset_value_function_and_policy_are_finite(multi_result):
    assert np.isfinite(multi_result.value_function.to_numpy()).all()
    assert np.isfinite(multi_result.initial_value)
    assert (multi_result.policy != "").all()


def test_q_learning_policy_is_not_better_than_exact_dp(rl_result):
    assert np.isfinite(rl_result.exact_policy_value)
    assert np.isfinite(rl_result.optimal_dp_value)
    assert rl_result.exact_policy_value <= rl_result.optimal_dp_value + 1e-8
    assert rl_result.value_gap >= -1e-8


def test_q_learning_policy_exact_evaluator_is_consistent(rl_result):
    value = evaluate_policy_exact(rl_result.policy)
    assert np.isclose(
        value,
        rl_result.exact_policy_value,
        atol=1e-10,
    )
