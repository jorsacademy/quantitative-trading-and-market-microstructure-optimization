import numpy as np

from trading_optimization.credit_rfq import (
    default_problem,
    solve,
)


def test_credit_rfq_selects_one_action_per_request():
    problem = default_problem()
    result = solve(problem)

    assert len(result.selected_quotes) == len(problem.rfqs)
    assert result.selected_quotes["rfq"].nunique() == len(problem.rfqs)


def test_credit_rfq_inventory_and_hedge_limits():
    problem = default_problem()
    result = solve(problem)

    for factor in problem.inventory_limit.index:
        assert (
            result.inventory_path[factor].abs().max()
            <= problem.inventory_limit.loc[factor] + 1e-8
        )

    for hedge in problem.maximum_hedge_trade.index:
        assert (
            result.hedge_trades[hedge].abs().max()
            <= problem.maximum_hedge_trade.loc[hedge] + 1e-8
        )


def test_credit_rfq_hedging_improves_or_matches_risk_adjusted_value():
    problem = default_problem()
    hedged = solve(problem, allow_hedging=True)
    unhedged = solve(problem, allow_hedging=False)

    assert hedged.risk_adjusted_value >= unhedged.risk_adjusted_value - 1e-8
    assert np.isfinite(hedged.expected_quote_pnl)


def test_credit_rfq_uses_more_than_one_quote_tier():
    result = solve(default_problem())

    active_quotes = set(result.selected_quotes["quote"]) - {"reject"}
    assert len(active_quotes) >= 2
