import numpy as np

from trading_optimization.rfq_pricing import (
    default_problem,
    solve,
)


def test_rfq_pricing_selects_one_action_per_request():
    p = default_problem()
    r = solve(p)

    assert len(r.selected_quotes) == len(p.rfqs)
    assert r.selected_quotes["rfq"].nunique() == len(p.rfqs)


def test_rfq_portfolio_limits():
    p = default_problem()
    r = solve(p)

    assert (
        r.expected_gross_notional
        <= p.expected_gross_notional_limit + 1e-7
    )
    assert (
        abs(r.expected_net_inventory)
        <= p.expected_net_inventory_limit + 1e-7
    )


def test_rfq_acceptance_probabilities_are_valid():
    r = solve(default_problem())

    probs = r.selected_quotes["acceptance_probability"]
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert np.isfinite(r.expected_pnl)
