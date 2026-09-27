import numpy as np

from trading_optimization.smart_order_routing import (
    default_problem,
    solve,
)


def test_routing_allocates_entire_parent_order():
    p = default_problem()
    r = solve(p)

    assert r.blocks.sum() == p.total_blocks


def test_routing_respects_venue_capacities():
    p = default_problem()
    r = solve(p)

    assert (
        r.blocks
        <= p.venues["capacity_blocks"]
    ).all()
    assert (r.blocks >= 0).all()


def test_routing_meets_expected_fill_target():
    p = default_problem()
    r = solve(p)

    target = (
        p.minimum_expected_fill_ratio
        * p.total_blocks
    )
    assert r.expected_filled_blocks >= target - 1e-7


def test_routing_respects_active_venue_limit():
    p = default_problem()
    r = solve(p)

    assert r.active_venues <= p.maximum_active_venues
    assert np.isfinite(r.objective_value)
