import numpy as np

from fragmented_market.dark_pool import (
    DarkPoolConfig,
    MidpointDarkPool,
)


def test_dark_pool_executes_at_midpoint():
    pool = MidpointDarkPool(
        DarkPoolConfig(
            latency_steps=0,
            base_match_probability=1.0,
            hidden_liquidity_mean=100.0,
            maximum_order_quantity=50,
            seed=1,
        ),
        tick_size=0.01,
    )

    pool.submit(
        trader_id="buyer",
        side="buy",
        quantity=20,
    )
    executions = pool.step(
        midpoint_tick=10_000.5,
    )

    assert sum(x.filled_quantity for x in executions) == 20
    assert np.isclose(
        executions[0].price,
        100.005,
    )
    assert pool.pending_quantity("buyer") == 0


def test_dark_pool_latency_delays_eligibility():
    pool = MidpointDarkPool(
        DarkPoolConfig(
            latency_steps=2,
            base_match_probability=1.0,
            hidden_liquidity_mean=100.0,
            seed=2,
        )
    )

    pool.submit(
        trader_id="seller",
        side="sell",
        quantity=10,
    )

    first = pool.step(midpoint_tick=10_000)
    second = pool.step(midpoint_tick=10_000)
    third = pool.step(midpoint_tick=10_000)

    assert len(first) == 0
    assert len(second) == 0
    assert sum(x.filled_quantity for x in third) == 10


def test_dark_fill_probability_increases_with_horizon():
    pool = MidpointDarkPool()

    p1 = pool.expected_fill_probability(
        horizon_steps=1,
    )
    p4 = pool.expected_fill_probability(
        horizon_steps=4,
    )

    assert 0.0 < p1 < p4 < 1.0
