import numpy as np

from limit_order_book_simulator.environment import AgentAction
from fragmented_market.dark_pool import MidpointDarkPool
from fragmented_market.environment import (
    MultiVenueMarket,
    VenueConfig,
)
from fragmented_market.hybrid_router import (
    candidate_table,
    hybrid_route_once,
    route_hybrid_parent_order,
)


def test_hybrid_candidates_include_all_execution_modes():
    market = MultiVenueMarket(seed=120)
    dark = MidpointDarkPool()
    table = candidate_table(
        market.snapshot(),
        dark,
        side="buy",
    )

    modes = set(table["mode"])
    assert {
        "lit_taker",
        "lit_maker",
        "dark_midpoint",
    }.issubset(modes)
    assert ((table["fill_probability"] > 0) & (table["fill_probability"] <= 1)).all()


def test_hybrid_wave_respects_candidate_capacities():
    market = MultiVenueMarket(seed=121)
    dark = MidpointDarkPool()
    allocation = hybrid_route_once(
        market.snapshot(),
        dark,
        side="sell",
        quantity=140,
        maximum_active_actions=4,
    )

    assert (allocation["quantity"] >= 0).all()
    assert (
        allocation["quantity"]
        <= allocation["capacity"]
    ).all()
    assert (allocation["quantity"] > 0).sum() <= 4


def test_maker_fill_uses_maker_rebate_accounting():
    market = MultiVenueMarket(
        venue_configs=(
            VenueConfig(
                name="venue_a",
                taker_fee_bps=0.18,
                maker_rebate_bps=-0.04,
                latency_steps=0,
                initial_mid_offset_ticks=0,
                initial_spread_ticks=2,
                initial_level_quantity=90,
                background_events_per_step=0,
                seed_offset=0,
            ),
        ),
        seed=122,
    )
    venue = "venue_a"
    env = market.venues[venue]
    obs = env.observe()

    order_id = market.schedule_limit_order(
        venue=venue,
        trader_id="maker_agent",
        side="sell",
        quantity=5,
        price_tick=obs.best_ask_tick,
    )
    market.step()

    trades = env.apply_action(
        AgentAction(
            action_type="market",
            trader_id="external_taker",
            side="buy",
            quantity=5,
        )
    )
    executions = market._record_maker_fills(
        venue,
        tuple(trades),
    )

    assert len(executions) == 1
    assert executions[0].order_id == order_id
    assert executions[0].liquidity_role == "maker"
    assert executions[0].filled_quantity == 5
    assert executions[0].explicit_fee <= 0.0


def test_hybrid_parent_order_preserves_quantity_accounting():
    market = MultiVenueMarket(seed=123)
    dark = MidpointDarkPool()
    result = route_hybrid_parent_order(
        market,
        dark,
        side="buy",
        quantity=90,
        trader_id="hybrid_test",
        max_steps=10,
    )

    assert (
        result.filled_quantity
        + result.remaining_quantity
        == result.requested_quantity
    )
    assert result.filled_quantity >= 0
    assert result.remaining_quantity >= 0
    assert np.isfinite(result.explicit_fees)


def test_hybrid_decisions_are_labeled_by_mode():
    market = MultiVenueMarket(seed=124)
    dark = MidpointDarkPool()
    result = route_hybrid_parent_order(
        market,
        dark,
        side="sell",
        quantity=80,
        trader_id="hybrid_modes",
        max_steps=8,
    )

    if not result.decisions.empty:
        assert set(result.decisions["mode"]).issubset(
            {
                "lit_taker",
                "lit_maker",
                "dark_midpoint",
                "deadline_lit_taker",
            }
        )
