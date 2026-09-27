import numpy as np

from fragmented_market.environment import (
    MultiVenueMarket,
    VenueConfig,
)
from fragmented_market.router import (
    route_once,
    route_parent_order,
    route_parent_order_static,
)


def test_consolidated_nbbo_matches_venue_books():
    market = MultiVenueMarket(seed=101)
    snapshot = market.snapshot()

    assert snapshot.nbbo_bid_tick == int(
        snapshot.venues["best_bid_tick"].max()
    )
    assert snapshot.nbbo_ask_tick == int(
        snapshot.venues["best_ask_tick"].min()
    )
    assert snapshot.nbbo_bid_tick < snapshot.nbbo_ask_tick


def test_venue_books_are_independent():
    market = MultiVenueMarket(seed=102)
    names = list(market.venues)
    assert len(names) >= 2

    first = market.venues[names[0]]
    second = market.venues[names[1]]

    before_second = second.observe().to_series().copy()

    first.submit_limit = first.book.submit_limit  # type: ignore[attr-defined]
    first.book.submit_limit(
        order_id="manual-bid",
        trader_id="manual",
        side="buy",
        price_tick=first.observe().best_bid_tick,
        quantity=7,
        timestamp=first.time,
    )

    after_second = second.observe().to_series()
    assert before_second.equals(after_second)


def test_latency_delays_child_order_arrival():
    config = (
        VenueConfig(
            name="slow",
            taker_fee_bps=0.10,
            maker_rebate_bps=-0.05,
            latency_steps=2,
            initial_mid_offset_ticks=0,
            initial_spread_ticks=2,
            initial_level_quantity=100,
            background_events_per_step=0,
            seed_offset=0,
        ),
    )
    market = MultiVenueMarket(
        venue_configs=config,
        seed=5,
    )

    market.schedule_market_order(
        venue="slow",
        trader_id="buyer",
        side="buy",
        quantity=10,
    )

    _, first = market.step()
    _, second = market.step()
    _, third = market.step()

    assert len(first) == 0
    assert len(second) == 0
    assert sum(x.filled_quantity for x in third) == 10
    assert market.consolidated_inventory("buyer") == 10


def test_route_once_respects_depth_and_active_venue_limit():
    market = MultiVenueMarket(seed=103)
    snapshot = market.snapshot()

    allocation = route_once(
        snapshot,
        side="buy",
        quantity=250,
        maximum_active_venues=2,
    )

    assert (allocation >= 0).all()
    assert (allocation > 0).sum() <= 2
    assert (
        allocation
        <= snapshot.venues["ask_depth"]
    ).all()


def test_dynamic_and_static_routing_preserve_quantity_accounting():
    static_market = MultiVenueMarket(seed=104)
    dynamic_market = MultiVenueMarket(seed=104)

    static = route_parent_order_static(
        static_market,
        side="buy",
        quantity=120,
    )
    dynamic = route_parent_order(
        dynamic_market,
        side="buy",
        quantity=120,
    )

    for result in (static, dynamic):
        assert (
            result.filled_quantity
            + result.remaining_quantity
            == result.requested_quantity
        )
        assert result.filled_quantity >= 0
        assert result.remaining_quantity >= 0
        if result.filled_quantity > 0:
            assert result.average_execution_price is not None
            assert np.isfinite(result.average_execution_price)
