"""Live fragmented-market benchmark for the Smart Order Routing project."""

from __future__ import annotations

import pandas as pd

from fragmented_market.environment import MultiVenueMarket
from fragmented_market.router import (
    LiveRoutingResult,
    route_parent_order,
    route_parent_order_static,
)


def implementation_shortfall(
    result: LiveRoutingResult,
    *,
    side: str,
    reference_price: float,
) -> float:
    """Currency implementation shortfall including explicit venue fees."""
    if (
        result.average_execution_price is None
        or result.filled_quantity <= 0
    ):
        return 0.0

    if side == "buy":
        price_cost = (
            result.average_execution_price - reference_price
        ) * result.filled_quantity
    elif side == "sell":
        price_cost = (
            reference_price - result.average_execution_price
        ) * result.filled_quantity
    else:
        raise ValueError("side must be buy or sell")

    return float(price_cost + result.explicit_fees)


def compare_static_dynamic_live_routing(
    *,
    side: str = "buy",
    quantity: int = 180,
    seed: int = 2026,
    maximum_active_venues: int = 3,
) -> tuple[pd.DataFrame, dict[str, LiveRoutingResult]]:
    """Compare one-shot routing with dynamic re-routing on matched market seeds."""
    static_market = MultiVenueMarket(seed=seed)
    dynamic_market = MultiVenueMarket(seed=seed)

    initial = static_market.snapshot()
    reference_tick = (
        initial.nbbo_ask_tick
        if side == "buy"
        else initial.nbbo_bid_tick
    )
    reference_price = reference_tick * static_market.tick_size

    static = route_parent_order_static(
        static_market,
        side=side,
        quantity=quantity,
        trader_id="static_router",
        maximum_active_venues=maximum_active_venues,
    )
    dynamic = route_parent_order(
        dynamic_market,
        side=side,
        quantity=quantity,
        trader_id="dynamic_router",
        maximum_active_venues=maximum_active_venues,
    )

    rows = []
    for name, result in (
        ("static_one_shot", static),
        ("dynamic_rerouting", dynamic),
    ):
        rows.append(
            {
                "policy": name,
                "requested_quantity": result.requested_quantity,
                "filled_quantity": result.filled_quantity,
                "fill_ratio": (
                    result.filled_quantity
                    / result.requested_quantity
                    if result.requested_quantity > 0
                    else 0.0
                ),
                "remaining_quantity": result.remaining_quantity,
                "average_execution_price": result.average_execution_price,
                "explicit_fees": result.explicit_fees,
                "implementation_shortfall": implementation_shortfall(
                    result,
                    side=side,
                    reference_price=reference_price,
                ),
                "routing_decisions": len(result.route_decisions),
                "market_steps": result.steps,
            }
        )

    return (
        pd.DataFrame(rows).set_index("policy"),
        {
            "static_one_shot": static,
            "dynamic_rerouting": dynamic,
        },
    )
