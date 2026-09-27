"""Benchmark lit-taker-only routing against hybrid maker/taker/dark routing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fragmented_market.dark_pool import DarkPoolConfig, MidpointDarkPool
from fragmented_market.environment import MultiVenueMarket
from fragmented_market.hybrid_router import (
    HybridRoutingResult,
    route_hybrid_parent_order,
)
from fragmented_market.router import LiveRoutingResult, route_parent_order


def _shortfall(
    *,
    side: str,
    average_price: float | None,
    filled_quantity: int,
    explicit_fees: float,
    reference_price: float,
) -> float:
    if average_price is None or filled_quantity <= 0:
        return float(explicit_fees)

    if side == "buy":
        price_cost = (
            average_price - reference_price
        ) * filled_quantity
    elif side == "sell":
        price_cost = (
            reference_price - average_price
        ) * filled_quantity
    else:
        raise ValueError("side must be buy or sell")

    return float(price_cost + explicit_fees)


def compare_lit_vs_hybrid(
    *,
    side: str = "buy",
    quantity: int = 160,
    seed: int = 2026,
) -> tuple[
    pd.DataFrame,
    dict[str, LiveRoutingResult | HybridRoutingResult],
]:
    """Compare live lit-taker rerouting with hybrid maker/taker/dark routing."""
    lit_market = MultiVenueMarket(seed=seed)
    hybrid_market = MultiVenueMarket(seed=seed)
    dark_pool = MidpointDarkPool(
        DarkPoolConfig(seed=seed + 10_000),
        tick_size=hybrid_market.tick_size,
    )

    initial = lit_market.snapshot()
    reference_tick = (
        initial.nbbo_ask_tick
        if side == "buy"
        else initial.nbbo_bid_tick
    )
    reference_price = reference_tick * lit_market.tick_size

    lit = route_parent_order(
        lit_market,
        side=side,
        quantity=quantity,
        trader_id="lit_router",
        maximum_active_venues=3,
        max_steps=20,
    )
    hybrid = route_hybrid_parent_order(
        hybrid_market,
        dark_pool,
        side=side,
        quantity=quantity,
        trader_id="hybrid_router",
        maker_patience_steps=3,
        max_steps=20,
    )

    hybrid_lit = hybrid.lit_executions
    maker_fill = (
        int(
            hybrid_lit.loc[
                hybrid_lit["liquidity_role"] == "maker",
                "filled_quantity",
            ].sum()
        )
        if not hybrid_lit.empty
        else 0
    )
    taker_fill = (
        int(
            hybrid_lit.loc[
                hybrid_lit["liquidity_role"] == "taker",
                "filled_quantity",
            ].sum()
        )
        if not hybrid_lit.empty
        else 0
    )
    dark_fill = (
        int(hybrid.dark_executions["filled_quantity"].sum())
        if not hybrid.dark_executions.empty
        else 0
    )

    rows = [
        {
            "policy": "lit_taker_only",
            "requested_quantity": lit.requested_quantity,
            "filled_quantity": lit.filled_quantity,
            "fill_ratio": (
                lit.filled_quantity / lit.requested_quantity
            ),
            "remaining_quantity": lit.remaining_quantity,
            "average_execution_price": lit.average_execution_price,
            "explicit_fees": lit.explicit_fees,
            "implementation_shortfall": _shortfall(
                side=side,
                average_price=lit.average_execution_price,
                filled_quantity=lit.filled_quantity,
                explicit_fees=lit.explicit_fees,
                reference_price=reference_price,
            ),
            "maker_fill": 0,
            "taker_fill": lit.filled_quantity,
            "dark_fill": 0,
            "market_steps": lit.steps,
        },
        {
            "policy": "hybrid_maker_taker_dark",
            "requested_quantity": hybrid.requested_quantity,
            "filled_quantity": hybrid.filled_quantity,
            "fill_ratio": (
                hybrid.filled_quantity / hybrid.requested_quantity
            ),
            "remaining_quantity": hybrid.remaining_quantity,
            "average_execution_price": hybrid.average_execution_price,
            "explicit_fees": hybrid.explicit_fees,
            "implementation_shortfall": _shortfall(
                side=side,
                average_price=hybrid.average_execution_price,
                filled_quantity=hybrid.filled_quantity,
                explicit_fees=hybrid.explicit_fees,
                reference_price=reference_price,
            ),
            "maker_fill": maker_fill,
            "taker_fill": taker_fill,
            "dark_fill": dark_fill,
            "market_steps": hybrid.steps,
        },
    ]

    table = pd.DataFrame(rows).set_index("policy")
    table["shortfall_difference_vs_lit"] = (
        table["implementation_shortfall"]
        - table.loc[
            "lit_taker_only",
            "implementation_shortfall",
        ]
    )
    table["fee_difference_vs_lit"] = (
        table["explicit_fees"]
        - table.loc["lit_taker_only", "explicit_fees"]
    )

    return table, {
        "lit_taker_only": lit,
        "hybrid_maker_taker_dark": hybrid,
    }
