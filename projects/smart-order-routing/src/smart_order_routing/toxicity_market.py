"""Benchmark toxicity-blind vs toxicity-aware hybrid routing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fragmented_market.dark_pool import (
    DarkPoolConfig,
    MidpointDarkPool,
)
from fragmented_market.environment import MultiVenueMarket
from fragmented_market.hybrid_router import route_hybrid_parent_order


def _weighted_selected_toxicity(decisions: pd.DataFrame) -> float:
    if decisions.empty or "toxicity_probability" not in decisions:
        return 0.0
    weights = decisions["quantity"].to_numpy(float)
    if weights.sum() <= 0:
        return 0.0
    values = decisions["toxicity_probability"].to_numpy(float)
    return float(np.dot(weights, values) / weights.sum())


def compare_toxicity_aware_routing(
    *,
    side: str = "buy",
    quantity: int = 140,
    seed: int = 2026,
    toxicity_penalty_ticks: float = 1.25,
) -> tuple[pd.DataFrame, dict]:
    """Compare matched calibrated markets with toxicity penalty off vs on."""
    blind_market = MultiVenueMarket(
        seed=seed,
        calibrated_order_flow=True,
    )
    aware_market = MultiVenueMarket(
        seed=seed,
        calibrated_order_flow=True,
    )

    blind_dark = MidpointDarkPool(
        DarkPoolConfig(seed=seed + 30_000),
        tick_size=blind_market.tick_size,
    )
    aware_dark = MidpointDarkPool(
        DarkPoolConfig(seed=seed + 30_000),
        tick_size=aware_market.tick_size,
    )

    blind = route_hybrid_parent_order(
        blind_market,
        blind_dark,
        side=side,
        quantity=quantity,
        trader_id="toxicity_blind",
        toxicity_penalty_ticks=0.0,
    )
    aware = route_hybrid_parent_order(
        aware_market,
        aware_dark,
        side=side,
        quantity=quantity,
        trader_id="toxicity_aware",
        toxicity_penalty_ticks=toxicity_penalty_ticks,
    )

    rows = []
    for name, result in (
        ("toxicity_blind", blind),
        ("toxicity_aware", aware),
    ):
        rows.append(
            {
                "policy": name,
                "filled_quantity": result.filled_quantity,
                "remaining_quantity": result.remaining_quantity,
                "fill_ratio": (
                    result.filled_quantity
                    / result.requested_quantity
                ),
                "average_execution_price": result.average_execution_price,
                "explicit_fees": result.explicit_fees,
                "weighted_selected_toxicity": _weighted_selected_toxicity(
                    result.decisions
                ),
                "decisions": len(result.decisions),
                "market_steps": result.steps,
            }
        )

    table = pd.DataFrame(rows).set_index("policy")
    table["toxicity_reduction_vs_blind"] = (
        table.loc[
            "toxicity_blind",
            "weighted_selected_toxicity",
        ]
        - table["weighted_selected_toxicity"]
    )

    return table, {
        "toxicity_blind": blind,
        "toxicity_aware": aware,
        "blind_market": blind_market,
        "aware_market": aware_market,
    }
