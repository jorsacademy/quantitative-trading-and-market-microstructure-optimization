"""Live MILP smart order routing over fragmented venue books."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .environment import MultiVenueMarket, MultiVenueSnapshot


@dataclass(frozen=True)
class LiveRoutingResult:
    side: str
    requested_quantity: int
    filled_quantity: int
    remaining_quantity: int
    route_decisions: pd.DataFrame
    executions: pd.DataFrame
    average_execution_price: float | None
    explicit_fees: float
    steps: int

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "requested_quantity": self.requested_quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_execution_price": self.average_execution_price,
            "explicit_fees": self.explicit_fees,
            "steps": self.steps,
        }


def _venue_unit_costs(
    snapshot: MultiVenueSnapshot,
    side: str,
    *,
    latency_penalty_per_step_ticks: float,
    imbalance_penalty_ticks: float,
) -> pd.Series:
    frame = snapshot.venues

    if side == "buy":
        raw_tick = frame["best_ask_tick"].astype(float)
        imbalance_penalty = (
            imbalance_penalty_ticks
            * frame["imbalance"].clip(lower=0.0)
        )
    elif side == "sell":
        # Convert proceeds maximization into cost minimization.
        raw_tick = -frame["best_bid_tick"].astype(float)
        imbalance_penalty = (
            imbalance_penalty_ticks
            * (-frame["imbalance"]).clip(lower=0.0)
        )
    else:
        raise ValueError("side must be buy or sell")

    fee_ticks = (
        frame["taker_fee_bps"]
        / 10_000.0
        * snapshot.consolidated_mid_tick
    )
    latency = (
        latency_penalty_per_step_ticks
        * frame["latency_steps"]
    )

    return (
        raw_tick
        + fee_ticks
        + latency
        + imbalance_penalty
    )


def route_once(
    snapshot: MultiVenueSnapshot,
    *,
    side: str,
    quantity: int,
    maximum_active_venues: int = 3,
    latency_penalty_per_step_ticks: float = 0.20,
    imbalance_penalty_ticks: float = 0.40,
) -> pd.Series:
    """Allocate the next child-order wave across current top-of-book liquidity."""
    if quantity <= 0:
        return pd.Series(
            0,
            index=snapshot.venues.index,
            dtype=int,
            name="quantity",
        )

    frame = snapshot.venues
    venues = list(frame.index)
    n = len(venues)

    depth_column = (
        "ask_depth"
        if side == "buy"
        else "bid_depth"
    )
    capacities = frame[depth_column].to_numpy(int)

    total_capacity = int(capacities.sum())
    target = min(int(quantity), total_capacity)

    if target <= 0:
        return pd.Series(
            0,
            index=venues,
            dtype=int,
            name="quantity",
        )

    costs = _venue_unit_costs(
        snapshot,
        side,
        latency_penalty_per_step_ticks=latency_penalty_per_step_ticks,
        imbalance_penalty_ticks=imbalance_penalty_ticks,
    ).loc[venues].to_numpy(float)

    # x[v] integer shares | y[v] activation binaries
    n_vars = 2 * n
    c = np.zeros(n_vars)
    c[:n] = costs

    eq = np.zeros((1, n_vars))
    eq[0, :n] = 1.0

    rows = []
    rhs = []

    for i in range(n):
        row = np.zeros(n_vars)
        row[i] = 1.0
        row[n + i] = -float(capacities[i])
        rows.append(row)
        rhs.append(0.0)

    row = np.zeros(n_vars)
    row[n:] = 1.0
    rows.append(row)
    rhs.append(float(maximum_active_venues))

    result = milp(
        c=c,
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(
            np.zeros(n_vars),
            np.concatenate(
                [
                    capacities.astype(float),
                    np.ones(n),
                ]
            ),
        ),
        constraints=[
            LinearConstraint(
                eq,
                lb=np.array([float(target)]),
                ub=np.array([float(target)]),
            ),
            LinearConstraint(
                np.vstack(rows),
                lb=np.full(len(rows), -np.inf),
                ub=np.asarray(rhs, dtype=float),
            ),
        ],
        options={"disp": False},
    )

    if not result.success:
        raise RuntimeError(
            f"live smart routing failed: {result.message}"
        )

    return pd.Series(
        np.rint(result.x[:n]).astype(int),
        index=venues,
        name="quantity",
    )


def route_parent_order(
    market: MultiVenueMarket,
    *,
    side: str,
    quantity: int,
    trader_id: str = "smart_router",
    maximum_active_venues: int = 3,
    max_steps: int = 20,
) -> LiveRoutingResult:
    """Dynamically re-route residual quantity after venue fills and latency."""
    requested = int(quantity)
    remaining = int(quantity)
    decisions = []
    total_filled = 0

    for decision_step in range(max_steps):
        if remaining <= 0:
            break

        snapshot = market.snapshot()
        allocation = route_once(
            snapshot,
            side=side,
            quantity=remaining,
            maximum_active_venues=maximum_active_venues,
        )

        if int(allocation.sum()) <= 0:
            market.step()
            continue

        for venue, child_quantity in allocation.items():
            if child_quantity <= 0:
                continue

            market.schedule_market_order(
                venue=venue,
                trader_id=trader_id,
                side=side,  # type: ignore[arg-type]
                quantity=int(child_quantity),
            )
            decisions.append(
                {
                    "decision_step": decision_step,
                    "market_time": market.time,
                    "venue": venue,
                    "quantity": int(child_quantity),
                    "best_bid_tick": int(
                        snapshot.venues.loc[
                            venue,
                            "best_bid_tick",
                        ]
                    ),
                    "best_ask_tick": int(
                        snapshot.venues.loc[
                            venue,
                            "best_ask_tick",
                        ]
                    ),
                    "latency_steps": int(
                        snapshot.venues.loc[
                            venue,
                            "latency_steps",
                        ]
                    ),
                }
            )

        _next_snapshot, executions = market.step()

        newly_filled = sum(
            execution.filled_quantity
            for execution in executions
            if execution.trader_id == trader_id
        )
        total_filled += int(newly_filled)
        committed = market.pending_quantity(trader_id)
        remaining = max(
            0,
            requested - total_filled - committed,
        )

    if remaining > 0 and market.pending:
        executions = market.flush_pending(
            max_steps=max_steps,
        )
        total_filled += int(
            sum(
                execution.filled_quantity
                for execution in executions
                if execution.trader_id == trader_id
            )
        )
        remaining = max(
            0,
            requested - total_filled,
        )

    frame = market.execution_frame(trader_id)

    if (
        not frame.empty
        and frame["filled_quantity"].sum() > 0
    ):
        weights = frame["filled_quantity"].to_numpy(float)
        prices = frame["average_price"].fillna(0.0).to_numpy(float)
        average_price = float(
            np.dot(weights, prices) / weights.sum()
        )
    else:
        average_price = None

    fees = (
        float(frame["explicit_fee"].sum())
        if not frame.empty
        else 0.0
    )

    return LiveRoutingResult(
        side=side,
        requested_quantity=requested,
        filled_quantity=total_filled,
        remaining_quantity=remaining,
        route_decisions=pd.DataFrame(decisions),
        executions=frame,
        average_execution_price=average_price,
        explicit_fees=fees,
        steps=market.time,
    )



def route_parent_order_static(
    market: MultiVenueMarket,
    *,
    side: str,
    quantity: int,
    trader_id: str = "static_router",
    maximum_active_venues: int = 3,
    flush_steps: int = 20,
) -> LiveRoutingResult:
    """Route once from the initial snapshot and never re-optimize after latency/fills."""
    requested = int(quantity)
    snapshot = market.snapshot()
    allocation = route_once(
        snapshot,
        side=side,
        quantity=requested,
        maximum_active_venues=maximum_active_venues,
    )

    decisions = []
    for venue, child_quantity in allocation.items():
        if child_quantity <= 0:
            continue

        market.schedule_market_order(
            venue=venue,
            trader_id=trader_id,
            side=side,  # type: ignore[arg-type]
            quantity=int(child_quantity),
        )
        decisions.append(
            {
                "decision_step": 0,
                "market_time": market.time,
                "venue": venue,
                "quantity": int(child_quantity),
                "best_bid_tick": int(
                    snapshot.venues.loc[venue, "best_bid_tick"]
                ),
                "best_ask_tick": int(
                    snapshot.venues.loc[venue, "best_ask_tick"]
                ),
                "latency_steps": int(
                    snapshot.venues.loc[venue, "latency_steps"]
                ),
            }
        )

    # Let delayed child orders arrive, but do not re-route residual quantity.
    market.step()
    market.flush_pending(max_steps=flush_steps)

    frame = market.execution_frame(trader_id)
    total_filled = (
        int(frame["filled_quantity"].sum())
        if not frame.empty
        else 0
    )
    remaining = max(0, requested - total_filled)

    if total_filled > 0:
        weights = frame["filled_quantity"].to_numpy(float)
        prices = frame["average_price"].fillna(0.0).to_numpy(float)
        average_price = float(
            np.dot(weights, prices) / weights.sum()
        )
    else:
        average_price = None

    fees = (
        float(frame["explicit_fee"].sum())
        if not frame.empty
        else 0.0
    )

    return LiveRoutingResult(
        side=side,
        requested_quantity=requested,
        filled_quantity=total_filled,
        remaining_quantity=remaining,
        route_decisions=pd.DataFrame(decisions),
        executions=frame,
        average_execution_price=average_price,
        explicit_fees=fees,
        steps=market.time,
    )
