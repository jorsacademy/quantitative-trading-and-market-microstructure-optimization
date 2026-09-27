"""Hybrid smart order routing across lit taker, lit maker, and dark midpoint.

The router decides both venue and execution mode. Passive maker candidates use
queue-ahead, imbalance, latency, and patience to estimate fill probability.
Dark-pool candidates use midpoint execution with stochastic hidden liquidity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .dark_pool import MidpointDarkPool
from .environment import MultiVenueMarket, MultiVenueSnapshot
from .router import route_parent_order


@dataclass(frozen=True)
class HybridRoutingResult:
    side: str
    requested_quantity: int
    filled_quantity: int
    remaining_quantity: int
    decisions: pd.DataFrame
    lit_executions: pd.DataFrame
    dark_executions: pd.DataFrame
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


def _maker_fill_probability(
    *,
    queue_ahead: float,
    imbalance: float,
    latency_steps: int,
    patience_steps: int,
    side: str,
) -> float:
    queue_component = 0.60 * np.exp(
        -queue_ahead / max(20.0, 18.0 * patience_steps)
    )
    patience_component = 0.05 * min(patience_steps, 5)

    if side == "buy":
        imbalance_component = -0.18 * imbalance
    else:
        imbalance_component = 0.18 * imbalance

    latency_component = -0.04 * latency_steps

    return float(
        np.clip(
            0.08
            + queue_component
            + patience_component
            + imbalance_component
            + latency_component,
            0.03,
            0.92,
        )
    )


def candidate_table(
    snapshot: MultiVenueSnapshot,
    dark_pool: MidpointDarkPool,
    *,
    side: str,
    maker_patience_steps: int = 3,
    maximum_passive_quantity_per_venue: int = 40,
    failure_penalty_ticks: float = 1.50,
    latency_penalty_per_step_ticks: float = 0.12,
) -> pd.DataFrame:
    """Build live venue×mode candidates for one routing wave."""
    rows = []
    mid = snapshot.consolidated_mid_tick

    for venue, row in snapshot.venues.iterrows():
        latency = int(row["latency_steps"])
        fee_tick_taker = (
            float(row["taker_fee_bps"])
            / 10_000.0
            * mid
        )
        fee_tick_maker = (
            float(row["maker_rebate_bps"])
            / 10_000.0
            * mid
        )

        if side == "buy":
            taker_price = float(row["best_ask_tick"])
            maker_price = float(row["best_bid_tick"])
            taker_capacity = int(row["ask_depth"])
            queue_ahead = float(row["bid_depth"])
        elif side == "sell":
            taker_price = -float(row["best_bid_tick"])
            maker_price = -float(row["best_ask_tick"])
            taker_capacity = int(row["bid_depth"])
            queue_ahead = float(row["ask_depth"])
        else:
            raise ValueError("side must be buy or sell")

        taker_fill_probability = float(
            np.clip(
                0.99 - 0.05 * latency,
                0.65,
                0.99,
            )
        )
        taker_cost = (
            taker_price
            + fee_tick_taker
            + latency_penalty_per_step_ticks * latency
            + failure_penalty_ticks
            * (1.0 - taker_fill_probability)
        )

        rows.append(
            {
                "candidate": f"{venue}:lit_taker",
                "venue": venue,
                "mode": "lit_taker",
                "price_tick": (
                    float(row["best_ask_tick"])
                    if side == "buy"
                    else float(row["best_bid_tick"])
                ),
                "capacity": max(0, taker_capacity),
                "fill_probability": taker_fill_probability,
                "queue_ahead": 0.0,
                "unit_cost": taker_cost,
                "latency_steps": latency,
            }
        )

        maker_probability = _maker_fill_probability(
            queue_ahead=queue_ahead,
            imbalance=float(row["imbalance"]),
            latency_steps=latency,
            patience_steps=maker_patience_steps,
            side=side,
        )

        if side == "buy":
            fallback_tick = float(snapshot.nbbo_ask_tick)
        else:
            fallback_tick = -float(snapshot.nbbo_bid_tick)

        maker_cost_if_fill = maker_price + fee_tick_maker
        maker_cost_if_fail = (
            fallback_tick
            + failure_penalty_ticks
            + latency_penalty_per_step_ticks
            * (latency + maker_patience_steps)
        )
        maker_expected_cost = (
            maker_probability * maker_cost_if_fill
            + (1.0 - maker_probability) * maker_cost_if_fail
        )

        rows.append(
            {
                "candidate": f"{venue}:lit_maker",
                "venue": venue,
                "mode": "lit_maker",
                "price_tick": (
                    float(row["best_bid_tick"])
                    if side == "buy"
                    else float(row["best_ask_tick"])
                ),
                "capacity": int(
                    min(
                        maximum_passive_quantity_per_venue,
                        max(1.0, queue_ahead),
                    )
                ),
                "fill_probability": maker_probability,
                "queue_ahead": queue_ahead,
                "unit_cost": maker_expected_cost,
                "latency_steps": latency,
            }
        )

    dark_probability = dark_pool.expected_fill_probability(
        horizon_steps=maker_patience_steps,
    )
    dark_fee_ticks = (
        dark_pool.config.fee_bps
        / 10_000.0
        * mid
    )
    midpoint_cost = mid if side == "buy" else -mid
    fallback = (
        float(snapshot.nbbo_ask_tick)
        if side == "buy"
        else -float(snapshot.nbbo_bid_tick)
    )
    dark_expected_cost = (
        dark_probability * (midpoint_cost + dark_fee_ticks)
        + (1.0 - dark_probability)
        * (
            fallback
            + failure_penalty_ticks
            + latency_penalty_per_step_ticks
            * dark_pool.config.latency_steps
        )
    )

    rows.append(
        {
            "candidate": f"{dark_pool.config.name}:dark_midpoint",
            "venue": dark_pool.config.name,
            "mode": "dark_midpoint",
            "price_tick": mid,
            "capacity": dark_pool.config.maximum_order_quantity,
            "fill_probability": dark_probability,
            "queue_ahead": np.nan,
            "unit_cost": dark_expected_cost,
            "latency_steps": dark_pool.config.latency_steps,
        }
    )

    return pd.DataFrame(rows).set_index("candidate")


def hybrid_route_once(
    snapshot: MultiVenueSnapshot,
    dark_pool: MidpointDarkPool,
    *,
    side: str,
    quantity: int,
    maker_patience_steps: int = 3,
    maximum_active_actions: int = 5,
    minimum_expected_fill_ratio: float = 0.70,
) -> pd.DataFrame:
    """Solve one hybrid maker/taker/dark allocation MILP."""
    candidates = candidate_table(
        snapshot,
        dark_pool,
        side=side,
        maker_patience_steps=maker_patience_steps,
    )
    candidates = candidates[candidates["capacity"] > 0].copy()

    if quantity <= 0 or candidates.empty:
        result = candidates.copy()
        result["quantity"] = 0
        return result

    n = len(candidates)
    capacities = candidates["capacity"].to_numpy(int)
    fill_prob = candidates["fill_probability"].to_numpy(float)
    costs = candidates["unit_cost"].to_numpy(float)

    active_limit = min(maximum_active_actions, n)
    reachable = int(np.sort(capacities)[-active_limit:].sum())
    target = min(int(quantity), int(capacities.sum()), reachable)

    if target <= 0:
        candidates["quantity"] = 0
        return candidates

    # x[candidate] integer quantity | y[candidate] activation
    n_vars = 2 * n
    c = np.zeros(n_vars)
    c[:n] = costs
    c[n:] = 0.02

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
    rhs.append(float(active_limit))

    # Expected fills must cover a configured fraction of the wave.
    row = np.zeros(n_vars)
    row[:n] = -fill_prob
    rows.append(row)
    rhs.append(
        -minimum_expected_fill_ratio * target
    )

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
            f"hybrid routing MILP failed: {result.message}"
        )

    candidates["quantity"] = np.rint(
        result.x[:n]
    ).astype(int)
    return candidates


def route_hybrid_parent_order(
    market: MultiVenueMarket,
    dark_pool: MidpointDarkPool,
    *,
    side: str,
    quantity: int,
    trader_id: str = "hybrid_router",
    maker_patience_steps: int = 3,
    max_steps: int = 20,
) -> HybridRoutingResult:
    """Execute a parent order using lit taker, lit maker, and dark midpoint."""
    requested = int(quantity)
    decisions = []
    maker_birth: dict[str, int] = {}
    cancel_requested: set[str] = set()

    for decision_step in range(max_steps):
        lit_frame = market.execution_frame(trader_id)
        lit_filled = (
            int(lit_frame["filled_quantity"].sum())
            if not lit_frame.empty
            else 0
        )
        dark_filled = int(
            sum(
                execution.filled_quantity
                for execution in dark_pool.executions
                if execution.trader_id == trader_id
            )
        )
        total_filled = lit_filled + dark_filled

        open_lit = market.open_orders(trader_id)
        open_lit_quantity = (
            int(open_lit["remaining"].sum())
            if not open_lit.empty
            else 0
        )
        committed = (
            market.pending_quantity(trader_id)
            + open_lit_quantity
            + dark_pool.pending_quantity(trader_id)
        )
        allocatable = max(
            0,
            requested - total_filled - committed,
        )

        # Cancel passive lit orders after their patience budget.
        if not open_lit.empty:
            for _, order in open_lit.iterrows():
                order_id = str(order["order_id"])
                born = maker_birth.get(order_id, decision_step)
                if (
                    decision_step - born >= maker_patience_steps
                    and order_id not in cancel_requested
                ):
                    market.schedule_cancel(
                        venue=str(order["venue"]),
                        trader_id=trader_id,
                        order_id=order_id,
                    )
                    cancel_requested.add(order_id)

        # Expire stale dark orders locally after the same patience window.
        for _, order in dark_pool.open_orders(trader_id).iterrows():
            if (
                dark_pool.time - int(order["eligible_time"])
                >= maker_patience_steps
            ):
                dark_pool.cancel(str(order["order_id"]))

        if total_filled >= requested and committed == 0:
            break

        if allocatable > 0:
            snapshot = market.snapshot()
            allocation = hybrid_route_once(
                snapshot,
                dark_pool,
                side=side,
                quantity=allocatable,
                maker_patience_steps=maker_patience_steps,
            )

            for candidate, row in allocation.iterrows():
                child_quantity = int(row["quantity"])
                if child_quantity <= 0:
                    continue

                mode = str(row["mode"])
                venue = str(row["venue"])

                if mode == "lit_taker":
                    order_id = market.schedule_market_order(
                        venue=venue,
                        trader_id=trader_id,
                        side=side,  # type: ignore[arg-type]
                        quantity=child_quantity,
                    )
                elif mode == "lit_maker":
                    order_id = (
                        f"{trader_id}-{venue}-maker-{decision_step}"
                    )
                    market.schedule_limit_order(
                        venue=venue,
                        trader_id=trader_id,
                        side=side,  # type: ignore[arg-type]
                        quantity=child_quantity,
                        price_tick=int(round(float(row["price_tick"]))),
                        order_id=order_id,
                    )
                    maker_birth[order_id] = decision_step
                else:
                    order_id = dark_pool.submit(
                        trader_id=trader_id,
                        side=side,  # type: ignore[arg-type]
                        quantity=min(
                            child_quantity,
                            dark_pool.config.maximum_order_quantity,
                        ),
                        tif_steps=maker_patience_steps,
                    )

                decisions.append(
                    {
                        "decision_step": decision_step,
                        "market_time": market.time,
                        "candidate": candidate,
                        "venue": venue,
                        "mode": mode,
                        "quantity": child_quantity,
                        "expected_fill_probability": float(
                            row["fill_probability"]
                        ),
                        "queue_ahead": row["queue_ahead"],
                        "unit_cost": float(row["unit_cost"]),
                        "order_id": order_id,
                    }
                )

        snapshot_before = market.snapshot()
        market.step()
        dark_pool.step(
            midpoint_tick=snapshot_before.consolidated_mid_tick,
        )

    # Cancel residual passive orders before deadline fallback.
    open_lit = market.open_orders(trader_id)
    if not open_lit.empty:
        for _, order in open_lit.iterrows():
            market.schedule_cancel(
                venue=str(order["venue"]),
                trader_id=trader_id,
                order_id=str(order["order_id"]),
            )
    for _, order in dark_pool.open_orders(trader_id).iterrows():
        dark_pool.cancel(str(order["order_id"]))

    market.flush_pending(max_steps=10)

    # A delayed maker placement may have arrived during the flush above.
    # Cancel any newly resting passive orders before deadline crossing.
    post_flush_open = market.open_orders(trader_id)
    if not post_flush_open.empty:
        for _, order in post_flush_open.iterrows():
            market.schedule_cancel(
                venue=str(order["venue"]),
                trader_id=trader_id,
                order_id=str(order["order_id"]),
            )
        market.flush_pending(max_steps=10)

    lit_frame = market.execution_frame(trader_id)
    lit_filled = (
        int(lit_frame["filled_quantity"].sum())
        if not lit_frame.empty
        else 0
    )
    dark_frame = dark_pool.execution_frame(trader_id)
    dark_filled = (
        int(dark_frame["filled_quantity"].sum())
        if not dark_frame.empty
        else 0
    )
    total_filled = lit_filled + dark_filled

    # Hard deadline: cross residual in lit venues.
    residual = max(0, requested - total_filled)
    if residual > 0:
        fallback = route_parent_order(
            market,
            side=side,
            quantity=residual,
            trader_id=trader_id,
            max_steps=10,
        )
        if not fallback.route_decisions.empty:
            extra = fallback.route_decisions.copy()
            extra["mode"] = "deadline_lit_taker"
            extra["candidate"] = (
                extra["venue"].astype(str)
                + ":deadline_lit_taker"
            )
            extra["expected_fill_probability"] = 1.0
            extra["queue_ahead"] = 0.0
            extra["unit_cost"] = np.nan
            extra["order_id"] = ""
            decisions.extend(extra.to_dict(orient="records"))

    lit_frame = market.execution_frame(trader_id)
    lit_filled = (
        int(lit_frame["filled_quantity"].sum())
        if not lit_frame.empty
        else 0
    )
    dark_frame = dark_pool.execution_frame(trader_id)
    dark_filled = (
        int(dark_frame["filled_quantity"].sum())
        if not dark_frame.empty
        else 0
    )
    total_filled = min(requested, lit_filled + dark_filled)
    remaining = max(0, requested - total_filled)

    prices = []
    weights = []
    fees = 0.0

    if not lit_frame.empty:
        for _, row in lit_frame.iterrows():
            if int(row["filled_quantity"]) <= 0:
                continue
            weights.append(float(row["filled_quantity"]))
            prices.append(float(row["average_price"]))
            fees += float(row["explicit_fee"])

    if not dark_frame.empty:
        for _, row in dark_frame.iterrows():
            if int(row["filled_quantity"]) <= 0:
                continue
            weights.append(float(row["filled_quantity"]))
            prices.append(float(row["average_price"]))
            fees += float(row["explicit_fee"])

    average_price = (
        float(np.dot(weights, prices) / np.sum(weights))
        if weights
        else None
    )

    return HybridRoutingResult(
        side=side,
        requested_quantity=requested,
        filled_quantity=total_filled,
        remaining_quantity=remaining,
        decisions=pd.DataFrame(decisions),
        lit_executions=lit_frame,
        dark_executions=dark_frame,
        average_execution_price=average_price,
        explicit_fees=float(fees),
        steps=market.time,
    )
