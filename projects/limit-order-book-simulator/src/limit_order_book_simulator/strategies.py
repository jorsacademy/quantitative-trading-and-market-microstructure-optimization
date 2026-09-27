"""Strategy adapters that run repository models on the shared LOB environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from limit_order_placement.model import (
    default_problem as default_limit_problem,
    solve as solve_limit_policy,
)
from market_making_inventory_control.avellaneda_stoikov import (
    default_problem as default_as_problem,
    solve_avellaneda_stoikov,
)

from .environment import AgentAction, LOBEnvironment


@dataclass(frozen=True)
class StrategyRun:
    steps: pd.DataFrame
    trades: pd.DataFrame
    terminal_inventory: int
    terminal_mark_to_market: float


def _trade_frame(env: LOBEnvironment, trader_id: str) -> pd.DataFrame:
    rows = []
    for trade in env.book.trades:
        if trader_id not in (
            trade.maker_trader_id,
            trade.taker_trader_id,
        ):
            continue
        role = (
            "maker"
            if trade.maker_trader_id == trader_id
            else "taker"
        )
        side = (
            "sell"
            if role == "maker" and trade.taker_side == "buy"
            else "buy"
            if role == "maker"
            else trade.taker_side
        )
        rows.append(
            {
                "timestamp": trade.timestamp,
                "price_tick": trade.price_tick,
                "quantity": trade.quantity,
                "role": role,
                "side": side,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "price_tick",
            "quantity",
            "role",
            "side",
        ],
    )


def run_execution_schedule(
    env: LOBEnvironment,
    schedule: pd.Series,
    *,
    trader_id: str = "execution_agent",
    side: str = "sell",
    background_events: int | None = None,
) -> StrategyRun:
    """Execute a fixed schedule as market orders against the event-driven book."""
    rows = []

    for step, quantity in enumerate(schedule.to_numpy()):
        qty = int(round(float(quantity)))
        actions = []
        if qty > 0:
            actions.append(
                AgentAction(
                    action_type="market",
                    trader_id=trader_id,
                    side=side,  # type: ignore[arg-type]
                    quantity=qty,
                )
            )

        result = env.step(
            actions,
            background_events=background_events,
        )
        obs = result.observation

        rows.append(
            {
                "step": step,
                "scheduled_quantity": qty,
                "inventory": env.inventory.get(trader_id, 0),
                "cash": env.cash.get(trader_id, 0.0),
                "mid_tick": obs.mid_tick,
                "spread_ticks": obs.spread_ticks,
                "imbalance": obs.imbalance,
                "mark_to_market": env.mark_to_market(trader_id),
            }
        )

    return StrategyRun(
        steps=pd.DataFrame(rows),
        trades=_trade_frame(env, trader_id),
        terminal_inventory=env.inventory.get(trader_id, 0),
        terminal_mark_to_market=env.mark_to_market(trader_id),
    )


def run_market_maker(
    env: LOBEnvironment,
    *,
    steps: int = 30,
    trader_id: str = "market_maker",
    quote_size: int = 8,
    maximum_inventory: int = 6,
    background_events: int | None = None,
) -> StrategyRun:
    """Run an Avellaneda-Stoikov-style inventory-skewed quoting adapter."""
    as_problem = default_as_problem()
    as_result = solve_avellaneda_stoikov(as_problem)
    rows = []

    for step in range(steps):
        obs = env.observe()
        inventory = env.inventory.get(trader_id, 0)

        # Cancel existing quotes before reposting.
        actions = [
            AgentAction(
                action_type="cancel",
                trader_id=trader_id,
                order_id=order_id,
            )
            for order_id in env.open_orders(trader_id)["order_id"].tolist()
        ]

        inv_state = int(
            np.clip(
                inventory,
                -as_problem.maximum_inventory,
                as_problem.maximum_inventory,
            )
        )
        as_time = min(step, as_problem.horizon - 1)
        quote = as_result.quote_surface.loc[
            (as_time, inv_state)
        ]

        bid_offset = max(
            1,
            int(round(float(quote["bid_offset"]))),
        )
        ask_offset = max(
            1,
            int(round(float(quote["ask_offset"]))),
        )

        if inventory < maximum_inventory:
            actions.append(
                AgentAction(
                    action_type="limit",
                    trader_id=trader_id,
                    side="buy",
                    quantity=quote_size,
                    price_tick=max(
                        1,
                        obs.best_bid_tick - bid_offset + 1,
                    ),
                    order_id=f"{trader_id}-bid-{step}",
                )
            )

        if inventory > -maximum_inventory:
            actions.append(
                AgentAction(
                    action_type="limit",
                    trader_id=trader_id,
                    side="sell",
                    quantity=quote_size,
                    price_tick=obs.best_ask_tick + ask_offset - 1,
                    order_id=f"{trader_id}-ask-{step}",
                )
            )

        result = env.step(
            actions,
            background_events=background_events,
        )
        after = result.observation

        rows.append(
            {
                "step": step,
                "inventory": env.inventory.get(trader_id, 0),
                "cash": env.cash.get(trader_id, 0.0),
                "mid_tick": after.mid_tick,
                "spread_ticks": after.spread_ticks,
                "imbalance": after.imbalance,
                "open_orders": len(env.open_orders(trader_id)),
                "mark_to_market": env.mark_to_market(trader_id),
            }
        )

    return StrategyRun(
        steps=pd.DataFrame(rows),
        trades=_trade_frame(env, trader_id),
        terminal_inventory=env.inventory.get(trader_id, 0),
        terminal_mark_to_market=env.mark_to_market(trader_id),
    )


def _queue_state(queue_ahead: int) -> int:
    if queue_ahead <= 0:
        return 2
    if queue_ahead <= 25:
        return 1
    return 0


def _imbalance_state(imbalance: float) -> int:
    if imbalance <= -0.15:
        return -1
    if imbalance >= 0.15:
        return 1
    return 0


def run_limit_order_policy(
    env: LOBEnvironment,
    *,
    quantity: int = 10,
    trader_id: str = "limit_agent",
    background_events: int | None = None,
) -> StrategyRun:
    """Run the repository's placement DP using live queue/imbalance states."""
    problem = default_limit_problem()
    policy = solve_limit_policy(problem).policy
    remaining = int(quantity)
    active_order_id: str | None = None
    rows = []

    for step in range(problem.horizon + 1):
        if remaining <= 0:
            break

        obs = env.observe()
        open_orders = env.open_orders(trader_id)

        queue_state = 0
        if (
            active_order_id is not None
            and not open_orders.empty
            and active_order_id in set(open_orders["order_id"])
        ):
            queue_ahead = int(
                open_orders.loc[
                    open_orders["order_id"] == active_order_id,
                    "queue_ahead",
                ].iloc[0]
            )
            queue_state = _queue_state(queue_ahead)
        else:
            active_order_id = None

        imbalance_state = _imbalance_state(obs.imbalance)
        policy_time = min(step, problem.horizon)
        action = str(
            policy.loc[
                (
                    policy_time,
                    queue_state,
                    imbalance_state,
                )
            ]
        )

        actions: list[AgentAction] = []

        if action == "market":
            if active_order_id is not None:
                actions.append(
                    AgentAction(
                        action_type="cancel",
                        trader_id=trader_id,
                        order_id=active_order_id,
                    )
                )
                active_order_id = None
            actions.append(
                AgentAction(
                    action_type="market",
                    trader_id=trader_id,
                    side="buy",
                    quantity=remaining,
                )
            )

        elif action in ("rest", "improve"):
            target_tick = (
                obs.best_bid_tick
                if action == "rest"
                else min(
                    obs.best_ask_tick,
                    obs.best_bid_tick + 1,
                )
            )

            needs_repost = True
            if active_order_id is not None:
                live = env.book.order(active_order_id)
                if (
                    live is not None
                    and live.price_tick == target_tick
                ):
                    needs_repost = False

            if needs_repost:
                if active_order_id is not None:
                    actions.append(
                        AgentAction(
                            action_type="cancel",
                            trader_id=trader_id,
                            order_id=active_order_id,
                        )
                    )
                active_order_id = (
                    f"{trader_id}-{action}-{step}"
                )
                actions.append(
                    AgentAction(
                        action_type="limit",
                        trader_id=trader_id,
                        side="buy",
                        quantity=remaining,
                        price_tick=target_tick,
                        order_id=active_order_id,
                    )
                )

        elif action == "wait":
            if active_order_id is not None:
                actions.append(
                    AgentAction(
                        action_type="cancel",
                        trader_id=trader_id,
                        order_id=active_order_id,
                    )
                )
                active_order_id = None

        before_inventory = env.inventory.get(trader_id, 0)
        result = env.step(
            actions,
            background_events=background_events,
        )
        after_inventory = env.inventory.get(trader_id, 0)
        newly_filled = max(0, after_inventory - before_inventory)
        remaining = max(0, remaining - newly_filled)

        rows.append(
            {
                "step": step,
                "action": action,
                "queue_state": queue_state,
                "imbalance_state": imbalance_state,
                "remaining": remaining,
                "inventory": after_inventory,
                "mid_tick": result.observation.mid_tick,
                "spread_ticks": result.observation.spread_ticks,
                "mark_to_market": env.mark_to_market(trader_id),
            }
        )

    # Deadline safety: if passive order remains, cross any residual.
    if remaining > 0:
        actions = []
        if active_order_id is not None:
            actions.append(
                AgentAction(
                    action_type="cancel",
                    trader_id=trader_id,
                    order_id=active_order_id,
                )
            )
        actions.append(
            AgentAction(
                action_type="market",
                trader_id=trader_id,
                side="buy",
                quantity=remaining,
            )
        )
        env.step(actions, background_events=0)

    return StrategyRun(
        steps=pd.DataFrame(rows),
        trades=_trade_frame(env, trader_id),
        terminal_inventory=env.inventory.get(trader_id, 0),
        terminal_mark_to_market=env.mark_to_market(trader_id),
    )
