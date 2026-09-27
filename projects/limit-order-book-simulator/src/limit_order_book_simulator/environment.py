"""Event-driven synthetic exchange environment built on the matching engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .engine import LimitOrderBook, Side, Trade


ActionType = Literal["limit", "market", "cancel"]


@dataclass(frozen=True)
class AgentAction:
    action_type: ActionType
    trader_id: str
    side: Side | None = None
    quantity: int = 0
    price_tick: int | None = None
    order_id: str | None = None


@dataclass(frozen=True)
class LOBEnvironmentConfig:
    tick_size: float = 0.01
    initial_mid_tick: int = 10_000
    initial_spread_ticks: int = 2
    initial_levels: int = 4
    initial_level_quantity: int = 80
    background_events_per_step: int = 4
    background_market_probability: float = 0.35
    background_limit_probability: float = 0.50
    background_cancel_probability: float = 0.15
    background_order_min: int = 2
    background_order_max: int = 14
    max_background_distance_ticks: int = 4
    seed: int = 23


@dataclass(frozen=True)
class Observation:
    time: int
    best_bid_tick: int
    best_ask_tick: int
    mid_tick: float
    spread_ticks: int
    bid_depth: int
    ask_depth: int
    imbalance: float
    last_trade_tick: int | None

    def to_series(self) -> pd.Series:
        return pd.Series(
            {
                "time": self.time,
                "best_bid_tick": self.best_bid_tick,
                "best_ask_tick": self.best_ask_tick,
                "mid_tick": self.mid_tick,
                "spread_ticks": self.spread_ticks,
                "bid_depth": self.bid_depth,
                "ask_depth": self.ask_depth,
                "imbalance": self.imbalance,
                "last_trade_tick": self.last_trade_tick,
            }
        )


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    trades: tuple[Trade, ...]
    trader_inventory: dict[str, int]
    trader_cash: dict[str, float]


class LOBEnvironment:
    """Small deterministic-seed exchange simulator."""

    def __init__(
        self,
        config: LOBEnvironmentConfig | None = None,
    ) -> None:
        self.config = config or LOBEnvironmentConfig()
        self.book = LimitOrderBook()
        self.rng = np.random.default_rng(self.config.seed)
        self.time = 0
        self.last_trade_tick: int | None = None
        self.inventory: dict[str, int] = {}
        self.cash: dict[str, float] = {}
        self._id_counter = 0
        self._background_ids: set[str] = set()
        self.reset()

    def _next_id(self, prefix: str) -> str:
        self._id_counter += 1
        return f"{prefix}-{self._id_counter:08d}"

    def _record_trades(self, trades: list[Trade]) -> None:
        for trade in trades:
            self.last_trade_tick = trade.price_tick
            price = trade.price_tick * self.config.tick_size
            quantity = trade.quantity

            taker_sign = 1 if trade.taker_side == "buy" else -1
            maker_sign = -taker_sign

            self.inventory[trade.taker_trader_id] = (
                self.inventory.get(trade.taker_trader_id, 0)
                + taker_sign * quantity
            )
            self.cash[trade.taker_trader_id] = (
                self.cash.get(trade.taker_trader_id, 0.0)
                - taker_sign * quantity * price
            )

            self.inventory[trade.maker_trader_id] = (
                self.inventory.get(trade.maker_trader_id, 0)
                + maker_sign * quantity
            )
            self.cash[trade.maker_trader_id] = (
                self.cash.get(trade.maker_trader_id, 0.0)
                - maker_sign * quantity * price
            )

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(self.config.seed)

        self.book.clear()
        self.time = 0
        self.last_trade_tick = None
        self.inventory = {}
        self.cash = {}
        self._id_counter = 0
        self._background_ids = set()

        half = max(1, self.config.initial_spread_ticks // 2)
        best_bid = self.config.initial_mid_tick - half
        best_ask = best_bid + self.config.initial_spread_ticks

        for level in range(self.config.initial_levels):
            quantity = self.config.initial_level_quantity + 10 * level

            bid_id = self._next_id("bg-bid")
            self._background_ids.add(bid_id)
            self.book.submit_limit(
                order_id=bid_id,
                trader_id="background",
                side="buy",
                price_tick=best_bid - level,
                quantity=quantity,
                timestamp=self.time,
            )

            ask_id = self._next_id("bg-ask")
            self._background_ids.add(ask_id)
            self.book.submit_limit(
                order_id=ask_id,
                trader_id="background",
                side="sell",
                price_tick=best_ask + level,
                quantity=quantity,
                timestamp=self.time,
            )

        return self.observe()

    def observe(self) -> Observation:
        bid = self.book.best_bid_tick
        ask = self.book.best_ask_tick

        if bid is None or ask is None:
            self._ensure_two_sided_book()
            bid = self.book.best_bid_tick
            ask = self.book.best_ask_tick

        assert bid is not None
        assert ask is not None

        bid_depth = self.book.depth_quantity("buy", bid)
        ask_depth = self.book.depth_quantity("sell", ask)
        total = bid_depth + ask_depth
        imbalance = (
            (bid_depth - ask_depth) / total
            if total > 0
            else 0.0
        )

        return Observation(
            time=self.time,
            best_bid_tick=bid,
            best_ask_tick=ask,
            mid_tick=0.5 * (bid + ask),
            spread_ticks=ask - bid,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            imbalance=float(imbalance),
            last_trade_tick=self.last_trade_tick,
        )

    def _ensure_two_sided_book(self) -> None:
        reference = (
            self.last_trade_tick
            if self.last_trade_tick is not None
            else self.config.initial_mid_tick
        )

        if self.book.best_bid_tick is None:
            order_id = self._next_id("bg-reseed-bid")
            self._background_ids.add(order_id)
            self.book.submit_limit(
                order_id=order_id,
                trader_id="background",
                side="buy",
                price_tick=max(1, int(reference) - 1),
                quantity=self.config.initial_level_quantity,
                timestamp=self.time,
            )

        if self.book.best_ask_tick is None:
            order_id = self._next_id("bg-reseed-ask")
            self._background_ids.add(order_id)
            bid = self.book.best_bid_tick
            ask_tick = (
                int(reference) + 1
                if bid is None
                else max(int(reference) + 1, bid + 1)
            )
            self.book.submit_limit(
                order_id=order_id,
                trader_id="background",
                side="sell",
                price_tick=ask_tick,
                quantity=self.config.initial_level_quantity,
                timestamp=self.time,
            )

    def apply_action(self, action: AgentAction) -> list[Trade]:
        trades: list[Trade] = []

        if action.action_type == "cancel":
            if action.order_id is not None:
                self.book.cancel(action.order_id)
            return trades

        if action.side not in ("buy", "sell"):
            raise ValueError("buy/sell side required")
        if action.quantity <= 0:
            raise ValueError("positive quantity required")

        order_id = action.order_id or self._next_id(
            f"agent-{action.trader_id}"
        )

        if action.action_type == "market":
            trades = self.book.submit_market(
                order_id=order_id,
                trader_id=action.trader_id,
                side=action.side,
                quantity=action.quantity,
                timestamp=self.time,
            )
        elif action.action_type == "limit":
            if action.price_tick is None:
                raise ValueError("limit action requires price_tick")
            trades = self.book.submit_limit(
                order_id=order_id,
                trader_id=action.trader_id,
                side=action.side,
                price_tick=action.price_tick,
                quantity=action.quantity,
                timestamp=self.time,
            )
        else:
            raise ValueError(f"unknown action type: {action.action_type}")

        self._record_trades(trades)
        return trades

    def _background_limit_event(self) -> list[Trade]:
        obs = self.observe()
        side: Side = "buy" if self.rng.random() < 0.5 else "sell"
        distance = int(
            self.rng.integers(
                0,
                self.config.max_background_distance_ticks + 1,
            )
        )
        quantity = int(
            self.rng.integers(
                self.config.background_order_min,
                self.config.background_order_max + 1,
            )
        )

        if side == "buy":
            price_tick = max(1, obs.best_bid_tick - distance)
        else:
            price_tick = obs.best_ask_tick + distance

        order_id = self._next_id("bg-limit")
        self._background_ids.add(order_id)
        trades = self.book.submit_limit(
            order_id=order_id,
            trader_id="background",
            side=side,
            price_tick=price_tick,
            quantity=quantity,
            timestamp=self.time,
        )
        self._record_trades(trades)
        return trades

    def _background_market_event(self) -> list[Trade]:
        side: Side = "buy" if self.rng.random() < 0.5 else "sell"
        quantity = int(
            self.rng.integers(
                self.config.background_order_min,
                self.config.background_order_max + 1,
            )
        )
        order_id = self._next_id("bg-market")
        trades = self.book.submit_market(
            order_id=order_id,
            trader_id="background",
            side=side,
            quantity=quantity,
            timestamp=self.time,
        )
        self._record_trades(trades)
        self._ensure_two_sided_book()
        return trades

    def _background_cancel_event(self) -> list[Trade]:
        live = [
            order_id
            for order_id in self._background_ids
            if self.book.order(order_id) is not None
        ]
        if not live:
            return []
        order_id = live[int(self.rng.integers(0, len(live)))]
        self.book.cancel(order_id)
        return []

    def background_event(self) -> list[Trade]:
        u = self.rng.random()
        p_market = self.config.background_market_probability
        p_limit = self.config.background_limit_probability

        if u < p_market:
            return self._background_market_event()
        if u < p_market + p_limit:
            return self._background_limit_event()
        return self._background_cancel_event()

    def step(
        self,
        actions: list[AgentAction] | tuple[AgentAction, ...] = (),
        background_events: int | None = None,
    ) -> StepResult:
        trades: list[Trade] = []

        for action in actions:
            trades.extend(self.apply_action(action))

        count = (
            self.config.background_events_per_step
            if background_events is None
            else int(background_events)
        )
        for _ in range(max(0, count)):
            trades.extend(self.background_event())

        self.time += 1
        self._ensure_two_sided_book()

        return StepResult(
            observation=self.observe(),
            trades=tuple(trades),
            trader_inventory=dict(self.inventory),
            trader_cash=dict(self.cash),
        )

    def mark_to_market(self, trader_id: str) -> float:
        obs = self.observe()
        mid_price = obs.mid_tick * self.config.tick_size
        return (
            self.cash.get(trader_id, 0.0)
            + self.inventory.get(trader_id, 0) * mid_price
        )

    def open_orders(self, trader_id: str) -> pd.DataFrame:
        rows = []
        for order in self.book.orders.values():
            if order.trader_id != trader_id:
                continue
            rows.append(
                {
                    "order_id": order.order_id,
                    "side": order.side,
                    "price_tick": order.price_tick,
                    "remaining": order.remaining,
                    "queue_ahead": self.book.queue_ahead(
                        order.order_id
                    ),
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "order_id",
                "side",
                "price_tick",
                "remaining",
                "queue_ahead",
            ],
        )
