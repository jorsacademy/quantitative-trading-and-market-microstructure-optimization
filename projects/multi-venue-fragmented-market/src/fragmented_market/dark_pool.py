"""Synthetic midpoint dark pool with hidden liquidity and delayed fills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class DarkPoolConfig:
    name: str = "dark_midpoint"
    latency_steps: int = 1
    base_match_probability: float = 0.58
    hidden_liquidity_mean: float = 45.0
    maximum_order_quantity: int = 80
    fee_bps: float = 0.04
    default_tif_steps: int = 4
    seed: int = 909


@dataclass
class DarkOrder:
    order_id: str
    trader_id: str
    side: Side
    quantity: int
    remaining: int
    submit_time: int
    eligible_time: int
    expiry_time: int


@dataclass(frozen=True)
class DarkExecution:
    order_id: str
    trader_id: str
    side: Side
    filled_quantity: int
    price: float
    explicit_fee: float
    execution_time: int


class MidpointDarkPool:
    """Midpoint crossing pool with unobserved stochastic contra liquidity."""

    def __init__(
        self,
        config: DarkPoolConfig | None = None,
        *,
        tick_size: float = 0.01,
    ) -> None:
        self.config = config or DarkPoolConfig()
        self.tick_size = float(tick_size)
        self.rng = np.random.default_rng(self.config.seed)
        self.time = 0
        self.orders: dict[str, DarkOrder] = {}
        self.executions: list[DarkExecution] = []
        self._counter = 0

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(
            self.config.seed if seed is None else seed
        )
        self.time = 0
        self.orders = {}
        self.executions = []
        self._counter = 0

    def _next_id(self, trader_id: str) -> str:
        self._counter += 1
        return f"{trader_id}-{self.config.name}-{self._counter:08d}"

    def submit(
        self,
        *,
        trader_id: str,
        side: Side,
        quantity: int,
        tif_steps: int | None = None,
        order_id: str | None = None,
    ) -> str:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if quantity > self.config.maximum_order_quantity:
            raise ValueError("dark order exceeds maximum_order_quantity")

        tif = (
            self.config.default_tif_steps
            if tif_steps is None
            else int(tif_steps)
        )
        if tif <= 0:
            raise ValueError("tif_steps must be positive")

        oid = order_id or self._next_id(trader_id)
        if oid in self.orders:
            raise ValueError(f"duplicate dark order_id: {oid}")

        self.orders[oid] = DarkOrder(
            order_id=oid,
            trader_id=trader_id,
            side=side,
            quantity=int(quantity),
            remaining=int(quantity),
            submit_time=self.time,
            eligible_time=self.time + self.config.latency_steps,
            expiry_time=self.time + self.config.latency_steps + tif,
        )
        return oid

    def cancel(self, order_id: str) -> int:
        order = self.orders.pop(order_id, None)
        return 0 if order is None else int(order.remaining)

    def expected_fill_probability(
        self,
        *,
        horizon_steps: int = 1,
    ) -> float:
        if horizon_steps <= 0:
            return 0.0
        p = self.config.base_match_probability
        return float(1.0 - (1.0 - p) ** horizon_steps)

    def step(
        self,
        *,
        midpoint_tick: float,
    ) -> tuple[DarkExecution, ...]:
        results: list[DarkExecution] = []

        active = list(self.orders.values())
        for order in active:
            if order.order_id not in self.orders:
                continue
            if self.time < order.eligible_time:
                continue
            if self.time >= order.expiry_time:
                del self.orders[order.order_id]
                continue

            if self.rng.random() >= self.config.base_match_probability:
                continue

            hidden = int(
                max(
                    1,
                    self.rng.poisson(
                        self.config.hidden_liquidity_mean
                    ),
                )
            )
            filled = min(order.remaining, hidden)
            if filled <= 0:
                continue

            price = float(midpoint_tick * self.tick_size)
            notional = price * filled
            fee = notional * self.config.fee_bps / 10_000.0

            execution = DarkExecution(
                order_id=order.order_id,
                trader_id=order.trader_id,
                side=order.side,
                filled_quantity=int(filled),
                price=price,
                explicit_fee=float(fee),
                execution_time=self.time,
            )
            results.append(execution)
            self.executions.append(execution)

            order.remaining -= int(filled)
            if order.remaining <= 0:
                del self.orders[order.order_id]

        self.time += 1
        return tuple(results)

    def pending_quantity(self, trader_id: str) -> int:
        return int(
            sum(
                order.remaining
                for order in self.orders.values()
                if order.trader_id == trader_id
            )
        )

    def open_orders(self, trader_id: str | None = None) -> pd.DataFrame:
        rows = []
        for order in self.orders.values():
            if trader_id is not None and order.trader_id != trader_id:
                continue
            rows.append(
                {
                    "order_id": order.order_id,
                    "trader_id": order.trader_id,
                    "side": order.side,
                    "remaining": order.remaining,
                    "eligible_time": order.eligible_time,
                    "expiry_time": order.expiry_time,
                }
            )
        return pd.DataFrame(rows)

    def execution_frame(
        self,
        trader_id: str | None = None,
    ) -> pd.DataFrame:
        rows = []
        for execution in self.executions:
            if (
                trader_id is not None
                and execution.trader_id != trader_id
            ):
                continue
            rows.append(
                {
                    "venue": self.config.name,
                    "side": execution.side,
                    "filled_quantity": execution.filled_quantity,
                    "average_price": execution.price,
                    "explicit_fee": execution.explicit_fee,
                    "execution_time": execution.execution_time,
                    "trader_id": execution.trader_id,
                    "liquidity_role": "dark_midpoint",
                    "order_id": execution.order_id,
                }
            )
        return pd.DataFrame(rows)
