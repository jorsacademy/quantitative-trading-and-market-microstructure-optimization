"""Price-time-priority limit order book matching engine.

The engine supports:

- limit orders;
- market orders;
- order cancellation;
- partial fills;
- price-time priority;
- depth snapshots;
- maker/taker trade attribution.

Prices are represented as integer ticks to avoid floating-point ordering issues.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Literal


Side = Literal["buy", "sell"]


@dataclass
class Order:
    order_id: str
    trader_id: str
    side: Side
    price_tick: int | None
    quantity: int
    remaining: int
    timestamp: int
    sequence: int
    is_market: bool = False


@dataclass(frozen=True)
class Trade:
    price_tick: int
    quantity: int
    maker_order_id: str
    taker_order_id: str
    maker_trader_id: str
    taker_trader_id: str
    taker_side: Side
    timestamp: int


class LimitOrderBook:
    """Single-instrument continuous double auction."""

    def __init__(self) -> None:
        self.orders: dict[str, Order] = {}
        self.bids: dict[int, deque[str]] = defaultdict(deque)
        self.asks: dict[int, deque[str]] = defaultdict(deque)
        self.trades: list[Trade] = []
        self._sequence = 0

    def clear(self) -> None:
        self.orders.clear()
        self.bids.clear()
        self.asks.clear()
        self.trades.clear()
        self._sequence = 0

    @property
    def best_bid_tick(self) -> int | None:
        self._drop_empty_levels()
        return max(self.bids) if self.bids else None

    @property
    def best_ask_tick(self) -> int | None:
        self._drop_empty_levels()
        return min(self.asks) if self.asks else None

    @property
    def spread_ticks(self) -> int | None:
        bid = self.best_bid_tick
        ask = self.best_ask_tick
        if bid is None or ask is None:
            return None
        return ask - bid

    @property
    def mid_tick(self) -> float | None:
        bid = self.best_bid_tick
        ask = self.best_ask_tick
        if bid is None or ask is None:
            return None
        return 0.5 * (bid + ask)

    def _book_for_side(self, side: Side) -> dict[int, deque[str]]:
        return self.bids if side == "buy" else self.asks

    def _opposite_book(self, side: Side) -> dict[int, deque[str]]:
        return self.asks if side == "buy" else self.bids

    def _drop_empty_levels(self) -> None:
        for book in (self.bids, self.asks):
            empty = []
            for price, queue in book.items():
                while queue and queue[0] not in self.orders:
                    queue.popleft()
                if not queue:
                    empty.append(price)
            for price in empty:
                del book[price]

    def _crosses(
        self,
        side: Side,
        limit_tick: int | None,
        opposite_tick: int,
        is_market: bool,
    ) -> bool:
        if is_market:
            return True
        if limit_tick is None:
            return False
        if side == "buy":
            return limit_tick >= opposite_tick
        return limit_tick <= opposite_tick

    def _best_opposite_tick(self, side: Side) -> int | None:
        if side == "buy":
            return self.best_ask_tick
        return self.best_bid_tick

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def submit_limit(
        self,
        *,
        order_id: str,
        trader_id: str,
        side: Side,
        price_tick: int,
        quantity: int,
        timestamp: int,
    ) -> list[Trade]:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if price_tick <= 0:
            raise ValueError("price_tick must be positive")
        if order_id in self.orders:
            raise ValueError(f"duplicate order_id: {order_id}")

        order = Order(
            order_id=order_id,
            trader_id=trader_id,
            side=side,
            price_tick=int(price_tick),
            quantity=int(quantity),
            remaining=int(quantity),
            timestamp=int(timestamp),
            sequence=self._next_sequence(),
            is_market=False,
        )
        return self._submit(order)

    def submit_market(
        self,
        *,
        order_id: str,
        trader_id: str,
        side: Side,
        quantity: int,
        timestamp: int,
    ) -> list[Trade]:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if order_id in self.orders:
            raise ValueError(f"duplicate order_id: {order_id}")

        order = Order(
            order_id=order_id,
            trader_id=trader_id,
            side=side,
            price_tick=None,
            quantity=int(quantity),
            remaining=int(quantity),
            timestamp=int(timestamp),
            sequence=self._next_sequence(),
            is_market=True,
        )
        return self._submit(order)

    def _submit(self, incoming: Order) -> list[Trade]:
        fills: list[Trade] = []

        while incoming.remaining > 0:
            opposite_tick = self._best_opposite_tick(incoming.side)
            if opposite_tick is None:
                break

            if not self._crosses(
                incoming.side,
                incoming.price_tick,
                opposite_tick,
                incoming.is_market,
            ):
                break

            opposite_book = self._opposite_book(incoming.side)
            queue = opposite_book[opposite_tick]

            while queue and incoming.remaining > 0:
                maker_id = queue[0]

                if maker_id not in self.orders:
                    queue.popleft()
                    continue

                maker = self.orders[maker_id]
                quantity = min(
                    incoming.remaining,
                    maker.remaining,
                )

                trade = Trade(
                    price_tick=opposite_tick,
                    quantity=quantity,
                    maker_order_id=maker.order_id,
                    taker_order_id=incoming.order_id,
                    maker_trader_id=maker.trader_id,
                    taker_trader_id=incoming.trader_id,
                    taker_side=incoming.side,
                    timestamp=incoming.timestamp,
                )
                fills.append(trade)
                self.trades.append(trade)

                incoming.remaining -= quantity
                maker.remaining -= quantity

                if maker.remaining == 0:
                    queue.popleft()
                    del self.orders[maker.order_id]

            if not queue:
                del opposite_book[opposite_tick]

        if incoming.remaining > 0 and not incoming.is_market:
            self.orders[incoming.order_id] = incoming
            self._book_for_side(incoming.side)[
                int(incoming.price_tick)
            ].append(incoming.order_id)

        return fills

    def cancel(self, order_id: str) -> int:
        """Cancel remaining quantity and return cancelled units."""
        order = self.orders.pop(order_id, None)
        if order is None:
            return 0
        return order.remaining

    def order(self, order_id: str) -> Order | None:
        return self.orders.get(order_id)

    def depth(
        self,
        side: Side,
        levels: int | None = None,
    ) -> pd.DataFrame:
        import pandas as pd

        self._drop_empty_levels()
        book = self.bids if side == "buy" else self.asks
        prices = sorted(
            book,
            reverse=(side == "buy"),
        )
        if levels is not None:
            prices = prices[:levels]

        rows = []
        for price in prices:
            active_orders = [
                self.orders[order_id]
                for order_id in book[price]
                if order_id in self.orders
            ]
            rows.append(
                {
                    "price_tick": price,
                    "quantity": sum(
                        order.remaining
                        for order in active_orders
                    ),
                    "orders": len(active_orders),
                }
            )
        return pd.DataFrame(
            rows,
            columns=["price_tick", "quantity", "orders"],
        )

    def depth_quantity(
        self,
        side: Side,
        price_tick: int,
    ) -> int:
        book = self._book_for_side(side)
        if price_tick not in book:
            return 0
        return sum(
            self.orders[order_id].remaining
            for order_id in book[price_tick]
            if order_id in self.orders
        )

    def queue_ahead(self, order_id: str) -> int:
        """Return visible same-price quantity ahead of the specified order."""
        order = self.orders.get(order_id)
        if order is None or order.price_tick is None:
            return 0

        queue = self._book_for_side(order.side).get(
            order.price_tick,
            deque(),
        )
        ahead = 0
        for queued_id in queue:
            if queued_id == order_id:
                break
            queued = self.orders.get(queued_id)
            if queued is not None:
                ahead += queued.remaining
        return ahead
