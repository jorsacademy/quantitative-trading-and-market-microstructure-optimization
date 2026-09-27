"""Fragmented market composed of independent event-driven venue books."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from limit_order_book_simulator.environment import (
    AgentAction,
    LOBEnvironment,
    LOBEnvironmentConfig,
)
from limit_order_book_simulator.calibrated_environment import (
    CalibratedLOBEnvironment,
)
from limit_order_book_simulator.engine import Trade


Side = Literal["buy", "sell"]
PendingType = Literal["market", "limit", "cancel"]


@dataclass(frozen=True)
class VenueConfig:
    name: str
    taker_fee_bps: float
    maker_rebate_bps: float
    latency_steps: int
    initial_mid_offset_ticks: int
    initial_spread_ticks: int
    initial_level_quantity: int
    background_events_per_step: int
    seed_offset: int


@dataclass(frozen=True)
class MultiVenueSnapshot:
    time: int
    venues: pd.DataFrame
    nbbo_bid_tick: int
    nbbo_ask_tick: int
    consolidated_mid_tick: float

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "nbbo_bid_tick": self.nbbo_bid_tick,
            "nbbo_ask_tick": self.nbbo_ask_tick,
            "consolidated_mid_tick": self.consolidated_mid_tick,
            "venues": self.venues.to_dict(orient="index"),
        }


@dataclass(frozen=True)
class VenueExecution:
    venue: str
    side: Side
    requested_quantity: int
    filled_quantity: int
    average_price: float | None
    explicit_fee: float
    arrival_time: int
    trader_id: str
    liquidity_role: str
    order_id: str | None = None


@dataclass
class _PendingAction:
    due_time: int
    venue: str
    trader_id: str
    action_type: PendingType
    side: Side | None
    quantity: int
    order_id: str
    price_tick: int | None = None
    cancel_order_id: str | None = None


def default_venue_configs() -> tuple[VenueConfig, ...]:
    return (
        VenueConfig(
            name="venue_a",
            taker_fee_bps=0.18,
            maker_rebate_bps=-0.04,
            latency_steps=0,
            initial_mid_offset_ticks=0,
            initial_spread_ticks=2,
            initial_level_quantity=90,
            background_events_per_step=4,
            seed_offset=0,
        ),
        VenueConfig(
            name="venue_b",
            taker_fee_bps=0.05,
            maker_rebate_bps=-0.12,
            latency_steps=1,
            initial_mid_offset_ticks=0,
            initial_spread_ticks=3,
            initial_level_quantity=120,
            background_events_per_step=5,
            seed_offset=101,
        ),
        VenueConfig(
            name="venue_c",
            taker_fee_bps=0.10,
            maker_rebate_bps=-0.06,
            latency_steps=2,
            initial_mid_offset_ticks=1,
            initial_spread_ticks=2,
            initial_level_quantity=70,
            background_events_per_step=3,
            seed_offset=211,
        ),
        VenueConfig(
            name="venue_d",
            taker_fee_bps=-0.02,
            maker_rebate_bps=-0.15,
            latency_steps=3,
            initial_mid_offset_ticks=-1,
            initial_spread_ticks=4,
            initial_level_quantity=140,
            background_events_per_step=6,
            seed_offset=307,
        ),
    )


class MultiVenueMarket:
    """One instrument traded on several independent venue order books."""

    def __init__(
        self,
        venue_configs: tuple[VenueConfig, ...] | None = None,
        *,
        tick_size: float = 0.01,
        initial_mid_tick: int = 10_000,
        seed: int = 2026,
        calibrated_order_flow: bool = False,
    ) -> None:
        self.venue_configs = venue_configs or default_venue_configs()
        self.tick_size = float(tick_size)
        self.initial_mid_tick = int(initial_mid_tick)
        self.seed = int(seed)
        self.calibrated_order_flow = bool(calibrated_order_flow)

        names = [cfg.name for cfg in self.venue_configs]
        if len(names) != len(set(names)):
            raise ValueError("venue names must be unique")

        self.venues: dict[str, LOBEnvironment] = {}
        self._config_by_name = {
            cfg.name: cfg for cfg in self.venue_configs
        }
        self.time = 0
        self.pending: list[_PendingAction] = []
        self.executions: list[VenueExecution] = []
        self._order_counter = 0
        self.reset()

    def _next_order_id(self, venue: str, trader_id: str) -> str:
        self._order_counter += 1
        return f"{trader_id}-{venue}-{self._order_counter:08d}"

    def reset(self) -> MultiVenueSnapshot:
        self.venues = {}
        self.time = 0
        self.pending = []
        self.executions = []
        self._order_counter = 0

        for cfg in self.venue_configs:
            env_cfg = LOBEnvironmentConfig(
                tick_size=self.tick_size,
                initial_mid_tick=(
                    self.initial_mid_tick + cfg.initial_mid_offset_ticks
                ),
                initial_spread_ticks=cfg.initial_spread_ticks,
                initial_levels=4,
                initial_level_quantity=cfg.initial_level_quantity,
                background_events_per_step=cfg.background_events_per_step,
                seed=self.seed + cfg.seed_offset,
            )
            environment_cls = (
                CalibratedLOBEnvironment
                if self.calibrated_order_flow
                else LOBEnvironment
            )
            self.venues[cfg.name] = environment_cls(env_cfg)

        return self.snapshot()

    def snapshot(self) -> MultiVenueSnapshot:
        rows = []
        for cfg in self.venue_configs:
            obs = self.venues[cfg.name].observe()
            micro = {
                "flow_pressure": 0.0,
                "hawkes_pressure": 0.0,
                "toxicity_probability": 0.0,
                "market_buy_intensity": 0.0,
                "market_sell_intensity": 0.0,
            }
            if hasattr(self.venues[cfg.name], "microstructure_state"):
                state = self.venues[cfg.name].microstructure_state()
                micro.update(
                    {
                        key: float(state.loc[key])
                        for key in micro
                    }
                )

            rows.append(
                {
                    "venue": cfg.name,
                    "best_bid_tick": obs.best_bid_tick,
                    "best_ask_tick": obs.best_ask_tick,
                    "mid_tick": obs.mid_tick,
                    "spread_ticks": obs.spread_ticks,
                    "bid_depth": obs.bid_depth,
                    "ask_depth": obs.ask_depth,
                    "imbalance": obs.imbalance,
                    "taker_fee_bps": cfg.taker_fee_bps,
                    "maker_rebate_bps": cfg.maker_rebate_bps,
                    "latency_steps": cfg.latency_steps,
                    **micro,
                }
            )

        frame = pd.DataFrame(rows).set_index("venue")
        nbbo_bid = int(frame["best_bid_tick"].max())
        nbbo_ask = int(frame["best_ask_tick"].min())

        return MultiVenueSnapshot(
            time=self.time,
            venues=frame,
            nbbo_bid_tick=nbbo_bid,
            nbbo_ask_tick=nbbo_ask,
            consolidated_mid_tick=0.5 * (nbbo_bid + nbbo_ask),
        )

    def schedule_market_order(
        self,
        *,
        venue: str,
        trader_id: str,
        side: Side,
        quantity: int,
    ) -> str:
        return self._schedule_order(
            venue=venue,
            trader_id=trader_id,
            action_type="market",
            side=side,
            quantity=quantity,
        )

    def schedule_limit_order(
        self,
        *,
        venue: str,
        trader_id: str,
        side: Side,
        quantity: int,
        price_tick: int,
        order_id: str | None = None,
    ) -> str:
        return self._schedule_order(
            venue=venue,
            trader_id=trader_id,
            action_type="limit",
            side=side,
            quantity=quantity,
            price_tick=price_tick,
            order_id=order_id,
        )

    def schedule_cancel(
        self,
        *,
        venue: str,
        trader_id: str,
        order_id: str,
    ) -> str:
        return self._schedule_order(
            venue=venue,
            trader_id=trader_id,
            action_type="cancel",
            side=None,
            quantity=0,
            cancel_order_id=order_id,
        )

    def _schedule_order(
        self,
        *,
        venue: str,
        trader_id: str,
        action_type: PendingType,
        side: Side | None,
        quantity: int,
        price_tick: int | None = None,
        order_id: str | None = None,
        cancel_order_id: str | None = None,
    ) -> str:
        if venue not in self.venues:
            raise ValueError(f"unknown venue: {venue}")
        if action_type != "cancel" and quantity <= 0:
            raise ValueError("quantity must be positive")
        if action_type == "limit" and price_tick is None:
            raise ValueError("limit order requires price_tick")

        cfg = self._config_by_name[venue]
        scheduled_id = order_id or self._next_order_id(venue, trader_id)
        self.pending.append(
            _PendingAction(
                due_time=self.time + cfg.latency_steps,
                venue=venue,
                trader_id=trader_id,
                action_type=action_type,
                side=side,
                quantity=int(quantity),
                order_id=scheduled_id,
                price_tick=price_tick,
                cancel_order_id=cancel_order_id,
            )
        )
        return scheduled_id

    def _execution_from_trades(
        self,
        *,
        venue: str,
        trader_id: str,
        side: Side,
        order_id: str,
        requested_quantity: int,
        trades: list[Trade] | tuple[Trade, ...],
        liquidity_role: str,
    ) -> VenueExecution | None:
        relevant = [
            trade
            for trade in trades
            if (
                trade.taker_trader_id == trader_id
                if liquidity_role == "taker"
                else trade.maker_trader_id == trader_id
            )
        ]
        if not relevant:
            return None

        filled = int(sum(trade.quantity for trade in relevant))
        notional = sum(
            trade.price_tick * self.tick_size * trade.quantity
            for trade in relevant
        )
        average_price = notional / filled if filled else None
        cfg = self._config_by_name[venue]
        fee_bps = (
            cfg.taker_fee_bps
            if liquidity_role == "taker"
            else cfg.maker_rebate_bps
        )
        explicit_fee = notional * fee_bps / 10_000.0

        execution = VenueExecution(
            venue=venue,
            side=side,
            requested_quantity=requested_quantity,
            filled_quantity=filled,
            average_price=average_price,
            explicit_fee=float(explicit_fee),
            arrival_time=self.time,
            trader_id=trader_id,
            liquidity_role=liquidity_role,
            order_id=order_id,
        )
        self.executions.append(execution)
        return execution

    def _execute_due_actions(self) -> list[VenueExecution]:
        due = [item for item in self.pending if item.due_time <= self.time]
        self.pending = [item for item in self.pending if item.due_time > self.time]
        results: list[VenueExecution] = []

        for item in due:
            env = self.venues[item.venue]

            if item.action_type == "cancel":
                if item.cancel_order_id is not None:
                    env.apply_action(
                        AgentAction(
                            action_type="cancel",
                            trader_id=item.trader_id,
                            order_id=item.cancel_order_id,
                        )
                    )
                continue

            assert item.side is not None
            action = AgentAction(
                action_type=item.action_type,
                trader_id=item.trader_id,
                side=item.side,
                quantity=item.quantity,
                price_tick=item.price_tick,
                order_id=item.order_id,
            )
            trades = env.apply_action(action)
            execution = self._execution_from_trades(
                venue=item.venue,
                trader_id=item.trader_id,
                side=item.side,
                order_id=item.order_id,
                requested_quantity=item.quantity,
                trades=trades,
                liquidity_role="taker",
            )
            if execution is not None:
                results.append(execution)

        return results

    def _record_maker_fills(
        self,
        venue: str,
        trades: tuple[Trade, ...],
    ) -> list[VenueExecution]:
        grouped: dict[tuple[str, str, Side], list[Trade]] = {}

        for trade in trades:
            if trade.maker_trader_id == "background":
                continue
            maker_side: Side = (
                "sell" if trade.taker_side == "buy" else "buy"
            )
            key = (
                trade.maker_trader_id,
                trade.maker_order_id,
                maker_side,
            )
            grouped.setdefault(key, []).append(trade)

        results = []
        for (trader_id, order_id, side), maker_trades in grouped.items():
            execution = self._execution_from_trades(
                venue=venue,
                trader_id=trader_id,
                side=side,
                order_id=order_id,
                requested_quantity=sum(t.quantity for t in maker_trades),
                trades=maker_trades,
                liquidity_role="maker",
            )
            if execution is not None:
                results.append(execution)

        return results

    def step(self) -> tuple[MultiVenueSnapshot, tuple[VenueExecution, ...]]:
        executions = self._execute_due_actions()

        for cfg in self.venue_configs:
            result = self.venues[cfg.name].step(
                (),
                background_events=cfg.background_events_per_step,
            )
            executions.extend(
                self._record_maker_fills(
                    cfg.name,
                    result.trades,
                )
            )

        self.time += 1
        return self.snapshot(), tuple(executions)

    def flush_pending(self, max_steps: int = 20) -> tuple[VenueExecution, ...]:
        all_results = []
        for _ in range(max_steps):
            if not self.pending:
                break
            _snapshot, results = self.step()
            all_results.extend(results)
        return tuple(all_results)

    def pending_quantity(self, trader_id: str) -> int:
        return int(
            sum(
                item.quantity
                for item in self.pending
                if item.trader_id == trader_id
                and item.action_type != "cancel"
            )
        )

    def consolidated_inventory(self, trader_id: str) -> int:
        return int(
            sum(
                env.inventory.get(trader_id, 0)
                for env in self.venues.values()
            )
        )

    def consolidated_cash(self, trader_id: str) -> float:
        cash = sum(
            env.cash.get(trader_id, 0.0)
            for env in self.venues.values()
        )
        fees = sum(
            execution.explicit_fee
            for execution in self.executions
            if execution.trader_id == trader_id
        )
        return float(cash - fees)

    def open_orders(self, trader_id: str) -> pd.DataFrame:
        frames = []
        for venue, env in self.venues.items():
            frame = env.open_orders(trader_id)
            if frame.empty:
                continue
            frame = frame.copy()
            frame.insert(0, "venue", venue)
            frames.append(frame)
        if not frames:
            return pd.DataFrame(
                columns=[
                    "venue",
                    "order_id",
                    "side",
                    "price_tick",
                    "remaining",
                    "queue_ahead",
                ]
            )
        return pd.concat(frames, ignore_index=True)

    def execution_frame(self, trader_id: str | None = None) -> pd.DataFrame:
        rows = []
        for execution in self.executions:
            if trader_id is not None and execution.trader_id != trader_id:
                continue
            rows.append(
                {
                    "venue": execution.venue,
                    "side": execution.side,
                    "requested_quantity": execution.requested_quantity,
                    "filled_quantity": execution.filled_quantity,
                    "average_price": execution.average_price,
                    "explicit_fee": execution.explicit_fee,
                    "arrival_time": execution.arrival_time,
                    "trader_id": execution.trader_id,
                    "liquidity_role": execution.liquidity_role,
                    "order_id": execution.order_id,
                }
            )
        return pd.DataFrame(rows)
