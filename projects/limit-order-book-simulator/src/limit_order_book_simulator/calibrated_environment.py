"""Calibrated LOB environment with Hawkes and queue-reactive order flow."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .environment import LOBEnvironment, LOBEnvironmentConfig
from .hawkes_flow import EVENT_TYPES, HawkesConfig, HawkesOrderFlow
from .queue_reactive import (
    QueueReactiveModel,
    fit_queue_reactive_model,
)
from .toxicity import (
    ToxicityModel,
    fit_toxicity_model,
    toxicity_features_from_state,
)


class CalibratedLOBEnvironment(LOBEnvironment):
    """LOB environment driven by self-exciting, state-dependent event flow."""

    def __init__(
        self,
        config: LOBEnvironmentConfig | None = None,
        *,
        queue_model: QueueReactiveModel | None = None,
        hawkes_config: HawkesConfig | None = None,
        toxicity_model: ToxicityModel | None = None,
        hawkes_weight: float = 0.60,
        queue_weight: float = 0.40,
    ) -> None:
        super().__init__(config)
        self.queue_model = queue_model or fit_queue_reactive_model()
        self.hawkes = HawkesOrderFlow(
            config=hawkes_config
            or HawkesConfig(seed=self.config.seed + 7_001)
        )
        self.toxicity_model = toxicity_model or fit_toxicity_model()
        self.hawkes_weight = float(hawkes_weight)
        self.queue_weight = float(queue_weight)
        if self.hawkes_weight < 0 or self.queue_weight < 0:
            raise ValueError("flow mixture weights must be nonnegative")
        if self.hawkes_weight + self.queue_weight <= 0:
            raise ValueError("at least one flow mixture weight must be positive")
        self.event_log: list[dict] = []

    def reset(self, seed: int | None = None):
        observation = super().reset(seed=seed)
        if hasattr(self, "hawkes"):
            self.hawkes.reset(
                self.config.seed + 7_001
                if seed is None
                else seed + 7_001
            )
        if hasattr(self, "event_log"):
            self.event_log = []
        return observation

    def _combined_event_probabilities(self) -> pd.Series:
        obs = self.observe()
        queue_prob = self.queue_model.probabilities(
            spread_ticks=obs.spread_ticks,
            bid_depth=obs.bid_depth,
            ask_depth=obs.ask_depth,
            imbalance=obs.imbalance,
        )
        hawkes_prob = self.hawkes.probabilities

        combined = (
            self.hawkes_weight * hawkes_prob
            + self.queue_weight * queue_prob
        )
        combined = combined.clip(lower=0.0)
        combined = combined / combined.sum()
        return combined

    def _sample_quantity(self, event_type: str) -> int:
        mean = max(
            1.0,
            self.queue_model.expected_quantity(event_type),
        )
        return int(max(1, self.rng.poisson(mean)))

    def _choose_background_order_id(
        self,
        *,
        side: str,
    ) -> str | None:
        candidates = []
        for order_id in self._background_ids:
            order = self.book.order(order_id)
            if order is None or order.side != side:
                continue
            candidates.append(order_id)
        if not candidates:
            return None
        return str(
            candidates[int(self.rng.integers(0, len(candidates)))]
        )

    def background_event(self) -> list:
        obs_before = self.observe()
        probabilities = self._combined_event_probabilities()
        event_type = str(
            self.rng.choice(
                EVENT_TYPES,
                p=probabilities.reindex(EVENT_TYPES).to_numpy(float),
            )
        )
        self.hawkes.observe(event_type)

        quantity = self._sample_quantity(event_type)
        trades = []

        if event_type == "market_buy":
            order_id = self._next_id("bg-hawkes-market")
            trades = self.book.submit_market(
                order_id=order_id,
                trader_id="background",
                side="buy",
                quantity=quantity,
                timestamp=self.time,
            )
            self._record_trades(trades)
            self._ensure_two_sided_book()

        elif event_type == "market_sell":
            order_id = self._next_id("bg-hawkes-market")
            trades = self.book.submit_market(
                order_id=order_id,
                trader_id="background",
                side="sell",
                quantity=quantity,
                timestamp=self.time,
            )
            self._record_trades(trades)
            self._ensure_two_sided_book()

        elif event_type in ("limit_bid", "limit_ask"):
            side = "buy" if event_type == "limit_bid" else "sell"
            distance = int(
                self.rng.integers(
                    0,
                    self.config.max_background_distance_ticks + 1,
                )
            )
            if side == "buy":
                price_tick = max(
                    1,
                    obs_before.best_bid_tick - distance,
                )
            else:
                price_tick = obs_before.best_ask_tick + distance

            order_id = self._next_id("bg-hawkes-limit")
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

        elif event_type in ("cancel_bid", "cancel_ask"):
            side = "buy" if event_type == "cancel_bid" else "sell"
            order_id = self._choose_background_order_id(side=side)
            if order_id is not None:
                quantity = self.book.cancel(order_id)

        obs_after = self.observe()
        self.event_log.append(
            {
                "time": self.time,
                "event_type": event_type,
                "quantity": quantity,
                "spread_ticks": obs_before.spread_ticks,
                "bid_depth": obs_before.bid_depth,
                "ask_depth": obs_before.ask_depth,
                "imbalance": obs_before.imbalance,
                "mid_before": obs_before.mid_tick,
                "mid_after": obs_after.mid_tick,
                "hawkes_pressure": self.hawkes.directional_pressure(),
            }
        )
        return trades

    def microstructure_state(self) -> pd.Series:
        obs = self.observe()
        recent = self.event_log[-20:]

        signed_flow = 0.0
        total_market = 0.0
        for row in recent:
            if row["event_type"] == "market_buy":
                signed_flow += row["quantity"]
                total_market += row["quantity"]
            elif row["event_type"] == "market_sell":
                signed_flow -= row["quantity"]
                total_market += row["quantity"]

        flow_pressure = (
            signed_flow / total_market
            if total_market > 0
            else 0.0
        )
        hawkes_pressure = self.hawkes.directional_pressure()

        features = toxicity_features_from_state(
            imbalance=obs.imbalance,
            bid_depth=obs.bid_depth,
            ask_depth=obs.ask_depth,
            spread_ticks=obs.spread_ticks,
            signed_flow_pressure=flow_pressure,
            hawkes_pressure=hawkes_pressure,
        )
        toxicity = float(
            self.toxicity_model.predict_probability(features).iloc[0]
        )

        return pd.Series(
            {
                "flow_pressure": flow_pressure,
                "hawkes_pressure": hawkes_pressure,
                "toxicity_probability": toxicity,
                "market_buy_intensity": float(
                    self.hawkes.intensities.loc["market_buy"]
                ),
                "market_sell_intensity": float(
                    self.hawkes.intensities.loc["market_sell"]
                ),
            }
        )

    def event_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.event_log)
