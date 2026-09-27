"""Queue-reactive event-probability calibration for synthetic LOB states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hawkes_flow import EVENT_TYPES


@dataclass(frozen=True)
class QueueReactiveModel:
    conditional_probabilities: pd.DataFrame
    global_probabilities: pd.Series
    mean_quantities: pd.Series
    depth_threshold: float = 100.0

    def state_key(
        self,
        *,
        spread_ticks: float,
        bid_depth: float,
        ask_depth: float,
        imbalance: float,
    ) -> tuple[str, str, str]:
        spread_state = "tight" if spread_ticks <= 2 else "wide"
        if imbalance <= -0.20:
            imbalance_state = "sell_heavy"
        elif imbalance >= 0.20:
            imbalance_state = "buy_heavy"
        else:
            imbalance_state = "balanced"

        total_depth = bid_depth + ask_depth
        depth_state = (
            "thin"
            if total_depth < 2.0 * self.depth_threshold
            else "thick"
        )
        return spread_state, imbalance_state, depth_state

    def probabilities(
        self,
        *,
        spread_ticks: float,
        bid_depth: float,
        ask_depth: float,
        imbalance: float,
    ) -> pd.Series:
        key = self.state_key(
            spread_ticks=spread_ticks,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            imbalance=imbalance,
        )

        if key in self.conditional_probabilities.index:
            row = self.conditional_probabilities.loc[key]
            return row.reindex(EVENT_TYPES).fillna(0.0)

        return self.global_probabilities.reindex(
            EVENT_TYPES,
        ).fillna(0.0)

    def expected_quantity(self, event_type: str) -> float:
        return float(self.mean_quantities.loc[event_type])


def generate_synthetic_queue_reactive_log(
    observations: int = 6_000,
    seed: int = 271,
) -> pd.DataFrame:
    """Generate reproducible state/event observations with queue dependence."""
    if observations < 100:
        raise ValueError("observations must be at least 100")

    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(observations):
        spread = int(rng.choice([1, 2, 3, 4], p=[0.15, 0.50, 0.25, 0.10]))
        bid_depth = float(rng.gamma(shape=4.0, scale=24.0))
        ask_depth = float(rng.gamma(shape=4.0, scale=24.0))
        total = max(bid_depth + ask_depth, 1e-9)
        imbalance = (bid_depth - ask_depth) / total

        weights = pd.Series(
            {
                "market_buy": 0.17 * np.exp(0.70 * imbalance),
                "market_sell": 0.17 * np.exp(-0.70 * imbalance),
                "limit_bid": 0.23
                * (1.0 + 0.55 * (bid_depth < 80))
                * (1.0 + 0.18 * (spread >= 3)),
                "limit_ask": 0.23
                * (1.0 + 0.55 * (ask_depth < 80))
                * (1.0 + 0.18 * (spread >= 3)),
                "cancel_bid": 0.10
                * (1.0 + 0.80 * max(-imbalance, 0.0)),
                "cancel_ask": 0.10
                * (1.0 + 0.80 * max(imbalance, 0.0)),
            }
        )
        probabilities = weights / weights.sum()
        event_type = str(
            rng.choice(
                probabilities.index,
                p=probabilities.to_numpy(float),
            )
        )

        base_quantity = {
            "market_buy": 8.0,
            "market_sell": 8.0,
            "limit_bid": 11.0,
            "limit_ask": 11.0,
            "cancel_bid": 9.0,
            "cancel_ask": 9.0,
        }[event_type]
        quantity = int(
            max(
                1,
                rng.poisson(
                    base_quantity
                    * (1.0 + 0.20 * abs(imbalance))
                ),
            )
        )

        rows.append(
            {
                "spread_ticks": spread,
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "imbalance": imbalance,
                "event_type": event_type,
                "quantity": quantity,
            }
        )

    return pd.DataFrame(rows)


def fit_queue_reactive_model(
    observations: pd.DataFrame | None = None,
    *,
    smoothing: float = 1.0,
    depth_threshold: float = 100.0,
) -> QueueReactiveModel:
    data = (
        observations.copy()
        if observations is not None
        else generate_synthetic_queue_reactive_log()
    )

    required = {
        "spread_ticks",
        "bid_depth",
        "ask_depth",
        "imbalance",
        "event_type",
        "quantity",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing queue-reactive columns: {sorted(missing)}")

    def state(row: pd.Series) -> tuple[str, str, str]:
        spread_state = "tight" if row["spread_ticks"] <= 2 else "wide"
        if row["imbalance"] <= -0.20:
            imbalance_state = "sell_heavy"
        elif row["imbalance"] >= 0.20:
            imbalance_state = "buy_heavy"
        else:
            imbalance_state = "balanced"
        depth_state = (
            "thin"
            if row["bid_depth"] + row["ask_depth"]
            < 2.0 * depth_threshold
            else "thick"
        )
        return spread_state, imbalance_state, depth_state

    keys = data.apply(state, axis=1)
    indexed = data.copy()
    indexed["state_key"] = list(keys)

    state_rows = {}
    for key, group in indexed.groupby("state_key"):
        counts = group["event_type"].value_counts().reindex(
            EVENT_TYPES,
            fill_value=0,
        ).astype(float)
        counts += smoothing
        state_rows[key] = counts / counts.sum()

    conditional = pd.DataFrame.from_dict(
        state_rows,
        orient="index",
    ).reindex(columns=EVENT_TYPES)

    global_counts = data["event_type"].value_counts().reindex(
        EVENT_TYPES,
        fill_value=0,
    ).astype(float)
    global_counts += smoothing
    global_probabilities = global_counts / global_counts.sum()

    mean_quantities = (
        data.groupby("event_type")["quantity"]
        .mean()
        .reindex(EVENT_TYPES)
        .fillna(data["quantity"].mean())
    )

    return QueueReactiveModel(
        conditional_probabilities=conditional,
        global_probabilities=global_probabilities,
        mean_quantities=mean_quantities,
        depth_threshold=depth_threshold,
    )
