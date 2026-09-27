"""Discrete-time multivariate Hawkes order-flow model.

The model tracks six event streams:
market buy/sell, limit bid/ask, and cancel bid/ask. Conditional intensities
decay toward a baseline and are excited by recent events through a nonnegative
cross-excitation matrix.

This is a compact event-time approximation intended for simulation, not a
continuous-time maximum-likelihood Hawkes calibration engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EVENT_TYPES = (
    "market_buy",
    "market_sell",
    "limit_bid",
    "limit_ask",
    "cancel_bid",
    "cancel_ask",
)


@dataclass(frozen=True)
class HawkesConfig:
    baseline: tuple[float, ...] = (
        0.18, 0.18, 0.26, 0.26, 0.10, 0.10
    )
    decay: float = 0.62
    maximum_intensity: float = 3.0
    seed: int = 314


def default_excitation_matrix() -> pd.DataFrame:
    """Return a stable, interpretable synthetic excitation matrix."""
    matrix = np.array(
        [
            [0.34, 0.05, 0.04, 0.02, 0.01, 0.06],
            [0.05, 0.34, 0.02, 0.04, 0.06, 0.01],
            [0.10, 0.02, 0.24, 0.05, 0.02, 0.08],
            [0.02, 0.10, 0.05, 0.24, 0.08, 0.02],
            [0.03, 0.12, 0.04, 0.02, 0.22, 0.04],
            [0.12, 0.03, 0.02, 0.04, 0.04, 0.22],
        ],
        dtype=float,
    )
    return pd.DataFrame(
        matrix,
        index=EVENT_TYPES,
        columns=EVENT_TYPES,
    )


class HawkesOrderFlow:
    """Event-time conditional-intensity Hawkes simulator."""

    def __init__(
        self,
        config: HawkesConfig | None = None,
        excitation: pd.DataFrame | None = None,
    ) -> None:
        self.config = config or HawkesConfig()
        self.excitation = (
            excitation.copy()
            if excitation is not None
            else default_excitation_matrix()
        )
        self._validate()
        self.rng = np.random.default_rng(self.config.seed)
        self._baseline = np.asarray(
            self.config.baseline,
            dtype=float,
        )
        self._intensity = self._baseline.copy()
        self.history: list[str] = []

    def _validate(self) -> None:
        if len(self.config.baseline) != len(EVENT_TYPES):
            raise ValueError("baseline length must match event types")
        if not 0.0 <= self.config.decay < 1.0:
            raise ValueError("decay must be in [0,1)")
        if self.config.maximum_intensity <= 0:
            raise ValueError("maximum_intensity must be positive")
        if list(self.excitation.index) != list(EVENT_TYPES):
            raise ValueError("excitation rows must match event types")
        if list(self.excitation.columns) != list(EVENT_TYPES):
            raise ValueError("excitation columns must match event types")
        if (self.excitation.to_numpy(float) < 0).any():
            raise ValueError("excitation must be nonnegative")

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(
            self.config.seed if seed is None else seed
        )
        self._intensity = self._baseline.copy()
        self.history = []

    @property
    def intensities(self) -> pd.Series:
        return pd.Series(
            self._intensity.copy(),
            index=EVENT_TYPES,
            name="conditional_intensity",
        )

    @property
    def probabilities(self) -> pd.Series:
        values = np.maximum(self._intensity, 1e-12)
        values = values / values.sum()
        return pd.Series(
            values,
            index=EVENT_TYPES,
            name="event_probability",
        )

    def observe(self, event_type: str) -> None:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type: {event_type}")

        event_index = EVENT_TYPES.index(event_type)
        shock = self.excitation.iloc[:, event_index].to_numpy(float)

        self._intensity = (
            self._baseline
            + self.config.decay
            * (self._intensity - self._baseline)
            + shock
        )
        self._intensity = np.clip(
            self._intensity,
            1e-9,
            self.config.maximum_intensity,
        )
        self.history.append(event_type)

    def sample_event(
        self,
        external_weights: pd.Series | None = None,
    ) -> str:
        probabilities = self.probabilities

        if external_weights is not None:
            weights = external_weights.reindex(
                EVENT_TYPES,
            ).fillna(0.0).to_numpy(float)
            weights = np.maximum(weights, 0.0)
            combined = probabilities.to_numpy(float) * weights
            if combined.sum() > 0:
                probabilities = pd.Series(
                    combined / combined.sum(),
                    index=EVENT_TYPES,
                )

        event = str(
            self.rng.choice(
                EVENT_TYPES,
                p=probabilities.to_numpy(float),
            )
        )
        self.observe(event)
        return event

    def directional_pressure(self) -> float:
        """Positive values indicate buy-side aggressive-flow pressure."""
        buy = self._intensity[EVENT_TYPES.index("market_buy")]
        sell = self._intensity[EVENT_TYPES.index("market_sell")]
        total = buy + sell
        return 0.0 if total <= 0 else float((buy - sell) / total)
