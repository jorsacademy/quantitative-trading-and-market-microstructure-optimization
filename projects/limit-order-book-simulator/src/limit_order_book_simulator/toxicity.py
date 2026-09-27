"""Synthetic adverse-selection / venue-toxicity calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TOXICITY_FEATURES = (
    "imbalance_abs",
    "signed_flow_pressure",
    "spread_ticks",
    "thin_depth",
    "hawkes_pressure_abs",
)


@dataclass(frozen=True)
class ToxicityModel:
    intercept: float
    coefficients: pd.Series
    train_log_loss: float

    def predict_probability(
        self,
        frame: pd.DataFrame,
    ) -> pd.Series:
        x = _feature_matrix(frame)
        beta = self.coefficients.loc[
            list(TOXICITY_FEATURES)
        ].to_numpy(float)
        z = self.intercept + x @ beta
        probability = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return pd.Series(
            probability,
            index=frame.index,
            name="toxicity_probability",
        )


def generate_synthetic_toxicity_data(
    observations: int = 5_000,
    seed: int = 919,
) -> pd.DataFrame:
    if observations < 100:
        raise ValueError("observations must be at least 100")

    rng = np.random.default_rng(seed)

    imbalance_abs = rng.uniform(0.0, 1.0, observations)
    signed_flow_pressure = rng.uniform(-1.0, 1.0, observations)
    spread_ticks = rng.integers(1, 6, observations).astype(float)
    thin_depth = rng.uniform(0.0, 1.0, observations)
    hawkes_pressure_abs = rng.uniform(0.0, 1.0, observations)

    z = (
        -2.25
        + 1.30 * imbalance_abs
        + 1.10 * np.abs(signed_flow_pressure)
        + 0.18 * spread_ticks
        + 0.95 * thin_depth
        + 1.20 * hawkes_pressure_abs
    )
    probability = 1.0 / (1.0 + np.exp(-z))
    adverse = rng.binomial(1, probability, observations)

    return pd.DataFrame(
        {
            "imbalance_abs": imbalance_abs,
            "signed_flow_pressure": signed_flow_pressure,
            "spread_ticks": spread_ticks,
            "thin_depth": thin_depth,
            "hawkes_pressure_abs": hawkes_pressure_abs,
            "adverse_move": adverse,
        }
    )


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    required = set(TOXICITY_FEATURES)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing toxicity features: {sorted(missing)}")

    return np.column_stack(
        [
            frame["imbalance_abs"].to_numpy(float),
            np.abs(
                frame["signed_flow_pressure"].to_numpy(float)
            ),
            frame["spread_ticks"].to_numpy(float),
            frame["thin_depth"].to_numpy(float),
            frame["hawkes_pressure_abs"].to_numpy(float),
        ]
    )


def fit_toxicity_model(
    observations: pd.DataFrame | None = None,
    *,
    l2: float = 0.02,
) -> ToxicityModel:
    data = (
        observations.copy()
        if observations is not None
        else generate_synthetic_toxicity_data()
    )

    x = _feature_matrix(data)
    y = data["adverse_move"].to_numpy(float)
    design = np.column_stack([np.ones(len(data)), x])

    def objective(beta: np.ndarray) -> float:
        z = design @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        eps = 1e-12
        loss = -np.mean(
            y * np.log(p + eps)
            + (1.0 - y) * np.log(1.0 - p + eps)
        )
        penalty = l2 * float(np.sum(beta[1:] ** 2))
        return float(loss + penalty)

    result = minimize(
        objective,
        x0=np.zeros(design.shape[1]),
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(
            f"toxicity calibration failed: {result.message}"
        )

    beta = np.asarray(result.x, dtype=float)
    train_loss = objective(beta)

    return ToxicityModel(
        intercept=float(beta[0]),
        coefficients=pd.Series(
            beta[1:],
            index=TOXICITY_FEATURES,
            name="coefficient",
        ),
        train_log_loss=float(train_loss),
    )


def toxicity_features_from_state(
    *,
    imbalance: float,
    bid_depth: float,
    ask_depth: float,
    spread_ticks: float,
    signed_flow_pressure: float,
    hawkes_pressure: float,
) -> pd.DataFrame:
    total_depth = max(bid_depth + ask_depth, 1e-9)
    thin_depth = float(
        np.clip(
            1.0 - total_depth / 300.0,
            0.0,
            1.0,
        )
    )
    return pd.DataFrame(
        [
            {
                "imbalance_abs": abs(float(imbalance)),
                "signed_flow_pressure": float(signed_flow_pressure),
                "spread_ticks": float(spread_ticks),
                "thin_depth": thin_depth,
                "hawkes_pressure_abs": abs(float(hawkes_pressure)),
            }
        ]
    )
