"""Synthetic market-impact calibration for execution optimization.

The module generates reproducible execution observations and fits a nonnegative
linear impact model over engineered microstructure features. The fitted model is
then usable inside the stochastic execution optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import nnls


FEATURES = (
    "participation",
    "participation_sq",
    "volatility_bps",
    "half_spread_bps",
    "imbalance_abs",
)


@dataclass(frozen=True)
class ImpactModel:
    intercept: float
    coefficients: pd.Series
    train_rmse_bps: float

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        x = _feature_matrix(frame)
        values = (
            self.intercept
            + x @ self.coefficients.loc[list(FEATURES)].to_numpy(float)
        )
        return pd.Series(
            values,
            index=frame.index,
            name="predicted_impact_bps",
        )


def generate_synthetic_impact_data(
    observations: int = 1_000,
    seed: int = 17,
) -> pd.DataFrame:
    """Generate deterministic synthetic execution-cost observations."""
    if observations < 20:
        raise ValueError("observations must be at least 20")

    rng = np.random.default_rng(seed)

    participation = rng.uniform(0.01, 0.25, observations)
    volatility_bps = rng.uniform(6.0, 35.0, observations)
    half_spread_bps = rng.uniform(0.20, 1.80, observations)
    imbalance = rng.uniform(-1.0, 1.0, observations)

    # Synthetic structural impact model with heteroskedastic noise.
    impact = (
        0.08
        + 4.0 * participation
        + 11.0 * participation**2
        + 0.020 * volatility_bps
        + 0.60 * half_spread_bps
        + 0.25 * np.abs(imbalance)
    )
    noise = rng.normal(
        loc=0.0,
        scale=0.08 + 0.20 * participation,
        size=observations,
    )

    return pd.DataFrame(
        {
            "participation": participation,
            "volatility_bps": volatility_bps,
            "half_spread_bps": half_spread_bps,
            "imbalance": imbalance,
            "realized_impact_bps": np.maximum(impact + noise, 0.0),
        }
    )


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    required = {
        "participation",
        "volatility_bps",
        "half_spread_bps",
        "imbalance",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing impact-model columns: {sorted(missing)}")

    participation = frame["participation"].to_numpy(float)

    return np.column_stack(
        [
            participation,
            participation**2,
            frame["volatility_bps"].to_numpy(float),
            frame["half_spread_bps"].to_numpy(float),
            frame["imbalance"].abs().to_numpy(float),
        ]
    )


def fit_impact_model(
    observations: pd.DataFrame | None = None,
) -> ImpactModel:
    """Fit a nonnegative impact model using NNLS."""
    data = (
        observations.copy()
        if observations is not None
        else generate_synthetic_impact_data()
    )

    x = _feature_matrix(data)
    y = data["realized_impact_bps"].to_numpy(float)

    # Add intercept and constrain all coefficients to be nonnegative.
    design = np.column_stack([np.ones(len(data)), x])
    beta, _ = nnls(design, y)

    prediction = design @ beta
    rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))

    return ImpactModel(
        intercept=float(beta[0]),
        coefficients=pd.Series(
            beta[1:],
            index=FEATURES,
            name="coefficient",
        ),
        train_rmse_bps=rmse,
    )


def calibration_table(
    model: ImpactModel,
) -> pd.DataFrame:
    rows = [{"feature": "intercept", "coefficient": model.intercept}]
    rows.extend(
        {
            "feature": feature,
            "coefficient": float(value),
        }
        for feature, value in model.coefficients.items()
    )
    return pd.DataFrame(rows)
