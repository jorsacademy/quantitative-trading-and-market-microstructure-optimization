"""Convex optimal execution with market impact and inventory risk.

A parent order is split across discrete time intervals. The model minimizes a
quadratic temporary-impact proxy plus a quadratic inventory-risk penalty while
respecting participation-rate limits.

This is an educational Almgren-Chriss-style model, not a production transaction
cost model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class ExecutionProblem:
    total_quantity: float
    market_volume: pd.Series
    volatility: pd.Series
    temporary_impact: float = 1.0e-5
    risk_aversion: float = 0.020
    maximum_participation: float = 0.18


@dataclass(frozen=True)
class ExecutionResult:
    schedule: pd.Series
    remaining_inventory: pd.Series
    impact_cost: float
    risk_cost: float
    objective_value: float
    twap_objective: float
    vwap_objective: float

    def to_dict(self) -> dict:
        return {
            "schedule": self.schedule.round(6).to_dict(),
            "remaining_inventory": self.remaining_inventory.round(6).to_dict(),
            "impact_cost": round(self.impact_cost, 6),
            "risk_cost": round(self.risk_cost, 6),
            "objective_value": round(self.objective_value, 6),
            "twap_objective": round(self.twap_objective, 6),
            "vwap_objective": round(self.vwap_objective, 6),
        }


def default_problem() -> ExecutionProblem:
    intervals = pd.Index(
        [f"T{i}" for i in range(1, 9)],
        name="interval",
    )
    market_volume = pd.Series(
        [90_000, 120_000, 150_000, 180_000, 200_000, 170_000, 140_000, 110_000],
        index=intervals,
        dtype=float,
        name="market_volume",
    )
    volatility = pd.Series(
        [0.0017, 0.0015, 0.0014, 0.0013, 0.0013, 0.0014, 0.0016, 0.0018],
        index=intervals,
        dtype=float,
        name="volatility",
    )
    return ExecutionProblem(
        total_quantity=100_000.0,
        market_volume=market_volume,
        volatility=volatility,
    )


def _validate(problem: ExecutionProblem) -> None:
    if problem.total_quantity <= 0:
        raise ValueError("total_quantity must be positive")
    if problem.temporary_impact <= 0:
        raise ValueError("temporary_impact must be positive")
    if problem.risk_aversion < 0:
        raise ValueError("risk_aversion must be nonnegative")
    if not 0 < problem.maximum_participation <= 1:
        raise ValueError("maximum_participation must be in (0, 1]")
    if not problem.market_volume.index.equals(problem.volatility.index):
        raise ValueError("market_volume and volatility must share the same index")
    if (problem.market_volume <= 0).any():
        raise ValueError("market_volume must be positive")
    if (problem.volatility < 0).any():
        raise ValueError("volatility must be nonnegative")

    capacity = (
        problem.maximum_participation
        * float(problem.market_volume.sum())
    )
    if capacity + 1e-8 < problem.total_quantity:
        raise ValueError("parent order exceeds aggregate participation capacity")


def _remaining_inventory(
    schedule: np.ndarray,
    total_quantity: float,
) -> np.ndarray:
    return total_quantity - np.cumsum(schedule)


def objective_components(
    schedule: np.ndarray | pd.Series,
    problem: ExecutionProblem,
) -> tuple[float, float, float]:
    q = np.asarray(schedule, dtype=float)
    volume = problem.market_volume.to_numpy(float)
    sigma = problem.volatility.to_numpy(float)

    mean_volume = float(volume.mean())
    liquidity = volume / mean_volume

    impact_cost = float(
        problem.temporary_impact
        * np.sum((q**2) / liquidity)
    )

    remaining = _remaining_inventory(
        q,
        problem.total_quantity,
    )
    risk_cost = float(
        problem.risk_aversion
        * np.sum((sigma**2) * (remaining**2))
    )
    return impact_cost, risk_cost, impact_cost + risk_cost


def twap_schedule(problem: ExecutionProblem) -> pd.Series:
    q = np.full(
        len(problem.market_volume),
        problem.total_quantity / len(problem.market_volume),
        dtype=float,
    )
    caps = (
        problem.maximum_participation
        * problem.market_volume.to_numpy(float)
    )
    if np.any(q > caps + 1e-9):
        raise ValueError("TWAP is infeasible under participation caps")
    return pd.Series(q, index=problem.market_volume.index, name="twap")


def vwap_schedule(problem: ExecutionProblem) -> pd.Series:
    volume = problem.market_volume.to_numpy(float)
    q = problem.total_quantity * volume / volume.sum()
    caps = problem.maximum_participation * volume

    if np.any(q > caps + 1e-9):
        raise ValueError("VWAP is infeasible under participation caps")
    return pd.Series(q, index=problem.market_volume.index, name="vwap")


def solve(
    problem: ExecutionProblem | None = None,
) -> ExecutionResult:
    p = problem or default_problem()
    _validate(p)

    n = len(p.market_volume)
    caps = (
        p.maximum_participation
        * p.market_volume.to_numpy(float)
    )

    initial = vwap_schedule(p).to_numpy(float)

    def objective(q: np.ndarray) -> float:
        return objective_components(q, p)[2]

    result = minimize(
        objective,
        x0=initial,
        method="SLSQP",
        bounds=[(0.0, float(cap)) for cap in caps],
        constraints=[
            {
                "type": "eq",
                "fun": lambda q: float(q.sum() - p.total_quantity),
            }
        ],
        options={
            "ftol": 1e-11,
            "maxiter": 2_000,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"optimal execution failed: {result.message}"
        )

    q = np.asarray(result.x, dtype=float)
    q[np.abs(q) < 1e-9] = 0.0

    schedule = pd.Series(
        q,
        index=p.market_volume.index,
        name="optimal_quantity",
    )
    remaining = pd.Series(
        _remaining_inventory(q, p.total_quantity),
        index=p.market_volume.index,
        name="remaining_inventory",
    )

    impact, risk, total = objective_components(q, p)
    twap_obj = objective_components(
        twap_schedule(p).to_numpy(),
        p,
    )[2]
    vwap_obj = objective_components(
        vwap_schedule(p).to_numpy(),
        p,
    )[2]

    return ExecutionResult(
        schedule=schedule,
        remaining_inventory=remaining,
        impact_cost=impact,
        risk_cost=risk,
        objective_value=total,
        twap_objective=twap_obj,
        vwap_objective=vwap_obj,
    )


def sensitivity(
    problem: ExecutionProblem | None = None,
    risk_aversions: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10),
) -> pd.DataFrame:
    from dataclasses import replace

    base = problem or default_problem()
    rows = []

    for risk_aversion in risk_aversions:
        candidate = replace(
            base,
            risk_aversion=float(risk_aversion),
        )
        result = solve(candidate)

        weighted_time = float(
            np.dot(
                np.arange(1, len(result.schedule) + 1),
                result.schedule.to_numpy(),
            )
            / candidate.total_quantity
        )

        rows.append(
            {
                "risk_aversion": risk_aversion,
                "impact_cost": result.impact_cost,
                "risk_cost": result.risk_cost,
                "objective_value": result.objective_value,
                "quantity_weighted_execution_time": weighted_time,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
