"""Multi-venue smart order routing with integer order blocks.

The model allocates a parent order across venues while balancing explicit fees,
spread, adverse selection, latency, and fill risk.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class RoutingProblem:
    venues: pd.DataFrame
    total_blocks: int = 30
    block_size: int = 1_000
    minimum_expected_fill_ratio: float = 0.94
    maximum_active_venues: int = 3
    failed_fill_penalty_bps: float = 3.0
    latency_cost_per_ms_bps: float = 0.002
    venue_activation_cost_bps: float = 0.15


@dataclass(frozen=True)
class RoutingResult:
    blocks: pd.Series
    expected_filled_blocks: float
    expected_cost_bps_blocks: float
    active_venues: int
    objective_value: float

    def to_dict(self) -> dict:
        return {
            "blocks": self.blocks.astype(int).to_dict(),
            "expected_filled_blocks": round(self.expected_filled_blocks, 6),
            "expected_cost_bps_blocks": round(self.expected_cost_bps_blocks, 6),
            "active_venues": self.active_venues,
            "objective_value": round(self.objective_value, 6),
        }


def default_problem() -> RoutingProblem:
    venues = pd.DataFrame(
        {
            "capacity_blocks": [14, 12, 10, 9, 8],
            "fee_bps": [0.15, -0.05, 0.10, -0.10, 0.20],
            "half_spread_bps": [0.55, 0.60, 0.48, 0.66, 0.42],
            "fill_probability": [0.98, 0.95, 0.92, 0.88, 0.90],
            "latency_ms": [0.35, 0.55, 0.75, 1.10, 0.45],
            "adverse_selection_bps": [0.20, 0.16, 0.12, 0.08, 0.25],
        },
        index=[
            "venue_a",
            "venue_b",
            "venue_c",
            "venue_d",
            "venue_e",
        ],
    )
    return RoutingProblem(venues=venues)


def _unit_costs(problem: RoutingProblem) -> pd.Series:
    v = problem.venues
    failure_cost = (
        (1.0 - v["fill_probability"])
        * problem.failed_fill_penalty_bps
    )
    latency_cost = (
        v["latency_ms"]
        * problem.latency_cost_per_ms_bps
    )
    adverse = (
        v["fill_probability"]
        * v["adverse_selection_bps"]
    )

    return (
        v["fee_bps"]
        + v["half_spread_bps"]
        + failure_cost
        + latency_cost
        + adverse
    )


def solve(
    problem: RoutingProblem | None = None,
) -> RoutingResult:
    p = problem or default_problem()
    venues = list(p.venues.index)
    n = len(venues)

    if p.total_blocks <= 0:
        raise ValueError("total_blocks must be positive")
    if not 0 < p.minimum_expected_fill_ratio <= 1:
        raise ValueError("minimum_expected_fill_ratio must be in (0,1]")
    if p.maximum_active_venues <= 0:
        raise ValueError("maximum_active_venues must be positive")

    capacities = p.venues.loc[venues, "capacity_blocks"].to_numpy(float)
    fill_prob = p.venues.loc[venues, "fill_probability"].to_numpy(float)
    costs = _unit_costs(p).loc[venues].to_numpy(float)

    # x[v] integer blocks | y[v] activation binaries
    n_vars = 2 * n
    c = np.zeros(n_vars)
    c[:n] = costs
    c[n:] = p.venue_activation_cost_bps

    eq = np.zeros((1, n_vars))
    eq[0, :n] = 1.0

    rows = []
    rhs = []

    # x_v <= capacity_v * y_v
    for i in range(n):
        row = np.zeros(n_vars)
        row[i] = 1.0
        row[n + i] = -capacities[i]
        rows.append(row)
        rhs.append(0.0)

    # Expected fills >= target.
    row = np.zeros(n_vars)
    row[:n] = -fill_prob
    rows.append(row)
    rhs.append(
        -p.minimum_expected_fill_ratio * p.total_blocks
    )

    # Number of active venues.
    row = np.zeros(n_vars)
    row[n:] = 1.0
    rows.append(row)
    rhs.append(float(p.maximum_active_venues))

    constraints = [
        LinearConstraint(
            eq,
            lb=np.array([float(p.total_blocks)]),
            ub=np.array([float(p.total_blocks)]),
        ),
        LinearConstraint(
            np.vstack(rows),
            lb=np.full(len(rows), -np.inf),
            ub=np.asarray(rhs, dtype=float),
        ),
    ]

    lower = np.zeros(n_vars)
    upper = np.concatenate(
        [
            capacities,
            np.ones(n),
        ]
    )
    integrality = np.ones(n_vars, dtype=int)

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(
            f"smart order routing failed: {result.message}"
        )

    blocks = pd.Series(
        np.rint(result.x[:n]).astype(int),
        index=venues,
        name="blocks",
    )
    expected_fills = float(
        np.dot(blocks.to_numpy(float), fill_prob)
    )
    routing_cost = float(
        np.dot(blocks.to_numpy(float), costs)
    )
    active = int((blocks > 0).sum())

    return RoutingResult(
        blocks=blocks,
        expected_filled_blocks=expected_fills,
        expected_cost_bps_blocks=routing_cost,
        active_venues=active,
        objective_value=float(result.fun),
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
