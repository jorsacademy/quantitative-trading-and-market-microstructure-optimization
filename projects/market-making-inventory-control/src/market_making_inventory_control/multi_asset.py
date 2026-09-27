"""Two-asset market-making dynamic program with correlated inventory risk."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MultiAssetProblem:
    horizon: int = 8
    maximum_inventory: int = 2
    quote_offsets: tuple[int, ...] = (1, 2)
    base_fill_probability: tuple[float, float] = (0.32, 0.28)
    fill_decay: float = 0.70
    tick_value: tuple[float, float] = (1.0, 1.2)
    inventory_risk: float = 0.10
    covariance: tuple[tuple[float, float], tuple[float, float]] = (
        (1.00, 0.55),
        (0.55, 1.30),
    )
    terminal_risk_multiplier: float = 3.0


@dataclass(frozen=True)
class MultiAssetResult:
    value_function: pd.Series
    policy: pd.Series
    initial_value: float


def default_multi_asset_problem() -> MultiAssetProblem:
    return MultiAssetProblem()


def _inventory_penalty(
    q1: int,
    q2: int,
    problem: MultiAssetProblem,
    multiplier: float = 1.0,
) -> float:
    q = np.array([q1, q2], dtype=float)
    covariance = np.asarray(problem.covariance, dtype=float)
    return float(
        multiplier
        * problem.inventory_risk
        * q @ covariance @ q
    )


def _fill_probability(
    asset: int,
    offset: int,
    problem: MultiAssetProblem,
) -> float:
    return float(
        problem.base_fill_probability[asset]
        * np.exp(-problem.fill_decay * (offset - 1))
    )


def solve_multi_asset(
    problem: MultiAssetProblem | None = None,
) -> MultiAssetResult:
    p = problem or default_multi_asset_problem()

    inventories = range(
        -p.maximum_inventory,
        p.maximum_inventory + 1,
    )
    states = [
        (t, q1, q2)
        for t in range(p.horizon + 1)
        for q1 in inventories
        for q2 in inventories
    ]

    index = pd.MultiIndex.from_tuples(
        states,
        names=["time", "inventory_1", "inventory_2"],
    )
    value = pd.Series(0.0, index=index)
    policy = pd.Series("", index=index, dtype=object)

    for q1 in inventories:
        for q2 in inventories:
            value.loc[(p.horizon, q1, q2)] = -_inventory_penalty(
                q1,
                q2,
                p,
                multiplier=p.terminal_risk_multiplier,
            )

    quote_actions = list(
        product(
            p.quote_offsets,
            p.quote_offsets,
            p.quote_offsets,
            p.quote_offsets,
        )
    )

    for t in reversed(range(p.horizon)):
        for q1 in inventories:
            for q2 in inventories:
                best_value = -np.inf
                best_action = None

                for action in quote_actions:
                    b1, a1, b2, a2 = action

                    if q1 >= p.maximum_inventory:
                        b1 = 0
                    if q1 <= -p.maximum_inventory:
                        a1 = 0
                    if q2 >= p.maximum_inventory:
                        b2 = 0
                    if q2 <= -p.maximum_inventory:
                        a2 = 0

                    probs = [
                        _fill_probability(0, b1, p) if b1 else 0.0,
                        _fill_probability(0, a1, p) if a1 else 0.0,
                        _fill_probability(1, b2, p) if b2 else 0.0,
                        _fill_probability(1, a2, p) if a2 else 0.0,
                    ]

                    expected = 0.0

                    for fills in product((0, 1), repeat=4):
                        probability = 1.0
                        for fill, prob in zip(fills, probs):
                            probability *= prob if fill else (1.0 - prob)

                        fb1, fa1, fb2, fa2 = fills
                        nq1 = q1 + fb1 - fa1
                        nq2 = q2 + fb2 - fa2

                        if (
                            abs(nq1) > p.maximum_inventory
                            or abs(nq2) > p.maximum_inventory
                        ):
                            continue

                        revenue = (
                            p.tick_value[0]
                            * (fb1 * b1 + fa1 * a1)
                            + p.tick_value[1]
                            * (fb2 * b2 + fa2 * a2)
                        )
                        risk = _inventory_penalty(
                            nq1,
                            nq2,
                            p,
                        )

                        expected += probability * (
                            revenue
                            - risk
                            + value.loc[(t + 1, nq1, nq2)]
                        )

                    if expected > best_value:
                        best_value = expected
                        best_action = (b1, a1, b2, a2)

                if best_action is None:
                    raise RuntimeError(
                        "multi-asset dealer DP found no action"
                    )

                value.loc[(t, q1, q2)] = best_value
                policy.loc[(t, q1, q2)] = (
                    f"b1={best_action[0]},a1={best_action[1]},"
                    f"b2={best_action[2]},a2={best_action[3]}"
                )

    return MultiAssetResult(
        value_function=value,
        policy=policy,
        initial_value=float(value.loc[(0, 0, 0)]),
    )
