"""Finite-horizon market-making and inventory-control dynamic program.

A dealer chooses bid and ask quote offsets in ticks. Wider quotes earn more per
fill but receive fewer arrivals. Inventory risk penalizes accumulated position,
so the optimal policy naturally skews quotes as inventory moves away from zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from itertools import product

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketMakingProblem:
    horizon: int = 20
    maximum_inventory: int = 4
    quote_offsets: tuple[int, ...] = (0, 1, 2, 3)
    base_fill_probability: float = 0.42
    fill_decay: float = 0.75
    tick_value: float = 1.0
    inventory_penalty: float = 0.08
    terminal_inventory_penalty: float = 0.50


@dataclass(frozen=True)
class MarketMakingResult:
    policy: pd.DataFrame
    value_function: pd.DataFrame
    initial_value: float

    def to_dict(self) -> dict:
        return {
            "initial_value": round(self.initial_value, 6),
            "policy": self.policy.to_dict(),
        }


def default_problem() -> MarketMakingProblem:
    return MarketMakingProblem()


def _fill_probability(
    offset: int,
    problem: MarketMakingProblem,
) -> float:
    if offset == 0:
        return 0.0
    return float(
        problem.base_fill_probability
        * np.exp(-problem.fill_decay * (offset - 1))
    )


def _actions_for_inventory(
    inventory: int,
    problem: MarketMakingProblem,
) -> list[tuple[int, int]]:
    actions = []

    for bid_offset, ask_offset in product(
        problem.quote_offsets,
        problem.quote_offsets,
    ):
        if bid_offset == 0 and ask_offset == 0:
            continue
        if inventory >= problem.maximum_inventory and bid_offset > 0:
            continue
        if inventory <= -problem.maximum_inventory and ask_offset > 0:
            continue
        actions.append((bid_offset, ask_offset))

    return actions


def _action_value(
    t: int,
    inventory: int,
    action: tuple[int, int],
    continuation: pd.DataFrame,
    problem: MarketMakingProblem,
) -> float:
    bid_offset, ask_offset = action

    p_bid = _fill_probability(bid_offset, problem)
    p_ask = _fill_probability(ask_offset, problem)

    outcomes = [
        (0, 0, (1.0 - p_bid) * (1.0 - p_ask)),
        (1, 0, p_bid * (1.0 - p_ask)),
        (0, 1, (1.0 - p_bid) * p_ask),
        (1, 1, p_bid * p_ask),
    ]

    expected = 0.0

    for bid_fill, ask_fill, probability in outcomes:
        next_inventory = (
            inventory + bid_fill - ask_fill
        )

        if abs(next_inventory) > problem.maximum_inventory:
            continue

        spread_revenue = problem.tick_value * (
            bid_fill * bid_offset
            + ask_fill * ask_offset
        )
        inventory_cost = (
            problem.inventory_penalty
            * float(next_inventory**2)
        )

        expected += probability * (
            spread_revenue
            - inventory_cost
            + continuation.loc[t + 1, next_inventory]
        )

    return float(expected)


def solve(
    problem: MarketMakingProblem | None = None,
) -> MarketMakingResult:
    p = problem or default_problem()

    if p.horizon <= 0:
        raise ValueError("horizon must be positive")
    if p.maximum_inventory <= 0:
        raise ValueError("maximum_inventory must be positive")
    if not 0 < p.base_fill_probability < 1:
        raise ValueError("base_fill_probability must be in (0, 1)")
    if p.fill_decay < 0:
        raise ValueError("fill_decay must be nonnegative")

    inventories = list(
        range(
            -p.maximum_inventory,
            p.maximum_inventory + 1,
        )
    )

    value = pd.DataFrame(
        0.0,
        index=range(p.horizon + 1),
        columns=inventories,
    )
    policy = pd.DataFrame(
        "",
        index=range(p.horizon),
        columns=inventories,
        dtype=object,
    )

    for inventory in inventories:
        value.loc[p.horizon, inventory] = (
            -p.terminal_inventory_penalty
            * float(inventory**2)
        )

    for t in reversed(range(p.horizon)):
        for inventory in inventories:
            actions = _actions_for_inventory(
                inventory,
                p,
            )

            best_value = -np.inf
            best_action = None

            for action in actions:
                candidate = _action_value(
                    t,
                    inventory,
                    action,
                    value,
                    p,
                )
                if candidate > best_value:
                    best_value = candidate
                    best_action = action

            if best_action is None:
                raise RuntimeError(
                    "market-making DP found no feasible action"
                )

            value.loc[t, inventory] = best_value
            policy.loc[t, inventory] = (
                f"bid={best_action[0]},ask={best_action[1]}"
            )

    return MarketMakingResult(
        policy=policy,
        value_function=value,
        initial_value=float(value.loc[0, 0]),
    )


def parse_action(action: str) -> tuple[int, int]:
    bid_text, ask_text = action.split(",")
    return (
        int(bid_text.split("=")[1]),
        int(ask_text.split("=")[1]),
    )


def simulate_policy(
    result: MarketMakingResult,
    problem: MarketMakingProblem | None = None,
    replications: int = 2_000,
    seed: int = 7,
) -> pd.DataFrame:
    p = problem or default_problem()
    rng = np.random.default_rng(seed)
    rows = []

    for replication in range(replications):
        inventory = 0
        pnl = 0.0
        fills = 0

        for t in range(p.horizon):
            action = parse_action(
                result.policy.loc[t, inventory]
            )
            bid_offset, ask_offset = action

            p_bid = _fill_probability(bid_offset, p)
            p_ask = _fill_probability(ask_offset, p)

            bid_fill = int(rng.random() < p_bid)
            ask_fill = int(rng.random() < p_ask)

            if (
                inventory >= p.maximum_inventory
                and bid_fill
            ):
                bid_fill = 0
            if (
                inventory <= -p.maximum_inventory
                and ask_fill
            ):
                ask_fill = 0

            inventory += bid_fill - ask_fill
            fills += bid_fill + ask_fill

            pnl += p.tick_value * (
                bid_fill * bid_offset
                + ask_fill * ask_offset
            )
            pnl -= (
                p.inventory_penalty
                * float(inventory**2)
            )

        pnl -= (
            p.terminal_inventory_penalty
            * float(inventory**2)
        )

        rows.append(
            {
                "replication": replication,
                "pnl": pnl,
                "terminal_inventory": inventory,
                "fills": fills,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
