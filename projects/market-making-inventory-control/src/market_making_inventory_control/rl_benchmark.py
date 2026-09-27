"""Tabular Q-learning benchmark against the exact market-making DP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import (
    MarketMakingProblem,
    _actions_for_inventory,
    _fill_probability,
    default_problem,
    parse_action,
    solve,
)


@dataclass(frozen=True)
class QLearningResult:
    q_values: dict[tuple[int, int], dict[tuple[int, int], float]]
    policy: pd.DataFrame
    exact_policy_value: float
    optimal_dp_value: float
    value_gap: float


def _sample_transition(
    inventory: int,
    action: tuple[int, int],
    problem: MarketMakingProblem,
    rng: np.random.Generator,
) -> tuple[int, float]:
    bid_offset, ask_offset = action
    p_bid = _fill_probability(bid_offset, problem)
    p_ask = _fill_probability(ask_offset, problem)

    bid_fill = int(rng.random() < p_bid)
    ask_fill = int(rng.random() < p_ask)

    next_inventory = inventory + bid_fill - ask_fill

    if abs(next_inventory) > problem.maximum_inventory:
        return inventory, -10.0

    spread_revenue = problem.tick_value * (
        bid_fill * bid_offset
        + ask_fill * ask_offset
    )
    inventory_cost = (
        problem.inventory_penalty
        * float(next_inventory**2)
    )
    reward = spread_revenue - inventory_cost

    return next_inventory, float(reward)


def evaluate_policy_exact(
    policy: pd.DataFrame,
    problem: MarketMakingProblem | None = None,
) -> float:
    p = problem or default_problem()
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

    for inventory in inventories:
        value.loc[p.horizon, inventory] = (
            -p.terminal_inventory_penalty
            * float(inventory**2)
        )

    for t in reversed(range(p.horizon)):
        for inventory in inventories:
            action = parse_action(
                policy.loc[t, inventory]
            )
            bid_offset, ask_offset = action

            p_bid = _fill_probability(bid_offset, p)
            p_ask = _fill_probability(ask_offset, p)

            expected = 0.0
            outcomes = [
                (0, 0, (1.0 - p_bid) * (1.0 - p_ask)),
                (1, 0, p_bid * (1.0 - p_ask)),
                (0, 1, (1.0 - p_bid) * p_ask),
                (1, 1, p_bid * p_ask),
            ]

            for bid_fill, ask_fill, probability in outcomes:
                next_inventory = inventory + bid_fill - ask_fill
                if abs(next_inventory) > p.maximum_inventory:
                    continue

                reward = (
                    p.tick_value
                    * (
                        bid_fill * bid_offset
                        + ask_fill * ask_offset
                    )
                    - p.inventory_penalty
                    * float(next_inventory**2)
                )
                expected += probability * (
                    reward
                    + value.loc[t + 1, next_inventory]
                )

            value.loc[t, inventory] = expected

    return float(value.loc[0, 0])


def train_q_learning(
    problem: MarketMakingProblem | None = None,
    episodes: int = 20_000,
    learning_rate: float = 0.12,
    epsilon_start: float = 0.30,
    epsilon_end: float = 0.02,
    seed: int = 41,
) -> QLearningResult:
    p = problem or default_problem()
    rng = np.random.default_rng(seed)

    states = [
        (t, inventory)
        for t in range(p.horizon)
        for inventory in range(
            -p.maximum_inventory,
            p.maximum_inventory + 1,
        )
    ]

    q_values: dict[
        tuple[int, int],
        dict[tuple[int, int], float],
    ] = {}

    for state in states:
        t, inventory = state
        q_values[state] = {
            action: 0.0
            for action in _actions_for_inventory(inventory, p)
        }

    for episode in range(episodes):
        inventory = 0
        epsilon = (
            epsilon_end
            + (epsilon_start - epsilon_end)
            * max(0.0, 1.0 - episode / episodes)
        )

        for t in range(p.horizon):
            state = (t, inventory)
            actions = list(q_values[state])

            if rng.random() < epsilon:
                action = actions[
                    rng.integers(0, len(actions))
                ]
            else:
                action = max(
                    actions,
                    key=lambda candidate: q_values[state][candidate],
                )

            next_inventory, reward = _sample_transition(
                inventory,
                action,
                p,
                rng,
            )

            if t == p.horizon - 1:
                target = (
                    reward
                    - p.terminal_inventory_penalty
                    * float(next_inventory**2)
                )
            else:
                next_state = (t + 1, next_inventory)
                target = reward + max(
                    q_values[next_state].values()
                )

            q_values[state][action] += (
                learning_rate
                * (target - q_values[state][action])
            )
            inventory = next_inventory

    policy = pd.DataFrame(
        "",
        index=range(p.horizon),
        columns=range(
            -p.maximum_inventory,
            p.maximum_inventory + 1,
        ),
        dtype=object,
    )

    for t in range(p.horizon):
        for inventory in policy.columns:
            state = (t, inventory)
            action = max(
                q_values[state],
                key=lambda candidate: q_values[state][candidate],
            )
            policy.loc[t, inventory] = (
                f"bid={action[0]},ask={action[1]}"
            )

    exact_policy_value = evaluate_policy_exact(policy, p)
    optimal = solve(p).initial_value

    return QLearningResult(
        q_values=q_values,
        policy=policy,
        exact_policy_value=exact_policy_value,
        optimal_dp_value=optimal,
        value_gap=float(optimal - exact_policy_value),
    )
