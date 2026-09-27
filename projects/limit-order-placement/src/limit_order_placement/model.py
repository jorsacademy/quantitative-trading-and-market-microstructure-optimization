"""Queue-aware limit-order placement as a finite-horizon MDP.

A trader must execute one buy order before the deadline. The state tracks time,
queue position, and order-book imbalance. Actions trade off price improvement
against fill probability and deadline risk.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LimitOrderProblem:
    horizon: int = 8
    market_order_cost: float = 0.60
    terminal_market_cost: float = 0.85
    best_bid_cost: float = -0.50
    improve_cost: float = 0.00
    base_rest_fill: float = 0.32
    base_improve_fill: float = 0.58
    imbalance_effect: float = 0.10


@dataclass(frozen=True)
class LimitOrderResult:
    policy: pd.Series
    value_function: pd.Series
    initial_state: tuple[int, int, int]
    initial_value: float

    def to_dict(self) -> dict:
        return {
            "initial_state": self.initial_state,
            "initial_value": round(self.initial_value, 6),
            "initial_action": self.policy.loc[self.initial_state],
        }


QUEUE_STATES = (0, 1, 2)  # 0=back, 1=middle, 2=front
IMBALANCE_STATES = (-1, 0, 1)
ACTIONS = ("market", "rest", "improve", "wait")


def default_problem() -> LimitOrderProblem:
    return LimitOrderProblem()


def _imbalance_transition(imbalance: int) -> dict[int, float]:
    if imbalance == -1:
        return {-1: 0.60, 0: 0.30, 1: 0.10}
    if imbalance == 0:
        return {-1: 0.20, 0: 0.60, 1: 0.20}
    return {-1: 0.10, 0: 0.30, 1: 0.60}


def _rest_fill_probability(
    queue: int,
    imbalance: int,
    problem: LimitOrderProblem,
) -> float:
    queue_bonus = (queue - 1) * 0.10
    # Positive imbalance means stronger bid-side demand; for a resting buy
    # order this generally reduces sell-arrival fill probability.
    imbalance_adjustment = -problem.imbalance_effect * imbalance
    return float(
        np.clip(
            problem.base_rest_fill
            + queue_bonus
            + imbalance_adjustment,
            0.02,
            0.95,
        )
    )


def _improve_fill_probability(
    imbalance: int,
    problem: LimitOrderProblem,
) -> float:
    return float(
        np.clip(
            problem.base_improve_fill
            - problem.imbalance_effect * imbalance,
            0.05,
            0.98,
        )
    )


def solve(
    problem: LimitOrderProblem | None = None,
) -> LimitOrderResult:
    p = problem or default_problem()

    if p.horizon <= 0:
        raise ValueError("horizon must be positive")

    states = [
        (t, queue, imbalance)
        for t in range(p.horizon + 1)
        for queue in QUEUE_STATES
        for imbalance in IMBALANCE_STATES
    ]

    value = pd.Series(
        0.0,
        index=pd.MultiIndex.from_tuples(
            states,
            names=["time", "queue", "imbalance"],
        ),
        dtype=float,
    )
    policy = pd.Series(
        "",
        index=value.index,
        dtype=object,
    )

    # At the deadline, any remaining order crosses the spread.
    for queue in QUEUE_STATES:
        for imbalance in IMBALANCE_STATES:
            state = (p.horizon, queue, imbalance)
            value.loc[state] = p.terminal_market_cost
            policy.loc[state] = "market"

    for t in reversed(range(p.horizon)):
        for queue in QUEUE_STATES:
            for imbalance in IMBALANCE_STATES:
                state = (t, queue, imbalance)
                candidates = {}

                # Immediate execution.
                candidates["market"] = p.market_order_cost

                # Keep resting at best bid. If not filled, queue priority
                # improves by one level.
                fill = _rest_fill_probability(
                    queue,
                    imbalance,
                    p,
                )
                next_queue = min(queue + 1, max(QUEUE_STATES))
                continuation = 0.0
                for next_imbalance, prob in _imbalance_transition(
                    imbalance
                ).items():
                    continuation += prob * value.loc[
                        (t + 1, next_queue, next_imbalance)
                    ]
                candidates["rest"] = (
                    fill * p.best_bid_cost
                    + (1.0 - fill) * continuation
                )

                # Improve price inside the spread. Better fill probability but
                # less price improvement; an unfilled repost loses priority.
                fill = _improve_fill_probability(
                    imbalance,
                    p,
                )
                continuation = 0.0
                for next_imbalance, prob in _imbalance_transition(
                    imbalance
                ).items():
                    continuation += prob * value.loc[
                        (t + 1, 0, next_imbalance)
                    ]
                candidates["improve"] = (
                    fill * p.improve_cost
                    + (1.0 - fill) * continuation
                )

                # Cancel / wait one period without execution.
                continuation = 0.0
                for next_imbalance, prob in _imbalance_transition(
                    imbalance
                ).items():
                    continuation += prob * value.loc[
                        (t + 1, queue, next_imbalance)
                    ]
                candidates["wait"] = continuation

                best_action = min(
                    candidates,
                    key=candidates.get,
                )
                value.loc[state] = candidates[best_action]
                policy.loc[state] = best_action

    initial_state = (0, 0, 0)
    return LimitOrderResult(
        policy=policy,
        value_function=value,
        initial_state=initial_state,
        initial_value=float(value.loc[initial_state]),
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
