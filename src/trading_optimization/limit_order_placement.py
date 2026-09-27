"""Compatibility wrapper for the limit order placement project."""

from limit_order_placement.model import (
    ACTIONS,
    IMBALANCE_STATES,
    QUEUE_STATES,
    LimitOrderProblem,
    LimitOrderResult,
    default_problem,
    solve,
    main,
)

__all__ = [
    "ACTIONS",
    "IMBALANCE_STATES",
    "QUEUE_STATES",
    "LimitOrderProblem",
    "LimitOrderResult",
    "default_problem",
    "solve",
]

if __name__ == "__main__":
    main()
