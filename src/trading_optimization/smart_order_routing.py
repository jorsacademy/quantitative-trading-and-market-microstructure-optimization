"""Compatibility wrapper for the smart order routing project."""

from smart_order_routing.model import (
    RoutingProblem,
    RoutingResult,
    default_problem,
    solve,
    main,
)

__all__ = [
    "RoutingProblem",
    "RoutingResult",
    "default_problem",
    "solve",
]

if __name__ == "__main__":
    main()
