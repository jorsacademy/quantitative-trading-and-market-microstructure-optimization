"""Compatibility wrapper for the market making project."""

from market_making_inventory_control.model import (
    MarketMakingProblem,
    MarketMakingResult,
    default_problem,
    parse_action,
    simulate_policy,
    solve,
    main,
)

__all__ = [
    "MarketMakingProblem",
    "MarketMakingResult",
    "default_problem",
    "parse_action",
    "simulate_policy",
    "solve",
]

if __name__ == "__main__":
    main()
