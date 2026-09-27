"""Compatibility wrapper for the RFQ pricing project."""

from rfq_pricing_dealer_optimization.model import (
    RFQProblem,
    RFQResult,
    default_problem,
    solve,
    main,
)

__all__ = [
    "RFQProblem",
    "RFQResult",
    "default_problem",
    "solve",
]

if __name__ == "__main__":
    main()
