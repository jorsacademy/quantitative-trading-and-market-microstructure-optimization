"""Compatibility wrapper for the advanced electronic-credit RFQ model."""

from rfq_pricing_dealer_optimization.credit_dealer import (
    CreditDealerProblem,
    CreditDealerResult,
    acceptance_probability,
    default_problem,
    solve,
    main,
)

__all__ = [
    "CreditDealerProblem",
    "CreditDealerResult",
    "acceptance_probability",
    "default_problem",
    "solve",
]

if __name__ == "__main__":
    main()
