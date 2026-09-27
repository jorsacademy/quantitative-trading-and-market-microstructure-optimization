from pathlib import Path

import pandas as pd

from trading_optimization.credit_rfq import default_problem, solve


def main() -> None:
    problem = default_problem()
    hedged = solve(problem, allow_hedging=True)
    unhedged = solve(problem, allow_hedging=False)

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    hedged.selected_quotes.to_csv(
        output_dir / "credit_rfq_quotes.csv",
        index=False,
    )
    hedged.inventory_path.to_csv(
        output_dir / "credit_inventory_path.csv"
    )
    hedged.hedge_trades.to_csv(
        output_dir / "credit_hedge_trades.csv"
    )

    pd.DataFrame(
        [
            {
                "policy": "unhedged",
                "expected_quote_pnl": unhedged.expected_quote_pnl,
                "hedge_cost": unhedged.hedge_cost,
                "inventory_penalty": unhedged.inventory_penalty,
                "risk_adjusted_value": unhedged.risk_adjusted_value,
            },
            {
                "policy": "optimized_cdx_hedging",
                "expected_quote_pnl": hedged.expected_quote_pnl,
                "hedge_cost": hedged.hedge_cost,
                "inventory_penalty": hedged.inventory_penalty,
                "risk_adjusted_value": hedged.risk_adjusted_value,
            },
        ]
    ).to_csv(
        output_dir / "hedged_vs_unhedged_credit_dealer.csv",
        index=False,
    )

    print(hedged.to_dict())


if __name__ == "__main__":
    main()
