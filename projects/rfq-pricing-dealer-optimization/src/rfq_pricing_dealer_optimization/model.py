"""RFQ quote optimization for a synthetic dealer book.

For each client RFQ, the dealer chooses one quote tier or rejects the request.
Acceptance probability falls as spread widens. The portfolio model constrains
expected gross accepted notional and expected signed inventory.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class RFQProblem:
    rfqs: pd.DataFrame
    quote_spreads_bps: tuple[float, ...] = (2.0, 4.0, 7.0)
    hedge_cost_bps: float = 0.8
    expected_gross_notional_limit: float = 70.0
    expected_net_inventory_limit: float = 18.0


@dataclass(frozen=True)
class RFQResult:
    selected_quotes: pd.DataFrame
    expected_pnl: float
    expected_gross_notional: float
    expected_net_inventory: float
    objective_value: float

    def to_dict(self) -> dict:
        return {
            "expected_pnl": round(self.expected_pnl, 6),
            "expected_gross_notional": round(
                self.expected_gross_notional,
                6,
            ),
            "expected_net_inventory": round(
                self.expected_net_inventory,
                6,
            ),
            "selected_quotes": self.selected_quotes.to_dict(
                orient="records"
            ),
        }


def default_problem() -> RFQProblem:
    rfqs = pd.DataFrame(
        [
            ("RFQ01", "client_buy", 12.0, 0.92, 0.11),
            ("RFQ02", "client_sell", 10.0, 0.88, 0.10),
            ("RFQ03", "client_buy", 8.0, 0.85, 0.14),
            ("RFQ04", "client_sell", 14.0, 0.90, 0.09),
            ("RFQ05", "client_buy", 11.0, 0.82, 0.13),
            ("RFQ06", "client_sell", 9.0, 0.86, 0.12),
            ("RFQ07", "client_buy", 7.0, 0.80, 0.16),
            ("RFQ08", "client_sell", 13.0, 0.91, 0.08),
            ("RFQ09", "client_buy", 9.0, 0.84, 0.15),
            ("RFQ10", "client_sell", 8.0, 0.83, 0.13),
        ],
        columns=[
            "rfq",
            "direction",
            "notional",
            "base_acceptance",
            "spread_sensitivity",
        ],
    ).set_index("rfq")
    return RFQProblem(rfqs=rfqs)


def _acceptance_probability(
    base: float,
    sensitivity: float,
    spread_bps: float,
) -> float:
    return float(
        np.clip(
            base * np.exp(-sensitivity * spread_bps),
            0.0,
            0.99,
        )
    )


def solve(
    problem: RFQProblem | None = None,
) -> RFQResult:
    p = problem or default_problem()
    rfqs = list(p.rfqs.index)
    tiers = list(p.quote_spreads_bps) + [np.inf]

    keys = [
        (rfq, tier_index)
        for rfq in rfqs
        for tier_index in range(len(tiers))
    ]
    index = {key: k for k, key in enumerate(keys)}
    n_vars = len(keys)

    c = np.zeros(n_vars)
    expected_gross = np.zeros(n_vars)
    expected_signed = np.zeros(n_vars)

    for rfq in rfqs:
        row = p.rfqs.loc[rfq]
        sign = (
            1.0
            if row["direction"] == "client_sell"
            else -1.0
        )

        for tier_index, spread in enumerate(tiers):
            k = index[(rfq, tier_index)]

            if np.isinf(spread):
                acceptance = 0.0
                pnl = 0.0
            else:
                acceptance = _acceptance_probability(
                    float(row["base_acceptance"]),
                    float(row["spread_sensitivity"]),
                    float(spread),
                )
                notional = float(row["notional"])
                pnl = (
                    acceptance
                    * notional
                    * (float(spread) - p.hedge_cost_bps)
                )

            c[k] = -pnl
            expected_gross[k] = (
                acceptance * float(row["notional"])
            )
            expected_signed[k] = (
                sign
                * acceptance
                * float(row["notional"])
            )

    # Exactly one quote tier / reject option per RFQ.
    a_eq = np.zeros((len(rfqs), n_vars))
    for i, rfq in enumerate(rfqs):
        for tier_index in range(len(tiers)):
            a_eq[i, index[(rfq, tier_index)]] = 1.0

    rows = []
    rhs = []

    rows.append(expected_gross.copy())
    rhs.append(p.expected_gross_notional_limit)

    rows.append(expected_signed.copy())
    rhs.append(p.expected_net_inventory_limit)

    rows.append(-expected_signed.copy())
    rhs.append(p.expected_net_inventory_limit)

    result = milp(
        c=c,
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(
            np.zeros(n_vars),
            np.ones(n_vars),
        ),
        constraints=[
            LinearConstraint(
                a_eq,
                lb=np.ones(len(rfqs)),
                ub=np.ones(len(rfqs)),
            ),
            LinearConstraint(
                np.vstack(rows),
                lb=np.full(len(rows), -np.inf),
                ub=np.asarray(rhs, dtype=float),
            ),
        ],
        options={"disp": False},
    )

    if not result.success:
        raise RuntimeError(
            f"RFQ pricing optimization failed: {result.message}"
        )

    selected_rows = []
    total_pnl = 0.0
    gross = 0.0
    net = 0.0

    for rfq in rfqs:
        row = p.rfqs.loc[rfq]
        sign = (
            1.0
            if row["direction"] == "client_sell"
            else -1.0
        )

        chosen = next(
            tier_index
            for tier_index in range(len(tiers))
            if result.x[index[(rfq, tier_index)]] > 0.5
        )
        spread = tiers[chosen]

        if np.isinf(spread):
            acceptance = 0.0
            pnl = 0.0
            quote_label = "reject"
        else:
            acceptance = _acceptance_probability(
                float(row["base_acceptance"]),
                float(row["spread_sensitivity"]),
                float(spread),
            )
            pnl = (
                acceptance
                * float(row["notional"])
                * (float(spread) - p.hedge_cost_bps)
            )
            quote_label = f"{spread:.1f} bps"

        accepted_notional = (
            acceptance * float(row["notional"])
        )
        total_pnl += pnl
        gross += accepted_notional
        net += sign * accepted_notional

        selected_rows.append(
            {
                "rfq": rfq,
                "direction": row["direction"],
                "quote": quote_label,
                "acceptance_probability": acceptance,
                "expected_accepted_notional": accepted_notional,
                "expected_pnl_bps_notional": pnl,
            }
        )

    return RFQResult(
        selected_quotes=pd.DataFrame(selected_rows),
        expected_pnl=float(total_pnl),
        expected_gross_notional=float(gross),
        expected_net_inventory=float(net),
        objective_value=float(result.fun),
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
