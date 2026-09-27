"""Multi-period electronic-credit RFQ pricing and dealer hedging.

The model chooses quote tiers for a synthetic corporate-bond RFQ stream while
jointly managing two credit-risk inventory factors with CDX-style hedges.

Expected client acceptance converts quote choices into expected inventory flow.
A mixed-integer program trades spread capture against hedge cost and inventory
risk through time.

This is a research model, not a production OTC pricing system.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class CreditDealerProblem:
    rfqs: pd.DataFrame
    hedge_effect: pd.DataFrame
    hedge_cost: pd.Series
    maximum_hedge_trade: pd.Series
    inventory_limit: pd.Series
    inventory_penalty: pd.Series
    quote_spreads_bps: tuple[float, ...] = (2.0, 4.0, 7.0)
    execution_cost_bps: float = 0.9
    period_gross_notional_limit: float = 32.0
    terminal_inventory_multiplier: float = 2.0


@dataclass(frozen=True)
class CreditDealerResult:
    selected_quotes: pd.DataFrame
    inventory_path: pd.DataFrame
    hedge_trades: pd.DataFrame
    expected_quote_pnl: float
    hedge_cost: float
    inventory_penalty: float
    objective_value: float

    @property
    def risk_adjusted_value(self) -> float:
        return -self.objective_value

    def to_dict(self) -> dict:
        return {
            "expected_quote_pnl": round(self.expected_quote_pnl, 6),
            "hedge_cost": round(self.hedge_cost, 6),
            "inventory_penalty": round(self.inventory_penalty, 6),
            "risk_adjusted_value": round(self.risk_adjusted_value, 6),
            "selected_quotes": self.selected_quotes.to_dict(orient="records"),
            "inventory_path": self.inventory_path.reset_index().to_dict(
                orient="records"
            ),
            "hedge_trades": self.hedge_trades.reset_index().to_dict(
                orient="records"
            ),
        }


def default_problem() -> CreditDealerProblem:
    """Return a deterministic synthetic corporate-credit RFQ stream."""
    rfqs = pd.DataFrame(
        [
            ("R01", 0, "client_sell", 10.0, 0.92, 0.11, 1.00, 0.10),
            ("R02", 0, "client_buy", 8.0, 0.88, 0.28, 0.85, 0.20),
            ("R03", 0, "client_sell", 7.0, 0.83, 0.16, 0.20, 0.95),
            ("R04", 1, "client_buy", 11.0, 0.90, 0.10, 1.05, 0.08),
            ("R05", 1, "client_sell", 9.0, 0.86, 0.26, 0.70, 0.35),
            ("R06", 1, "client_sell", 8.0, 0.80, 0.17, 0.15, 1.10),
            ("R07", 2, "client_buy", 12.0, 0.93, 0.09, 1.10, 0.05),
            ("R08", 2, "client_sell", 10.0, 0.84, 0.27, 0.55, 0.55),
            ("R09", 2, "client_buy", 6.0, 0.82, 0.15, 0.10, 1.15),
            ("R10", 3, "client_sell", 13.0, 0.91, 0.10, 0.95, 0.15),
            ("R11", 3, "client_buy", 9.0, 0.87, 0.25, 0.35, 0.85),
            ("R12", 3, "client_sell", 7.0, 0.78, 0.18, 0.05, 1.20),
        ],
        columns=[
            "rfq",
            "period",
            "direction",
            "notional",
            "base_acceptance",
            "spread_sensitivity",
            "IG",
            "HY",
        ],
    ).set_index("rfq")

    hedge_effect = pd.DataFrame(
        [
            [-1.00, -0.10],
            [-0.08, -1.00],
        ],
        index=["IG", "HY"],
        columns=["CDX_IG", "CDX_HY"],
    )

    return CreditDealerProblem(
        rfqs=rfqs,
        hedge_effect=hedge_effect,
        hedge_cost=pd.Series(
            {"CDX_IG": 0.18, "CDX_HY": 0.28},
            name="cost",
        ),
        maximum_hedge_trade=pd.Series(
            {"CDX_IG": 12.0, "CDX_HY": 10.0},
            name="maximum_trade",
        ),
        inventory_limit=pd.Series(
            {"IG": 14.0, "HY": 12.0},
            name="limit",
        ),
        inventory_penalty=pd.Series(
            {"IG": 0.35, "HY": 0.50},
            name="penalty",
        ),
    )


def acceptance_probability(
    base_acceptance: float,
    spread_sensitivity: float,
    spread_bps: float,
) -> float:
    return float(
        np.clip(
            base_acceptance * np.exp(-spread_sensitivity * spread_bps),
            0.0,
            0.99,
        )
    )


def _validate(problem: CreditDealerProblem) -> None:
    required = {
        "period",
        "direction",
        "notional",
        "base_acceptance",
        "spread_sensitivity",
    }
    if not required.issubset(problem.rfqs.columns):
        raise ValueError("RFQ table is missing required columns")

    factors = list(problem.hedge_effect.index)
    hedges = list(problem.hedge_effect.columns)

    if not set(factors).issubset(problem.rfqs.columns):
        raise ValueError("RFQ table must contain every risk-factor loading")
    if list(problem.hedge_cost.index) != hedges:
        raise ValueError("hedge costs must match hedge-effect columns")
    if list(problem.maximum_hedge_trade.index) != hedges:
        raise ValueError("hedge limits must match hedge-effect columns")
    if list(problem.inventory_limit.index) != factors:
        raise ValueError("inventory limits must match risk factors")
    if list(problem.inventory_penalty.index) != factors:
        raise ValueError("inventory penalties must match risk factors")
    if problem.period_gross_notional_limit <= 0:
        raise ValueError("period gross notional limit must be positive")
    if problem.execution_cost_bps < 0:
        raise ValueError("execution cost must be nonnegative")
    if problem.terminal_inventory_multiplier < 1:
        raise ValueError("terminal inventory multiplier must be at least one")
    if (problem.inventory_limit <= 0).any():
        raise ValueError("inventory limits must be positive")
    if (problem.maximum_hedge_trade < 0).any():
        raise ValueError("hedge limits must be nonnegative")

    directions = set(problem.rfqs["direction"])
    if not directions.issubset({"client_buy", "client_sell"}):
        raise ValueError("direction must be client_buy or client_sell")


def solve(
    problem: CreditDealerProblem | None = None,
    *,
    allow_hedging: bool = True,
) -> CreditDealerResult:
    """Solve the multi-period RFQ pricing and credit-inventory MILP."""
    p = problem or default_problem()
    _validate(p)

    rfqs = list(p.rfqs.index)
    factors = list(p.hedge_effect.index)
    hedges = list(p.hedge_effect.columns)
    periods = sorted({int(v) for v in p.rfqs["period"]})
    tiers = list(p.quote_spreads_bps) + [np.inf]

    x_keys = [
        (rfq, tier_index)
        for rfq in rfqs
        for tier_index in range(len(tiers))
    ]
    x_index = {key: i for i, key in enumerate(x_keys)}

    offset = len(x_index)
    h_keys = [(period, hedge) for period in periods for hedge in hedges]
    h_index = {key: offset + i for i, key in enumerate(h_keys)}

    offset += len(h_index)
    inventory_keys = [
        (period, factor)
        for period in periods
        for factor in factors
    ]
    inventory_index = {
        key: offset + i for i, key in enumerate(inventory_keys)
    }

    offset += len(inventory_index)
    inventory_abs_index = {
        key: offset + i for i, key in enumerate(inventory_keys)
    }

    offset += len(inventory_abs_index)
    hedge_abs_index = {
        key: offset + i for i, key in enumerate(h_keys)
    }

    n_vars = offset + len(hedge_abs_index)
    c = np.zeros(n_vars)

    accepted_notional: dict[tuple[str, int], float] = {}
    expected_pnl: dict[tuple[str, int], float] = {}
    expected_exposure: dict[tuple[str, int, str], float] = {}

    for rfq in rfqs:
        row = p.rfqs.loc[rfq]
        dealer_sign = 1.0 if row["direction"] == "client_sell" else -1.0

        for tier_index, spread in enumerate(tiers):
            key = (rfq, tier_index)

            if np.isinf(spread):
                acceptance = 0.0
                accepted = 0.0
                pnl = 0.0
            else:
                acceptance = acceptance_probability(
                    float(row["base_acceptance"]),
                    float(row["spread_sensitivity"]),
                    float(spread),
                )
                accepted = acceptance * float(row["notional"])
                pnl = accepted * (
                    float(spread) - p.execution_cost_bps
                )

            accepted_notional[key] = accepted
            expected_pnl[key] = pnl
            c[x_index[key]] = -pnl

            for factor in factors:
                expected_exposure[(rfq, tier_index, factor)] = (
                    dealer_sign
                    * accepted
                    * float(row[factor])
                )

    final_period = periods[-1]
    for period in periods:
        for factor in factors:
            multiplier = (
                p.terminal_inventory_multiplier
                if period == final_period
                else 1.0
            )
            c[inventory_abs_index[(period, factor)]] = (
                float(p.inventory_penalty.loc[factor]) * multiplier
            )
        for hedge in hedges:
            c[hedge_abs_index[(period, hedge)]] = float(
                p.hedge_cost.loc[hedge]
            )

    rows: list[np.ndarray] = []
    lower_rows: list[float] = []
    upper_rows: list[float] = []

    # Exactly one quote tier or reject choice per RFQ.
    for rfq in rfqs:
        row = np.zeros(n_vars)
        for tier_index in range(len(tiers)):
            row[x_index[(rfq, tier_index)]] = 1.0
        rows.append(row)
        lower_rows.append(1.0)
        upper_rows.append(1.0)

    # Factor inventory evolves from expected RFQ flow plus hedge trades.
    for period_position, period in enumerate(periods):
        for factor in factors:
            row = np.zeros(n_vars)
            row[inventory_index[(period, factor)]] = 1.0

            if period_position > 0:
                previous = periods[period_position - 1]
                row[inventory_index[(previous, factor)]] = -1.0

            for rfq in rfqs:
                if int(p.rfqs.loc[rfq, "period"]) != period:
                    continue
                for tier_index in range(len(tiers)):
                    row[x_index[(rfq, tier_index)]] -= expected_exposure[
                        (rfq, tier_index, factor)
                    ]

            for hedge in hedges:
                row[h_index[(period, hedge)]] -= float(
                    p.hedge_effect.loc[factor, hedge]
                )

            rows.append(row)
            lower_rows.append(0.0)
            upper_rows.append(0.0)

    # Absolute inventory variables.
    for period, factor in inventory_keys:
        for sign in (1.0, -1.0):
            row = np.zeros(n_vars)
            row[inventory_index[(period, factor)]] = sign
            row[inventory_abs_index[(period, factor)]] = -1.0
            rows.append(row)
            lower_rows.append(-np.inf)
            upper_rows.append(0.0)

    # Absolute hedge-trade variables.
    for period, hedge in h_keys:
        for sign in (1.0, -1.0):
            row = np.zeros(n_vars)
            row[h_index[(period, hedge)]] = sign
            row[hedge_abs_index[(period, hedge)]] = -1.0
            rows.append(row)
            lower_rows.append(-np.inf)
            upper_rows.append(0.0)

    # Desk capacity: expected accepted notional per RFQ wave.
    for period in periods:
        row = np.zeros(n_vars)
        for rfq in rfqs:
            if int(p.rfqs.loc[rfq, "period"]) != period:
                continue
            for tier_index in range(len(tiers)):
                row[x_index[(rfq, tier_index)]] = accepted_notional[
                    (rfq, tier_index)
                ]
        rows.append(row)
        lower_rows.append(-np.inf)
        upper_rows.append(float(p.period_gross_notional_limit))

    lower = np.full(n_vars, -np.inf)
    upper = np.full(n_vars, np.inf)
    integrality = np.zeros(n_vars, dtype=int)

    for index in x_index.values():
        lower[index] = 0.0
        upper[index] = 1.0
        integrality[index] = 1

    for (period, hedge), index in h_index.items():
        limit = (
            float(p.maximum_hedge_trade.loc[hedge])
            if allow_hedging
            else 0.0
        )
        lower[index] = -limit
        upper[index] = limit

    for (period, factor), index in inventory_index.items():
        limit = float(p.inventory_limit.loc[factor])
        lower[index] = -limit
        upper[index] = limit

    for index in inventory_abs_index.values():
        lower[index] = 0.0

    for index in hedge_abs_index.values():
        lower[index] = 0.0

    constraints = LinearConstraint(
        np.vstack(rows),
        lb=np.asarray(lower_rows, dtype=float),
        ub=np.asarray(upper_rows, dtype=float),
    )

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=[constraints],
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(
            f"credit-dealer optimization failed: {result.message}"
        )

    selected_rows: list[dict] = []
    quote_pnl = 0.0

    for rfq in rfqs:
        chosen = max(
            range(len(tiers)),
            key=lambda tier_index: result.x[
                x_index[(rfq, tier_index)]
            ],
        )
        spread = tiers[chosen]
        row = p.rfqs.loc[rfq]

        if np.isinf(spread):
            quote = "reject"
            acceptance = 0.0
        else:
            quote = f"{spread:.1f} bps"
            acceptance = acceptance_probability(
                float(row["base_acceptance"]),
                float(row["spread_sensitivity"]),
                float(spread),
            )

        accepted = accepted_notional[(rfq, chosen)]
        pnl = expected_pnl[(rfq, chosen)]
        quote_pnl += pnl

        selected_rows.append(
            {
                "rfq": rfq,
                "period": int(row["period"]),
                "direction": row["direction"],
                "quote": quote,
                "acceptance_probability": acceptance,
                "expected_accepted_notional": accepted,
                "expected_quote_pnl_bps_notional": pnl,
            }
        )

    inventory_path = pd.DataFrame(
        {
            factor: [
                result.x[inventory_index[(period, factor)]]
                for period in periods
            ]
            for factor in factors
        },
        index=pd.Index(periods, name="period"),
    )

    hedge_trades = pd.DataFrame(
        {
            hedge: [
                result.x[h_index[(period, hedge)]]
                for period in periods
            ]
            for hedge in hedges
        },
        index=pd.Index(periods, name="period"),
    )

    total_hedge_cost = float(
        sum(
            abs(result.x[h_index[(period, hedge)]])
            * float(p.hedge_cost.loc[hedge])
            for period, hedge in h_keys
        )
    )
    total_inventory_penalty = float(
        sum(
            abs(result.x[inventory_index[(period, factor)]])
            * float(p.inventory_penalty.loc[factor])
            * (
                p.terminal_inventory_multiplier
                if period == final_period
                else 1.0
            )
            for period, factor in inventory_keys
        )
    )

    return CreditDealerResult(
        selected_quotes=pd.DataFrame(selected_rows),
        inventory_path=inventory_path,
        hedge_trades=hedge_trades,
        expected_quote_pnl=float(quote_pnl),
        hedge_cost=total_hedge_cost,
        inventory_penalty=total_inventory_penalty,
        objective_value=float(result.fun),
    )


def main() -> None:
    result = solve()
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
