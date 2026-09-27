"""Avellaneda-Stoikov-style analytical quote surface.

The module computes reservation-price inventory skew and optimal spread proxies
across time and inventory states, then simulates the resulting quoting policy
under synthetic order-arrival and mid-price dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ASProblem:
    horizon: int = 30
    dt: float = 1.0 / 30.0
    sigma: float = 0.020
    gamma: float = 0.15
    intensity_decay: float = 1.20
    base_intensity: float = 0.55
    maximum_inventory: int = 6
    initial_mid: float = 100.0


@dataclass(frozen=True)
class ASResult:
    quote_surface: pd.DataFrame
    reservation_price_shift: pd.DataFrame
    half_spread: pd.Series


def default_problem() -> ASProblem:
    return ASProblem()


def solve_avellaneda_stoikov(
    problem: ASProblem | None = None,
) -> ASResult:
    p = problem or default_problem()

    if p.horizon <= 0:
        raise ValueError("horizon must be positive")
    if p.gamma <= 0:
        raise ValueError("gamma must be positive")
    if p.sigma < 0:
        raise ValueError("sigma must be nonnegative")
    if p.intensity_decay <= 0:
        raise ValueError("intensity_decay must be positive")

    inventories = range(
        -p.maximum_inventory,
        p.maximum_inventory + 1,
    )

    rows = []
    shift = pd.DataFrame(
        0.0,
        index=range(p.horizon),
        columns=inventories,
    )
    spreads = pd.Series(
        0.0,
        index=range(p.horizon),
        name="half_spread",
    )

    for t in range(p.horizon):
        tau = (p.horizon - t) * p.dt

        half_spread = (
            (1.0 / p.gamma)
            * np.log(1.0 + p.gamma / p.intensity_decay)
            + 0.5 * p.gamma * (p.sigma**2) * tau
        )
        spreads.loc[t] = half_spread

        for inventory in inventories:
            reservation_shift = (
                -inventory
                * p.gamma
                * (p.sigma**2)
                * tau
            )
            shift.loc[t, inventory] = reservation_shift

            bid_offset = half_spread - reservation_shift
            ask_offset = half_spread + reservation_shift

            rows.append(
                {
                    "time": t,
                    "inventory": inventory,
                    "reservation_shift": reservation_shift,
                    "bid_offset": max(float(bid_offset), 1e-8),
                    "ask_offset": max(float(ask_offset), 1e-8),
                }
            )

    surface = pd.DataFrame(rows).set_index(
        ["time", "inventory"]
    )

    return ASResult(
        quote_surface=surface,
        reservation_price_shift=shift,
        half_spread=spreads,
    )


def simulate_avellaneda_stoikov(
    result: ASResult,
    problem: ASProblem | None = None,
    replications: int = 1_000,
    seed: int = 13,
) -> pd.DataFrame:
    p = problem or default_problem()
    rng = np.random.default_rng(seed)
    rows = []

    for replication in range(replications):
        inventory = 0
        cash = 0.0
        mid = p.initial_mid
        fills = 0

        for t in range(p.horizon):
            state = result.quote_surface.loc[(t, inventory)]
            bid_offset = float(state["bid_offset"])
            ask_offset = float(state["ask_offset"])

            bid_intensity = (
                p.base_intensity
                * np.exp(-p.intensity_decay * bid_offset)
            )
            ask_intensity = (
                p.base_intensity
                * np.exp(-p.intensity_decay * ask_offset)
            )

            p_bid = float(
                1.0 - np.exp(-bid_intensity * p.dt)
            )
            p_ask = float(
                1.0 - np.exp(-ask_intensity * p.dt)
            )

            bid_fill = int(
                inventory < p.maximum_inventory
                and rng.random() < p_bid
            )
            ask_fill = int(
                inventory > -p.maximum_inventory
                and rng.random() < p_ask
            )

            bid_price = mid - bid_offset
            ask_price = mid + ask_offset

            cash -= bid_fill * bid_price
            cash += ask_fill * ask_price
            inventory += bid_fill - ask_fill
            fills += bid_fill + ask_fill

            mid += (
                p.sigma
                * np.sqrt(p.dt)
                * rng.normal()
            )

        marked_pnl = cash + inventory * mid

        rows.append(
            {
                "replication": replication,
                "marked_pnl": marked_pnl,
                "terminal_inventory": inventory,
                "fills": fills,
            }
        )

    return pd.DataFrame(rows)
