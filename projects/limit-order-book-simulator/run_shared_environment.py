"""Run execution, market-making, and limit-placement strategies on one LOB design."""

from pathlib import Path

import numpy as np
import pandas as pd

from limit_order_book_simulator.environment import (
    LOBEnvironment,
    LOBEnvironmentConfig,
)
from limit_order_book_simulator.strategies import (
    run_execution_schedule,
    run_limit_order_policy,
    run_market_maker,
)
from optimal_trade_execution.model import default_problem, solve


def scaled_execution_schedule(total_quantity: int = 120) -> pd.Series:
    baseline = solve(default_problem()).schedule
    weights = baseline / baseline.sum()

    raw = weights.to_numpy(float) * total_quantity
    integer = np.floor(raw).astype(int)

    remainder = total_quantity - int(integer.sum())
    if remainder > 0:
        fractional_order = np.argsort(-(raw - integer))
        for index in fractional_order[:remainder]:
            integer[index] += 1

    return pd.Series(
        integer,
        index=baseline.index,
        dtype=float,
        name="lob_execution_quantity",
    )


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = LOBEnvironmentConfig(
        seed=2026,
        initial_level_quantity=100,
        background_events_per_step=4,
    )

    execution_env = LOBEnvironment(config)
    execution = run_execution_schedule(
        execution_env,
        scaled_execution_schedule(),
        side="sell",
    )

    market_maker_env = LOBEnvironment(config)
    market_maker = run_market_maker(
        market_maker_env,
        steps=40,
        quote_size=8,
        maximum_inventory=6,
    )

    limit_env = LOBEnvironment(config)
    placement = run_limit_order_policy(
        limit_env,
        quantity=20,
    )

    execution.steps.to_csv(
        output_dir / "execution_steps.csv",
        index=False,
    )
    execution.trades.to_csv(
        output_dir / "execution_trades.csv",
        index=False,
    )

    market_maker.steps.to_csv(
        output_dir / "market_maker_steps.csv",
        index=False,
    )
    market_maker.trades.to_csv(
        output_dir / "market_maker_trades.csv",
        index=False,
    )

    placement.steps.to_csv(
        output_dir / "limit_placement_steps.csv",
        index=False,
    )
    placement.trades.to_csv(
        output_dir / "limit_placement_trades.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "strategy": "execution",
                "trades": len(execution.trades),
                "filled_quantity": execution.trades["quantity"].sum(),
                "terminal_inventory": execution.terminal_inventory,
                "terminal_mark_to_market": execution.terminal_mark_to_market,
            },
            {
                "strategy": "market_maker",
                "trades": len(market_maker.trades),
                "filled_quantity": market_maker.trades["quantity"].sum(),
                "terminal_inventory": market_maker.terminal_inventory,
                "terminal_mark_to_market": market_maker.terminal_mark_to_market,
            },
            {
                "strategy": "limit_placement",
                "trades": len(placement.trades),
                "filled_quantity": placement.trades["quantity"].sum(),
                "terminal_inventory": placement.terminal_inventory,
                "terminal_mark_to_market": placement.terminal_mark_to_market,
            },
        ]
    )
    summary.to_csv(
        output_dir / "shared_environment_summary.csv",
        index=False,
    )

    print(summary.round(6).to_string(index=False))
