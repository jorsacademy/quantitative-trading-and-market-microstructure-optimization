"""Run analytical, multi-asset, and RL market-making experiments."""

from pathlib import Path

from market_making_inventory_control.avellaneda_stoikov import (
    default_problem as default_as_problem,
    simulate_avellaneda_stoikov,
    solve_avellaneda_stoikov,
)
from market_making_inventory_control.multi_asset import (
    default_multi_asset_problem,
    solve_multi_asset,
)
from market_making_inventory_control.rl_benchmark import train_q_learning


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    as_problem = default_as_problem()
    as_result = solve_avellaneda_stoikov(as_problem)
    as_simulation = simulate_avellaneda_stoikov(
        as_result,
        as_problem,
        replications=2_000,
        seed=13,
    )

    multi = solve_multi_asset(
        default_multi_asset_problem()
    )

    rl = train_q_learning(
        episodes=20_000,
        seed=41,
    )

    as_result.quote_surface.to_csv(
        output_dir / "avellaneda_stoikov_quote_surface.csv",
    )
    as_result.reservation_price_shift.to_csv(
        output_dir / "avellaneda_stoikov_inventory_skew.csv",
    )
    as_simulation.to_csv(
        output_dir / "avellaneda_stoikov_simulation.csv",
        index=False,
    )

    multi.value_function.to_csv(
        output_dir / "multi_asset_value_function.csv",
    )
    multi.policy.to_csv(
        output_dir / "multi_asset_policy.csv",
    )

    rl.policy.to_csv(
        output_dir / "q_learning_policy.csv",
    )

    print("AS mean marked PnL:", round(as_simulation["marked_pnl"].mean(), 6))
    print("AS mean abs inventory:", round(as_simulation["terminal_inventory"].abs().mean(), 6))
    print("multi_asset_initial_value:", round(multi.initial_value, 6))
    print("q_learning_exact_value:", round(rl.exact_policy_value, 6))
    print("optimal_dp_value:", round(rl.optimal_dp_value, 6))
    print("q_learning_value_gap:", round(rl.value_gap, 6))
