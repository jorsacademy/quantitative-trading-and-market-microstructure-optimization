"""Run learned-impact and stochastic alpha-aware execution experiments."""

from pathlib import Path

from optimal_trade_execution.impact_model import (
    calibration_table,
    fit_impact_model,
    generate_synthetic_impact_data,
)
from optimal_trade_execution.stochastic_execution import (
    default_stochastic_problem,
    generate_scenarios,
    solve_stochastic,
)


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = generate_synthetic_impact_data(
        observations=1_500,
        seed=17,
    )
    impact_model = fit_impact_model(observations)

    problem = default_stochastic_problem()
    alpha_paths, volatility_paths = generate_scenarios(problem)
    result = solve_stochastic(
        problem,
        impact_model=impact_model,
    )

    observations.to_csv(
        output_dir / "impact_training_data.csv",
        index=False,
    )
    calibration_table(impact_model).to_csv(
        output_dir / "impact_model_coefficients.csv",
        index=False,
    )
    alpha_paths.to_csv(
        output_dir / "execution_alpha_scenarios.csv",
    )
    volatility_paths.to_csv(
        output_dir / "execution_volatility_scenarios.csv",
    )
    result.schedule.to_csv(
        output_dir / "stochastic_execution_schedule.csv",
    )
    result.remaining_inventory.to_csv(
        output_dir / "stochastic_remaining_inventory.csv",
    )
    result.scenario_costs.to_csv(
        output_dir / "stochastic_execution_scenario_costs.csv",
    )

    print("impact_model_rmse_bps:", round(impact_model.train_rmse_bps, 6))
    print("expected_cost:", round(result.expected_cost, 6))
    print("cvar_cost:", round(result.cvar_cost, 6))
    print("weighted_execution_time:", round(result.weighted_execution_time, 6))
    print("twap_expected_cost:", round(result.twap_expected_cost, 6))
    print("vwap_expected_cost:", round(result.vwap_expected_cost, 6))
