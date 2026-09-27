"""Optimal trade execution project package."""

from .model import (
    ExecutionProblem,
    ExecutionResult,
    default_problem,
    objective_components,
    sensitivity,
    solve,
    twap_schedule,
    vwap_schedule,
)

__all__ = [
    "ExecutionProblem",
    "ExecutionResult",
    "default_problem",
    "objective_components",
    "sensitivity",
    "solve",
    "twap_schedule",
    "vwap_schedule",
]

from .impact_model import (
    ImpactModel,
    calibration_table,
    fit_impact_model,
    generate_synthetic_impact_data,
)
from .stochastic_execution import (
    StochasticExecutionProblem,
    StochasticExecutionResult,
    default_stochastic_problem,
    generate_scenarios,
    scenario_costs,
    solve_stochastic,
)

__all__ += [
    "ImpactModel",
    "calibration_table",
    "fit_impact_model",
    "generate_synthetic_impact_data",
    "StochasticExecutionProblem",
    "StochasticExecutionResult",
    "default_stochastic_problem",
    "generate_scenarios",
    "scenario_costs",
    "solve_stochastic",
]
