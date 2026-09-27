"""Alpha-aware stochastic execution with learned impact and CVaR.

A sell parent order is scheduled across time. Executing earlier can capture a
short-lived alpha advantage, but aggressive participation increases learned
market impact. Price/alpha scenarios create pathwise execution costs, and an
optional CVaR term penalizes expensive tail scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .impact_model import ImpactModel, fit_impact_model
from .model import ExecutionProblem, default_problem, twap_schedule, vwap_schedule


@dataclass(frozen=True)
class StochasticExecutionProblem:
    base: ExecutionProblem
    half_spread_bps: pd.Series
    imbalance: pd.Series
    alpha0_bps: float = 6.0
    alpha_decay: float = 0.35
    scenarios: int = 24
    alpha_sigma_bps: float = 1.5
    volatility_multiplier_sigma: float = 0.12
    cvar_alpha: float = 0.90
    cvar_weight: float = 0.35
    scenario_seed: int = 29


@dataclass(frozen=True)
class StochasticExecutionResult:
    schedule: pd.Series
    remaining_inventory: pd.Series
    expected_cost: float
    var_cost: float
    cvar_cost: float
    objective_value: float
    weighted_execution_time: float
    twap_expected_cost: float
    vwap_expected_cost: float
    scenario_costs: pd.Series

    def to_dict(self) -> dict:
        return {
            "schedule": self.schedule.round(6).to_dict(),
            "remaining_inventory": self.remaining_inventory.round(6).to_dict(),
            "expected_cost": round(self.expected_cost, 6),
            "var_cost": round(self.var_cost, 6),
            "cvar_cost": round(self.cvar_cost, 6),
            "objective_value": round(self.objective_value, 6),
            "weighted_execution_time": round(self.weighted_execution_time, 6),
            "twap_expected_cost": round(self.twap_expected_cost, 6),
            "vwap_expected_cost": round(self.vwap_expected_cost, 6),
        }


def default_stochastic_problem() -> StochasticExecutionProblem:
    base = default_problem()
    idx = base.market_volume.index

    return StochasticExecutionProblem(
        base=base,
        half_spread_bps=pd.Series(
            [0.65, 0.55, 0.48, 0.44, 0.42, 0.46, 0.52, 0.60],
            index=idx,
            dtype=float,
        ),
        imbalance=pd.Series(
            [0.35, 0.20, 0.05, -0.10, -0.20, -0.05, 0.10, 0.25],
            index=idx,
            dtype=float,
        ),
    )


def generate_scenarios(
    problem: StochasticExecutionProblem,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = problem
    idx = p.base.market_volume.index
    rng = np.random.default_rng(p.scenario_seed)

    base_alpha = np.array(
        [
            p.alpha0_bps * np.exp(-p.alpha_decay * t)
            for t in range(len(idx))
        ],
        dtype=float,
    )

    alpha_rows = []
    vol_rows = []

    for _ in range(p.scenarios):
        alpha_noise = rng.normal(
            0.0,
            p.alpha_sigma_bps,
            len(idx),
        )
        alpha_path = base_alpha + alpha_noise

        vol_multiplier = rng.lognormal(
            mean=-0.5 * p.volatility_multiplier_sigma**2,
            sigma=p.volatility_multiplier_sigma,
            size=len(idx),
        )
        vol_path = (
            p.base.volatility.to_numpy(float)
            * vol_multiplier
        )

        alpha_rows.append(alpha_path)
        vol_rows.append(vol_path)

    scenario_index = pd.Index(
        [f"S{i:02d}" for i in range(1, p.scenarios + 1)],
        name="scenario",
    )

    return (
        pd.DataFrame(alpha_rows, index=scenario_index, columns=idx),
        pd.DataFrame(vol_rows, index=scenario_index, columns=idx),
    )


def _impact_cost(
    schedule: np.ndarray,
    problem: StochasticExecutionProblem,
    impact_model: ImpactModel,
) -> float:
    volume = problem.base.market_volume.to_numpy(float)
    participation = np.divide(
        schedule,
        volume,
        out=np.zeros_like(schedule, dtype=float),
        where=volume > 0,
    )

    frame = pd.DataFrame(
        {
            "participation": participation,
            "volatility_bps": 10_000.0
            * problem.base.volatility.to_numpy(float),
            "half_spread_bps": problem.half_spread_bps.to_numpy(float),
            "imbalance": problem.imbalance.to_numpy(float),
        }
    )

    predicted_bps = impact_model.predict(frame).to_numpy(float)

    # Quantity-weighted bps cost.
    return float(
        np.sum(
            (schedule / problem.base.total_quantity)
            * predicted_bps
        )
    )


def scenario_costs(
    schedule: np.ndarray | pd.Series,
    problem: StochasticExecutionProblem,
    impact_model: ImpactModel,
    alpha_paths: pd.DataFrame | None = None,
    volatility_paths: pd.DataFrame | None = None,
) -> pd.Series:
    q = np.asarray(schedule, dtype=float)
    alpha, vol = (
        generate_scenarios(problem)
        if alpha_paths is None or volatility_paths is None
        else (alpha_paths, volatility_paths)
    )

    remaining = problem.base.total_quantity - np.cumsum(q)
    impact = _impact_cost(q, problem, impact_model)

    costs = []
    for scenario in alpha.index:
        alpha_capture = float(
            np.sum(
                (q / problem.base.total_quantity)
                * alpha.loc[scenario].to_numpy(float)
            )
        )
        inventory_risk = float(
            problem.base.risk_aversion
            * np.sum(
                (
                    10_000.0
                    * vol.loc[scenario].to_numpy(float)
                )
                ** 2
                * (remaining / problem.base.total_quantity) ** 2
            )
        )

        # Lower is better. Alpha captured by earlier trading reduces cost.
        costs.append(
            impact
            + inventory_risk
            - alpha_capture
        )

    return pd.Series(costs, index=alpha.index, name="scenario_cost")


def _weighted_var_cvar(
    costs: pd.Series,
    alpha: float,
) -> tuple[float, float]:
    ordered = np.sort(costs.to_numpy(float))
    n = len(ordered)
    probs = np.full(n, 1.0 / n)

    candidates = np.unique(ordered)
    best_eta = float(candidates[0])
    best_value = float("inf")

    for eta in candidates:
        value = eta + float(
            np.sum(
                probs * np.maximum(ordered - eta, 0.0)
            )
        ) / (1.0 - alpha)
        if value < best_value:
            best_eta = float(eta)
            best_value = float(value)

    return best_eta, best_value


def _objective_from_costs(
    costs: pd.Series,
    problem: StochasticExecutionProblem,
) -> tuple[float, float, float, float]:
    expected = float(costs.mean())
    var, cvar = _weighted_var_cvar(
        costs,
        problem.cvar_alpha,
    )
    objective = expected + problem.cvar_weight * cvar
    return expected, var, cvar, objective


def solve_stochastic(
    problem: StochasticExecutionProblem | None = None,
    impact_model: ImpactModel | None = None,
) -> StochasticExecutionResult:
    """Solve stochastic execution with an explicit CVaR auxiliary formulation."""
    p = problem or default_stochastic_problem()
    model = impact_model or fit_impact_model()

    if not 0.0 < p.cvar_alpha < 1.0:
        raise ValueError("cvar_alpha must be in (0,1)")
    if p.cvar_weight < 0:
        raise ValueError("cvar_weight must be nonnegative")

    base = p.base
    alpha_paths, volatility_paths = generate_scenarios(p)
    n = len(base.market_volume)
    s = p.scenarios

    caps = (
        base.maximum_participation
        * base.market_volume.to_numpy(float)
    )
    initial_q = vwap_schedule(base).to_numpy(float)

    initial_costs = scenario_costs(
        initial_q,
        p,
        model,
        alpha_paths,
        volatility_paths,
    )
    initial_eta = float(
        np.quantile(
            initial_costs.to_numpy(float),
            p.cvar_alpha,
        )
    )
    initial_xi = np.maximum(
        initial_costs.to_numpy(float) - initial_eta,
        0.0,
    )

    # Variables: q[0:n] | eta | xi[0:s]
    x0 = np.concatenate(
        [
            initial_q,
            np.array([initial_eta]),
            initial_xi,
        ]
    )

    def unpack(x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        return (
            x[:n],
            float(x[n]),
            x[n + 1 :],
        )

    def objective(x: np.ndarray) -> float:
        q, eta, xi = unpack(x)
        costs = scenario_costs(
            q,
            p,
            model,
            alpha_paths,
            volatility_paths,
        )
        expected = float(costs.mean())
        cvar_proxy = (
            eta
            + float(xi.mean())
            / (1.0 - p.cvar_alpha)
        )
        return expected + p.cvar_weight * cvar_proxy

    def cvar_constraints(x: np.ndarray) -> np.ndarray:
        q, eta, xi = unpack(x)
        costs = scenario_costs(
            q,
            p,
            model,
            alpha_paths,
            volatility_paths,
        ).to_numpy(float)

        # SLSQP ineq requires >= 0:
        # xi_s >= cost_s(q) - eta.
        return xi + eta - costs

    bounds = (
        [(0.0, float(cap)) for cap in caps]
        + [(None, None)]
        + [(0.0, None)] * s
    )

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {
                "type": "eq",
                "fun": lambda x: float(
                    x[:n].sum() - base.total_quantity
                ),
            },
            {
                "type": "ineq",
                "fun": cvar_constraints,
            },
        ],
        options={
            "ftol": 1e-10,
            "maxiter": 2_000,
            "disp": False,
        },
    )
    if not result.success:
        raise RuntimeError(
            f"stochastic execution failed: {result.message}"
        )

    q, _eta_opt, _xi_opt = unpack(np.asarray(result.x, dtype=float))
    q[np.abs(q) < 1e-8] = 0.0

    schedule = pd.Series(
        q,
        index=base.market_volume.index,
        name="stochastic_schedule",
    )
    remaining = pd.Series(
        base.total_quantity - np.cumsum(q),
        index=base.market_volume.index,
        name="remaining_inventory",
    )

    costs = scenario_costs(
        q,
        p,
        model,
        alpha_paths,
        volatility_paths,
    )
    expected, var, cvar, objective_value = _objective_from_costs(
        costs,
        p,
    )

    twap_costs = scenario_costs(
        twap_schedule(base),
        p,
        model,
        alpha_paths,
        volatility_paths,
    )
    vwap_costs = scenario_costs(
        vwap_schedule(base),
        p,
        model,
        alpha_paths,
        volatility_paths,
    )

    weighted_time = float(
        np.dot(
            np.arange(1, len(q) + 1),
            q,
        )
        / base.total_quantity
    )

    return StochasticExecutionResult(
        schedule=schedule,
        remaining_inventory=remaining,
        expected_cost=expected,
        var_cost=var,
        cvar_cost=cvar,
        objective_value=objective_value,
        weighted_execution_time=weighted_time,
        twap_expected_cost=float(twap_costs.mean()),
        vwap_expected_cost=float(vwap_costs.mean()),
        scenario_costs=costs,
    )
