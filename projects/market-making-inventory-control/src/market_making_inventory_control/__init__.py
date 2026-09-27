"""Market making and inventory control project package."""

from .model import (
    MarketMakingProblem,
    MarketMakingResult,
    default_problem,
    parse_action,
    simulate_policy,
    solve,
)

__all__ = [
    "MarketMakingProblem",
    "MarketMakingResult",
    "default_problem",
    "parse_action",
    "simulate_policy",
    "solve",
]

from .avellaneda_stoikov import (
    ASProblem,
    ASResult,
    solve_avellaneda_stoikov,
    simulate_avellaneda_stoikov,
)
from .multi_asset import (
    MultiAssetProblem,
    MultiAssetResult,
    default_multi_asset_problem,
    solve_multi_asset,
)
from .rl_benchmark import (
    QLearningResult,
    evaluate_policy_exact,
    train_q_learning,
)

__all__ += [
    "ASProblem",
    "ASResult",
    "solve_avellaneda_stoikov",
    "simulate_avellaneda_stoikov",
    "MultiAssetProblem",
    "MultiAssetResult",
    "default_multi_asset_problem",
    "solve_multi_asset",
    "QLearningResult",
    "evaluate_policy_exact",
    "train_q_learning",
]
