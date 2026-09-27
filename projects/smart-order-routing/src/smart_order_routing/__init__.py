"""Smart order routing project package."""

from .model import (
    RoutingProblem,
    RoutingResult,
    default_problem,
    solve,
)

__all__ = [
    "RoutingProblem",
    "RoutingResult",
    "default_problem",
    "solve",
]

from .live_market import (
    compare_static_dynamic_live_routing,
    implementation_shortfall,
)

__all__ += [
    "compare_static_dynamic_live_routing",
    "implementation_shortfall",
]

from .hybrid_market import compare_lit_vs_hybrid

__all__ += [
    "compare_lit_vs_hybrid",
]

from .toxicity_market import compare_toxicity_aware_routing

__all__ += [
    "compare_toxicity_aware_routing",
]
