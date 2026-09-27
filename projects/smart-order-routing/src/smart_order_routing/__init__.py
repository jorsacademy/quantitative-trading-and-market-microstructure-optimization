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
