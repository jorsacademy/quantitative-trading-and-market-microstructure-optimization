"""Run static vs dynamic smart routing on fragmented venue books."""

from pathlib import Path

from fragmented_market.environment import MultiVenueMarket
from smart_order_routing.live_market import (
    compare_static_dynamic_live_routing,
)


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    preview = MultiVenueMarket(seed=2026).snapshot()
    comparison, results = compare_static_dynamic_live_routing(
        side="buy",
        quantity=180,
        seed=2026,
        maximum_active_venues=3,
    )

    preview.venues.to_csv(
        output_dir / "initial_venue_snapshot.csv"
    )
    comparison.to_csv(
        output_dir / "static_vs_dynamic_routing.csv"
    )

    for name, result in results.items():
        result.route_decisions.to_csv(
            output_dir / f"{name}_route_decisions.csv",
            index=False,
        )
        result.executions.to_csv(
            output_dir / f"{name}_executions.csv",
            index=False,
        )

    print("\nInitial fragmented market")
    print(preview.venues.round(4).to_string())

    print("\nStatic vs dynamic live routing")
    print(comparison.round(6).to_string())
