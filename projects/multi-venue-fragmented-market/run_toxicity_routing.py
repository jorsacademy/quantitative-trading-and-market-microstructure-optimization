"""Compare toxicity-blind and toxicity-aware hybrid routing."""

from pathlib import Path

from smart_order_routing.toxicity_market import (
    compare_toxicity_aware_routing,
)


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison, results = compare_toxicity_aware_routing(
        side="buy",
        quantity=140,
        seed=2026,
        toxicity_penalty_ticks=1.25,
    )

    comparison.to_csv(
        output_dir / "toxicity_aware_routing_comparison.csv"
    )

    for name in ("toxicity_blind", "toxicity_aware"):
        result = results[name]
        result.decisions.to_csv(
            output_dir / f"{name}_routing_decisions.csv",
            index=False,
        )
        result.lit_executions.to_csv(
            output_dir / f"{name}_lit_executions.csv",
            index=False,
        )
        result.dark_executions.to_csv(
            output_dir / f"{name}_dark_executions.csv",
            index=False,
        )

    blind_market = results["blind_market"]
    aware_market = results["aware_market"]

    for venue, env in blind_market.venues.items():
        if hasattr(env, "event_frame"):
            env.event_frame().to_csv(
                output_dir / f"blind_{venue}_event_log.csv",
                index=False,
            )

    for venue, env in aware_market.venues.items():
        if hasattr(env, "event_frame"):
            env.event_frame().to_csv(
                output_dir / f"aware_{venue}_event_log.csv",
                index=False,
            )

    print(comparison.round(6).to_string())
