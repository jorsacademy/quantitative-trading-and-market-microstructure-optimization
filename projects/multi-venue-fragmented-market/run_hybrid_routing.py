"""Run lit-taker-only vs maker/taker/dark hybrid smart routing."""

from pathlib import Path

from smart_order_routing.hybrid_market import compare_lit_vs_hybrid


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison, results = compare_lit_vs_hybrid(
        side="buy",
        quantity=160,
        seed=2026,
    )

    comparison.to_csv(
        output_dir / "lit_vs_hybrid_routing.csv"
    )

    lit = results["lit_taker_only"]
    hybrid = results["hybrid_maker_taker_dark"]

    lit.route_decisions.to_csv(
        output_dir / "lit_taker_route_decisions.csv",
        index=False,
    )
    lit.executions.to_csv(
        output_dir / "lit_taker_executions.csv",
        index=False,
    )

    hybrid.decisions.to_csv(
        output_dir / "hybrid_route_decisions.csv",
        index=False,
    )
    hybrid.lit_executions.to_csv(
        output_dir / "hybrid_lit_executions.csv",
        index=False,
    )
    hybrid.dark_executions.to_csv(
        output_dir / "hybrid_dark_executions.csv",
        index=False,
    )

    print(comparison.round(6).to_string())
