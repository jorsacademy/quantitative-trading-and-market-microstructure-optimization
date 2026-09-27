import numpy as np

from smart_order_routing.hybrid_market import (
    compare_lit_vs_hybrid,
)


def test_hybrid_benchmark_contains_both_policies():
    table, results = compare_lit_vs_hybrid(
        side="buy",
        quantity=80,
        seed=404,
    )

    assert set(table.index) == {
        "lit_taker_only",
        "hybrid_maker_taker_dark",
    }
    assert set(results) == {
        "lit_taker_only",
        "hybrid_maker_taker_dark",
    }


def test_hybrid_fill_attribution_matches_total():
    table, _ = compare_lit_vs_hybrid(
        side="sell",
        quantity=70,
        seed=405,
    )

    row = table.loc["hybrid_maker_taker_dark"]
    attributed = (
        row["maker_fill"]
        + row["taker_fill"]
        + row["dark_fill"]
    )

    assert attributed == row["filled_quantity"]


def test_hybrid_benchmark_metrics_are_finite():
    table, _ = compare_lit_vs_hybrid(
        side="buy",
        quantity=60,
        seed=406,
    )

    assert ((table["fill_ratio"] >= 0) & (table["fill_ratio"] <= 1)).all()
    assert np.isfinite(table["explicit_fees"]).all()
    assert np.isfinite(table["implementation_shortfall"]).all()
    assert np.isfinite(table["shortfall_difference_vs_lit"]).all()
