import numpy as np

from smart_order_routing.toxicity_market import (
    compare_toxicity_aware_routing,
)


def test_toxicity_routing_benchmark_has_both_policies():
    table, results = compare_toxicity_aware_routing(
        side="buy",
        quantity=60,
        seed=510,
    )

    assert set(table.index) == {
        "toxicity_blind",
        "toxicity_aware",
    }
    assert "blind_market" in results
    assert "aware_market" in results


def test_toxicity_routing_preserves_parent_order_accounting():
    table, _ = compare_toxicity_aware_routing(
        side="sell",
        quantity=55,
        seed=511,
    )

    for _, row in table.iterrows():
        assert (
            row["filled_quantity"]
            + row["remaining_quantity"]
            == 55
        )
        assert 0.0 <= row["fill_ratio"] <= 1.0


def test_toxicity_routing_metrics_are_finite():
    table, _ = compare_toxicity_aware_routing(
        side="buy",
        quantity=50,
        seed=512,
    )

    for column in (
        "explicit_fees",
        "weighted_selected_toxicity",
        "toxicity_reduction_vs_blind",
    ):
        assert np.isfinite(table[column]).all()

    assert (
        (table["weighted_selected_toxicity"] >= 0.0)
        & (table["weighted_selected_toxicity"] <= 1.0)
    ).all()
