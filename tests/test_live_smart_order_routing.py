import numpy as np

from smart_order_routing.live_market import (
    compare_static_dynamic_live_routing,
    implementation_shortfall,
)


def test_live_routing_comparison_has_both_policies():
    table, results = compare_static_dynamic_live_routing(
        side="buy",
        quantity=120,
        seed=77,
    )

    assert set(table.index) == {
        "static_one_shot",
        "dynamic_rerouting",
    }
    assert set(results) == {
        "static_one_shot",
        "dynamic_rerouting",
    }


def test_live_routing_metrics_are_valid():
    table, _ = compare_static_dynamic_live_routing(
        side="sell",
        quantity=100,
        seed=88,
    )

    assert ((table["fill_ratio"] >= 0.0) & (table["fill_ratio"] <= 1.0)).all()
    assert (table["remaining_quantity"] >= 0).all()
    assert np.isfinite(table["explicit_fees"]).all()


def test_implementation_shortfall_sign_convention():
    _, results = compare_static_dynamic_live_routing(
        side="buy",
        quantity=80,
        seed=99,
    )
    result = results["dynamic_rerouting"]

    if result.average_execution_price is not None:
        reference = result.average_execution_price
        value = implementation_shortfall(
            result,
            side="buy",
            reference_price=reference,
        )
        assert np.isclose(
            value,
            result.explicit_fees,
        )
