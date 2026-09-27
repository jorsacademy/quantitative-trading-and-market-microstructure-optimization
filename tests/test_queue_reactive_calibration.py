import numpy as np

from limit_order_book_simulator.queue_reactive import (
    fit_queue_reactive_model,
    generate_synthetic_queue_reactive_log,
)


def test_queue_reactive_probabilities_sum_to_one():
    model = fit_queue_reactive_model(
        generate_synthetic_queue_reactive_log(
            observations=2_000,
            seed=10,
        )
    )

    probabilities = model.probabilities(
        spread_ticks=2,
        bid_depth=60,
        ask_depth=170,
        imbalance=-0.48,
    )

    assert np.isclose(probabilities.sum(), 1.0)
    assert (probabilities >= 0.0).all()


def test_queue_state_changes_event_distribution():
    model = fit_queue_reactive_model(
        generate_synthetic_queue_reactive_log(
            observations=5_000,
            seed=11,
        )
    )

    buy_heavy = model.probabilities(
        spread_ticks=2,
        bid_depth=180,
        ask_depth=55,
        imbalance=0.53,
    )
    sell_heavy = model.probabilities(
        spread_ticks=2,
        bid_depth=55,
        ask_depth=180,
        imbalance=-0.53,
    )

    assert not np.allclose(
        buy_heavy.to_numpy(),
        sell_heavy.to_numpy(),
    )


def test_calibrated_quantities_are_positive():
    model = fit_queue_reactive_model()

    assert (
        model.mean_quantities > 0
    ).all()
