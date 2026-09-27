import numpy as np

from limit_order_book_simulator.hawkes_flow import (
    HawkesOrderFlow,
)


def test_hawkes_event_excites_conditional_intensity():
    model = HawkesOrderFlow()
    before = model.intensities.copy()

    model.observe("market_buy")
    after = model.intensities

    assert after["market_buy"] > before["market_buy"]
    assert after["limit_bid"] > before["limit_bid"]


def test_hawkes_sampling_is_reproducible():
    a = HawkesOrderFlow()
    b = HawkesOrderFlow()

    seq_a = [a.sample_event() for _ in range(30)]
    seq_b = [b.sample_event() for _ in range(30)]

    assert seq_a == seq_b


def test_hawkes_probabilities_are_valid():
    model = HawkesOrderFlow()

    for _ in range(15):
        model.sample_event()

    probabilities = model.probabilities

    assert np.isclose(probabilities.sum(), 1.0)
    assert (probabilities > 0.0).all()
    assert np.isfinite(model.directional_pressure())
