import numpy as np

from limit_order_book_simulator.calibrated_environment import (
    CalibratedLOBEnvironment,
)
from limit_order_book_simulator.environment import (
    LOBEnvironmentConfig,
)
from fragmented_market.dark_pool import MidpointDarkPool
from fragmented_market.environment import MultiVenueMarket
from fragmented_market.hybrid_router import candidate_table


def test_calibrated_environment_is_reproducible():
    config = LOBEnvironmentConfig(
        seed=55,
        background_events_per_step=3,
    )
    a = CalibratedLOBEnvironment(config)
    b = CalibratedLOBEnvironment(config)

    seq_a = []
    seq_b = []

    for _ in range(12):
        a.step()
        b.step()
        seq_a.append(a.event_log[-1]["event_type"])
        seq_b.append(b.event_log[-1]["event_type"])

    assert seq_a == seq_b


def test_calibrated_environment_exposes_microstructure_state():
    env = CalibratedLOBEnvironment(
        LOBEnvironmentConfig(
            seed=56,
            background_events_per_step=4,
        )
    )

    for _ in range(10):
        env.step()

    state = env.microstructure_state()

    assert -1.0 <= state["flow_pressure"] <= 1.0
    assert -1.0 <= state["hawkes_pressure"] <= 1.0
    assert 0.0 < state["toxicity_probability"] < 1.0
    assert len(env.event_frame()) > 0


def test_fragmented_market_snapshot_contains_calibrated_features():
    market = MultiVenueMarket(
        seed=57,
        calibrated_order_flow=True,
    )

    for _ in range(4):
        market.step()

    snapshot = market.snapshot()

    for column in (
        "flow_pressure",
        "hawkes_pressure",
        "toxicity_probability",
        "market_buy_intensity",
        "market_sell_intensity",
    ):
        assert column in snapshot.venues.columns

    assert (
        (snapshot.venues["toxicity_probability"] > 0.0)
        & (snapshot.venues["toxicity_probability"] < 1.0)
    ).all()


def test_toxicity_penalty_increases_candidate_cost():
    market = MultiVenueMarket(
        seed=58,
        calibrated_order_flow=True,
    )
    for _ in range(5):
        market.step()

    snapshot = market.snapshot()
    dark = MidpointDarkPool()

    blind = candidate_table(
        snapshot,
        dark,
        side="buy",
        toxicity_penalty_ticks=0.0,
    )
    aware = candidate_table(
        snapshot,
        dark,
        side="buy",
        toxicity_penalty_ticks=2.0,
    )

    shared = blind.index.intersection(aware.index)
    delta = (
        aware.loc[shared, "unit_cost"]
        - blind.loc[shared, "unit_cost"]
    )

    assert (delta >= -1e-10).all()
    assert (delta > 1e-8).any()
