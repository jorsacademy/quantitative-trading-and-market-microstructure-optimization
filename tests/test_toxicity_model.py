import numpy as np
import pandas as pd

from limit_order_book_simulator.toxicity import (
    fit_toxicity_model,
    generate_synthetic_toxicity_data,
)


def test_toxicity_model_returns_valid_probabilities():
    model = fit_toxicity_model(
        generate_synthetic_toxicity_data(
            observations=2_500,
            seed=21,
        )
    )

    frame = pd.DataFrame(
        [
            {
                "imbalance_abs": 0.30,
                "signed_flow_pressure": 0.20,
                "spread_ticks": 2.0,
                "thin_depth": 0.25,
                "hawkes_pressure_abs": 0.20,
            }
        ]
    )
    probability = model.predict_probability(frame).iloc[0]

    assert 0.0 < probability < 1.0
    assert np.isfinite(model.train_log_loss)


def test_toxicity_increases_for_more_extreme_state():
    model = fit_toxicity_model()

    frame = pd.DataFrame(
        [
            {
                "imbalance_abs": 0.05,
                "signed_flow_pressure": 0.05,
                "spread_ticks": 1.0,
                "thin_depth": 0.05,
                "hawkes_pressure_abs": 0.05,
            },
            {
                "imbalance_abs": 0.90,
                "signed_flow_pressure": 0.90,
                "spread_ticks": 5.0,
                "thin_depth": 0.95,
                "hawkes_pressure_abs": 0.90,
            },
        ]
    )
    probabilities = model.predict_probability(frame)

    assert probabilities.iloc[1] > probabilities.iloc[0]
