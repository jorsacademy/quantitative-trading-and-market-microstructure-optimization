"""Run queue-reactive calibration, Hawkes flow, and toxicity diagnostics."""

from pathlib import Path

import pandas as pd

from limit_order_book_simulator.calibrated_environment import (
    CalibratedLOBEnvironment,
)
from limit_order_book_simulator.environment import (
    LOBEnvironmentConfig,
)
from limit_order_book_simulator.hawkes_flow import (
    default_excitation_matrix,
)
from limit_order_book_simulator.queue_reactive import (
    fit_queue_reactive_model,
    generate_synthetic_queue_reactive_log,
)
from limit_order_book_simulator.toxicity import (
    fit_toxicity_model,
    generate_synthetic_toxicity_data,
)


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    queue_data = generate_synthetic_queue_reactive_log(
        observations=8_000,
        seed=271,
    )
    queue_model = fit_queue_reactive_model(queue_data)

    toxicity_data = generate_synthetic_toxicity_data(
        observations=6_000,
        seed=919,
    )
    toxicity_model = fit_toxicity_model(toxicity_data)

    env = CalibratedLOBEnvironment(
        LOBEnvironmentConfig(
            seed=2026,
            background_events_per_step=5,
        ),
        queue_model=queue_model,
        toxicity_model=toxicity_model,
    )

    state_rows = []
    for _ in range(200):
        env.step()
        state_rows.append(env.microstructure_state())

    queue_data.to_csv(
        output_dir / "queue_reactive_training_data.csv",
        index=False,
    )
    queue_model.conditional_probabilities.to_csv(
        output_dir / "queue_reactive_conditional_probabilities.csv",
    )
    queue_model.mean_quantities.to_csv(
        output_dir / "queue_reactive_mean_quantities.csv",
    )

    default_excitation_matrix().to_csv(
        output_dir / "hawkes_excitation_matrix.csv",
    )
    env.event_frame().to_csv(
        output_dir / "hawkes_queue_reactive_event_log.csv",
        index=False,
    )
    pd.DataFrame(state_rows).to_csv(
        output_dir / "microstructure_state_path.csv",
        index=False,
    )

    toxicity_data.to_csv(
        output_dir / "toxicity_training_data.csv",
        index=False,
    )
    toxicity_coefficients = pd.Series(
        {
            "intercept": toxicity_model.intercept,
            **toxicity_model.coefficients.to_dict(),
        },
        name="coefficient",
    )
    toxicity_coefficients.to_csv(
        output_dir / "toxicity_model_coefficients.csv",
    )

    event_log = env.event_frame()
    same_event_rate = float(
        (
            event_log["event_type"]
            == event_log["event_type"].shift(1)
        ).mean()
    )

    print("events:", len(event_log))
    print("same_event_rate:", round(same_event_rate, 6))
    print(
        "final_hawkes_pressure:",
        round(env.microstructure_state()["hawkes_pressure"], 6),
    )
    print(
        "final_toxicity_probability:",
        round(env.microstructure_state()["toxicity_probability"], 6),
    )
    print(
        "toxicity_train_log_loss:",
        round(toxicity_model.train_log_loss, 6),
    )
