import pandas as pd

from limit_order_book_simulator.environment import (
    AgentAction,
    LOBEnvironment,
    LOBEnvironmentConfig,
)
from limit_order_book_simulator.strategies import (
    run_execution_schedule,
    run_limit_order_policy,
    run_market_maker,
)


def test_environment_is_reproducible_for_same_seed():
    config = LOBEnvironmentConfig(
        seed=77,
        background_events_per_step=3,
    )
    a = LOBEnvironment(config)
    b = LOBEnvironment(config)

    a_rows = []
    b_rows = []

    for _ in range(8):
        a_rows.append(a.step().observation.to_series())
        b_rows.append(b.step().observation.to_series())

    assert pd.DataFrame(a_rows).equals(pd.DataFrame(b_rows))


def test_agent_trade_updates_inventory_and_cash():
    env = LOBEnvironment(
        LOBEnvironmentConfig(
            background_events_per_step=0,
        )
    )
    obs = env.observe()

    result = env.step(
        [
            AgentAction(
                action_type="market",
                trader_id="buyer",
                side="buy",
                quantity=10,
            )
        ],
        background_events=0,
    )

    assert env.inventory["buyer"] == 10
    assert env.cash["buyer"] < 0
    assert len(result.trades) >= 1
    assert env.observe().best_ask_tick >= obs.best_ask_tick


def test_execution_schedule_fills_against_shared_book():
    env = LOBEnvironment(
        LOBEnvironmentConfig(
            background_events_per_step=0,
        )
    )
    schedule = pd.Series([5, 7, 8], dtype=float)

    run = run_execution_schedule(
        env,
        schedule,
        side="sell",
        background_events=0,
    )

    assert run.terminal_inventory == -20
    assert run.trades["quantity"].sum() == 20
    assert len(run.steps) == 3


def test_market_maker_respects_inventory_guardrail():
    env = LOBEnvironment(
        LOBEnvironmentConfig(
            seed=13,
            background_events_per_step=4,
        )
    )

    run = run_market_maker(
        env,
        steps=20,
        quote_size=5,
        maximum_inventory=4,
    )

    assert abs(run.terminal_inventory) <= 4
    assert len(run.steps) == 20


def test_limit_order_policy_completes_order_by_deadline():
    env = LOBEnvironment(
        LOBEnvironmentConfig(
            seed=19,
            background_events_per_step=2,
        )
    )

    run = run_limit_order_policy(
        env,
        quantity=8,
    )

    assert run.terminal_inventory == 8
    assert run.trades["quantity"].sum() == 8
