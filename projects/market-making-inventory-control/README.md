# Market Making and Inventory Control

A finite-horizon stochastic-control model for a dealer that continuously chooses bid and ask quote offsets.

The dealer earns spread when orders arrive, but accumulated inventory creates risk.

## State

At time (t):

[
(t, I_t)
]

where (I_t) is dealer inventory.

## Action

The dealer chooses:

[
(delta_t^{bid}, delta_t^{ask})
]

in discrete ticks.

An offset of zero disables that side of the quote.

## Fill model

Fill probability decreases exponentially as the quote moves farther from the market:

[
P(fill mid delta)
=
p_0 e^{-k(delta-1)}
]

for active quotes.

## Reward

A fill earns quoted spread but the resulting inventory is penalized:

[
spread revenue
-
lambda I_{t+1}^2
]

A terminal quadratic inventory penalty discourages ending the horizon with a large position.

## Solution method

The complete finite-horizon policy is solved exactly by backward dynamic programming.

The result is not one quote pair. It is a table:

```text
time × inventory → bid offset / ask offset
```

Positive inventory should generally encourage selling and discourage additional buying; negative inventory creates the opposite pressure.

## Simulation

The optimized policy can be simulated under the same synthetic fill process to inspect:

- PnL distribution;
- terminal inventory;
- fill count.

## Run

```bash
python -m trading_optimization.market_making
```

## Limitations

The model does not include adverse selection, queue priority, stochastic mid-price dynamics, latency, order cancellation cost, multi-level order books, multi-asset inventory, or exchange rebates. It is a compact stochastic-control example rather than a production market-making engine.


## Advanced market-making stack

The project now contains four related market-making/control models:

1. exact finite-horizon single-asset inventory DP;
2. Avellaneda-Stoikov-style analytical quote surface;
3. two-asset dealer DP with correlated inventory risk;
4. tabular Q-learning benchmark evaluated against the exact DP.

### Avellaneda-Stoikov quote surface

Module:

\`\`\`text
market_making_inventory_control.avellaneda_stoikov
\`\`\`

The analytical layer computes an inventory-skewed reservation-price proxy:

\[
r_t
=
S_t
-
q_t\gamma\sigma^2(T-t)
\]

and a time-dependent spread proxy.

A long inventory position shifts reservation value downward, pushing the dealer
to quote more aggressively on the ask and less aggressively on the bid. A
short position produces the opposite skew.

The project exports the full:

\`\`\`text
time × inventory → bid offset / ask offset
\`\`\`

surface and includes synthetic fill/mid-price simulation.

This module is an analytical approximation, not the exact discrete DP.

### Multi-asset dealer inventory control

Module:

\`\`\`text
market_making_inventory_control.multi_asset
\`\`\`

The dealer simultaneously makes markets in two correlated assets.

Inventory risk is quadratic:

\[
q^\top \Sigma q
\]

so positions cannot be managed independently when the covariance term is
nonzero.

The backward DP chooses four quote offsets per state:

\[
(\delta_1^{bid},
\delta_1^{ask},
\delta_2^{bid},
\delta_2^{ask})
\]

and returns a complete:

\`\`\`text
time × inventory_1 × inventory_2 → quote action
\`\`\`

policy.

### Reinforcement-learning benchmark

Module:

\`\`\`text
market_making_inventory_control.rl_benchmark
\`\`\`

A tabular Q-learning agent learns the same single-asset quoting problem from
sampled transitions.

The RL policy is **not** evaluated only by noisy Monte Carlo averages. After
training, its deterministic policy is passed through exact dynamic-programming
policy evaluation.

This gives:

\[
V^{DP,*}(0,0)
-
V^{Q-learning}(0,0)
\]

as a clean value gap against the exact optimum.

The comparison therefore separates:

- optimization error;
- learning error;
- simulation noise.

### Run advanced experiment

\`\`\`bash
python projects/market-making-inventory-control/run_advanced.py
\`\`\`

Generated outputs include:

\`\`\`text
avellaneda_stoikov_quote_surface.csv
avellaneda_stoikov_inventory_skew.csv
avellaneda_stoikov_simulation.csv
multi_asset_value_function.csv
multi_asset_policy.csv
q_learning_policy.csv
\`\`\`

## Project code

\`\`\`text
src/market_making_inventory_control/
├── model.py
├── avellaneda_stoikov.py
├── multi_asset.py
└── rl_benchmark.py
\`\`\`

## Advanced limitations

The analytical quote surface is stylized and the RL environment shares the
same synthetic fill model as the DP benchmark.

The multi-asset model uses a small discrete state/action space. It does not yet
include cross-asset hedging instruments, stochastic covariance regimes, queue
priority, adverse selection, latency, or live order-book state.
