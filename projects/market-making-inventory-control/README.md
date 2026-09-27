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
