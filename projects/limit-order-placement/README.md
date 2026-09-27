# Limit Order Placement

A finite-horizon Markov decision process for a trader that must complete one buy order before a deadline.

The model explicitly represents the trade-off between:

- crossing the spread immediately;
- resting passively at the best bid;
- improving the price inside the spread;
- waiting for a better state.

## State

\[
(t, q, b)
\]

where:

- \(t\): time remaining;
- \(q\): queue position (back / middle / front);
- \(b\): simplified order-book imbalance state.

## Actions

- market;
- rest;
- improve;
- wait.

Passive fill probability depends on queue priority and imbalance.

If a resting order is not filled, queue priority improves. Repricing inside the spread increases fill probability but loses queue priority if it remains unfilled.

## Objective

Minimize expected execution cost before the deadline.

A terminal crossing cost represents deadline urgency.

## Solution

Backward dynamic programming produces:

~~~text
time × queue position × imbalance → action
~~~

## Run

~~~bash
python -m trading_optimization.limit_order_placement
~~~

## Limitations

The state space is deliberately compact. A production model would require richer queue dynamics, order-book levels, price movement, partial fills, cancellation latency, adverse selection, and exchange-specific priority rules.


## Shared event-driven benchmark

This project can also be evaluated on the repository's common synthetic exchange:

~~~bash
python projects/limit-order-book-simulator/run_shared_environment.py
~~~

The shared environment uses a price-time-priority matching engine, live queue depth, background order flow, cancellations, partial fills, and trader-level inventory/cash accounting.
