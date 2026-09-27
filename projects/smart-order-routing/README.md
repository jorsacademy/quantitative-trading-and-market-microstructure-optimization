# Smart Order Routing

A multi-venue execution-allocation MILP for routing a parent order across fragmented liquidity.

The optimizer allocates integer order blocks across venues with different:

- available capacity;
- explicit fee or rebate;
- quoted spread;
- fill probability;
- latency;
- adverse-selection cost.

## Decision variables

For venue \(v\):

- integer order blocks \(x_v\);
- binary venue activation \(y_v\).

## Objective

Minimize an expected execution-cost proxy combining:

- fee/rebate;
- half-spread;
- failed-fill penalty;
- latency cost;
- adverse selection;
- venue activation overhead.

## Constraints

- all parent-order blocks must be allocated;
- venue capacity;
- expected fill target;
- maximum number of active venues;
- allocation/activation linking.

## Run

~~~bash
python -m trading_optimization.smart_order_routing
~~~

## Extensions

Natural extensions include dynamic rerouting after partial fills, queue position, dark-pool conditional orders, maker/taker choice, venue toxicity, correlated fill uncertainty, and contextual-bandit routing policies.

## Limitations

This model is a synthetic routing allocator, not an exchange/broker smart-order router. It does not connect to venues or represent real execution rules.


## Live fragmented-market extension

The project now also connects to the repository's multi-venue event-driven market simulator.

The live extension replaces fixed venue attributes with observations from independent venue books:

- executable best price;
- top-of-book depth;
- current imbalance;
- fee/rebate schedule;
- venue latency.

The dynamic router re-solves after realized fills and book changes.

### Benchmark

~~~bash
python projects/multi-venue-fragmented-market/run_fragmented_market.py
~~~

This compares:

~~~text
initial one-shot routing
vs.
dynamic latency-aware rerouting
~~~

on matched synthetic market seeds.

The original MILP remains the transparent static baseline; the live environment is the microstructure execution layer.


## Maker/taker and dark-pool extension

The live router now supports three execution modes:

- lit taker;
- lit maker;
- midpoint dark pool.

The maker path uses live queue-ahead, imbalance, venue latency, patience, and maker rebates.

The dark path uses midpoint execution with stochastic hidden liquidity and time-in-force.

A hybrid MILP allocates each routing wave across venue × mode candidates and re-routes residual quantity after realized fills.

Run:

~~~bash
python projects/multi-venue-fragmented-market/run_hybrid_routing.py
~~~

This benchmark compares dynamic lit-taker-only routing against the hybrid maker/taker/dark policy on matched market seeds.
