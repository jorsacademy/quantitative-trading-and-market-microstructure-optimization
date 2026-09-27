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
