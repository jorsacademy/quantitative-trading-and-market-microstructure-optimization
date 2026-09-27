# Multi-Venue Fragmented Market Simulator

A synthetic fragmented-market environment in which the same instrument trades on several independent venue order books.

Each venue has its own:

- limit order book;
- bid/ask depth;
- fee or rebate schedule;
- latency;
- spread;
- background event process;
- queue state.

The project connects the repository's Smart Order Routing model to live event-driven venue books.

## Market architecture

~~~text
Venue A LOB ─┐
Venue B LOB ─┤
Venue C LOB ─┼→ consolidated market snapshot → smart router
Venue D LOB ─┘
~~~

The consolidated snapshot reports:

- per-venue best bid/ask;
- top-of-book depth;
- spread;
- imbalance;
- taker fee;
- maker rebate;
- latency;
- NBBO;
- consolidated mid.

## Latency

A child order is not necessarily executed when the router sends it.

~~~text
decision at time t
      ↓
venue latency
      ↓
arrival at venue later
~~~

The venue book can change before the order arrives.

## Live smart routing

The dynamic router repeatedly:

1. observes every venue book;
2. calculates live venue execution cost;
3. solves an integer allocation MILP;
4. sends child market orders;
5. waits for latency and venue events;
6. observes actual fills;
7. re-routes only unfilled and uncommitted residual quantity.

The live per-unit cost includes current executable price, taker fee, latency penalty, and an imbalance/toxicity proxy.

## Pending-order accounting

Residual quantity is separated into:

~~~text
parent quantity
=
already filled
+ in-flight child orders
+ currently allocatable residual
~~~

This avoids double-routing quantity while delayed child orders are still traveling to venues.

## Static vs dynamic benchmark

Two routing policies are compared on matched seeds.

Static one-shot observes the initial fragmented market once, allocates child orders, and never re-optimizes.

Dynamic rerouting repeatedly re-solves after fills, latency, and book changes.

The benchmark reports fill ratio, residual quantity, average execution price, explicit venue fees, implementation shortfall, routing decisions, and elapsed market steps.

Dynamic routing is not assumed to dominate on every synthetic path; the benchmark measures the realized outcome.

## Run

~~~bash
python projects/multi-venue-fragmented-market/run_fragmented_market.py
~~~

Generated outputs:

~~~text
initial_venue_snapshot.csv
static_vs_dynamic_routing.csv
static_one_shot_route_decisions.csv
static_one_shot_executions.csv
dynamic_rerouting_route_decisions.csv
dynamic_rerouting_executions.csv
~~~

## Project code

~~~text
src/fragmented_market/
├── environment.py
└── router.py
~~~

The venue-level matching engine comes from the limit-order-book-simulator project.

## Relationship to Smart Order Routing

The original Smart Order Routing project is a static venue-allocation MILP using synthetic venue attributes.

The fragmented-market extension moves that problem into an event-driven setting:

~~~text
static venue parameters
        ↓
live independent books
        ↓
NBBO / depth / imbalance
        ↓
latency-aware routing
        ↓
realized fills
        ↓
dynamic rerouting
~~~

## Limitations

The environment is still synthetic. It does not model real exchange matching-rule differences, co-location infrastructure, microsecond timestamping, hidden liquidity, dark pools, pegged orders, venue-specific order types, exchange outages, or real fee tiers.

It is a research testbed for fragmented-market decision problems, not a broker routing system.
