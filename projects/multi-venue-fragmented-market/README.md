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


## Hybrid lit-maker / lit-taker / dark-pool routing

The fragmented-market layer now also supports a second execution architecture:

~~~text
live venue books
      +
midpoint dark pool
      ↓
candidate generation
      ↓
lit taker
lit maker
dark midpoint
      ↓
hybrid allocation MILP
      ↓
fills / queue / latency / hidden liquidity
      ↓
residual rerouting
~~~

### Lit taker

A taker child order crosses the current venue book.

Its routing score reflects:

- executable best price;
- taker fee;
- venue latency;
- probability that displayed liquidity is still available on arrival;
- fallback penalty for unfilled quantity.

### Lit maker

A maker child order rests at the current best same-side quote.

Expected fill probability depends on:

- current queue-ahead quantity;
- order-book imbalance;
- venue latency;
- configured patience horizon.

Maker economics include the venue maker rebate.

Resting orders are cancelled after their patience budget and any residual quantity can be rerouted.

### Dark midpoint

The dark-pool simulator has:

- no displayed depth;
- midpoint execution;
- stochastic hidden contra liquidity;
- delayed eligibility;
- time-in-force;
- explicit dark-pool fee.

Fill probability increases with the waiting horizon, but execution is uncertain.

### Hybrid optimization

Each routing wave solves an integer allocation over venue × execution-mode candidates.

The model chooses quantity across:

~~~text
venue_a:lit_taker
venue_a:lit_maker
venue_b:lit_taker
venue_b:lit_maker
...
dark_midpoint:dark_midpoint
~~~

under:

- per-candidate capacity;
- maximum active actions;
- minimum expected fill ratio;
- parent-order conservation.

The expected-cost proxy incorporates price, fee/rebate, latency, queue risk, and fallback cost.

### Hard deadline

Passive lit and dark orders are cancelled before the final deadline.

Any remaining quantity is crossed through the dynamic lit-taker router so the benchmark distinguishes:

- price improvement from passive/dark execution;
- fill risk and waiting;
- deadline completion cost.

## Hybrid benchmark

Run:

~~~bash
python projects/multi-venue-fragmented-market/run_hybrid_routing.py
~~~

The benchmark compares:

~~~text
dynamic lit-taker-only routing
vs.
hybrid lit-maker / lit-taker / dark routing
~~~

on matched synthetic market seeds.

Reported metrics include:

- total fill ratio;
- average execution price;
- explicit fees / maker rebates;
- implementation shortfall;
- maker fill quantity;
- taker fill quantity;
- dark fill quantity;
- market steps.

Generated outputs include:

~~~text
lit_vs_hybrid_routing.csv
lit_taker_route_decisions.csv
lit_taker_executions.csv
hybrid_route_decisions.csv
hybrid_lit_executions.csv
hybrid_dark_executions.csv
~~~

## Extended project code

~~~text
src/fragmented_market/
├── environment.py
├── router.py
├── dark_pool.py
└── hybrid_router.py
~~~


## Toxicity-aware fragmented routing

Every lit venue can now optionally run the calibrated Hawkes + queue-reactive order-flow environment.

The consolidated venue snapshot then adds:

~~~text
recent signed flow pressure
Hawkes directional pressure
toxicity probability
market-buy intensity
market-sell intensity
~~~

The hybrid router consumes these states directly.

For lit taker orders, the toxicity penalty increases when the router is chasing aggressive flow in the same direction.

For passive maker orders, adverse-selection risk increases when flow pressure is likely to trade against the resting quote.

Dark midpoint candidates receive a lower toxicity loading because displayed-queue signaling is absent, while still retaining fill-risk and waiting costs.

### Benchmark

~~~bash
python projects/multi-venue-fragmented-market/run_toxicity_routing.py
~~~

This compares:

~~~text
hybrid routing with toxicity penalty disabled
vs.
toxicity-aware hybrid routing
~~~

on matched calibrated market seeds.

The benchmark exports route decisions, lit/dark executions, venue event logs, and probability-weighted toxicity of selected child-order actions.

The toxicity-aware policy is not assumed to dominate every path; it explicitly trades execution price/fill probability against modeled adverse-selection exposure.
