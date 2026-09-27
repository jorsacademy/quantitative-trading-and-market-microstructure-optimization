# Event-Driven Limit Order Book Simulator

A shared synthetic exchange environment for the execution, limit-order placement, and market-making projects in this repository.

The simulator is designed to move the repository from isolated cost/control models toward a common market-microstructure environment.

## Matching engine

The core engine implements a continuous double auction with:

- integer price ticks;
- price-time priority;
- limit orders;
- market orders;
- cancellation;
- partial fills;
- maker/taker attribution;
- same-price FIFO queues;
- visible depth;
- queue-ahead measurement.

Marketable limit orders first consume opposite-side liquidity and rest any residual quantity at their limit price.

## Event-driven environment

The environment seeds a multi-level bid/ask book and generates reproducible background order flow:

- passive limit arrivals;
- market-order arrivals;
- cancellations.

A fixed random seed makes complete market paths reproducible.

The observation exposes:

~~~text
time
best bid / ask
mid
spread
best-level bid / ask depth
order-book imbalance
last trade
~~~

## Trader accounting

Every trade updates:

- signed inventory;
- cash;
- marked-to-market PnL.

The simulator tracks maker and taker economics separately by trader ID.

## Shared strategy adapters

Three existing repository projects can now operate on the same exchange engine.

### Execution

A discrete execution schedule is converted to market-order child orders.

This makes it possible to compare optimized schedules not only through analytical impact models but through realized fills against a changing synthetic book.

### Market making

An Avellaneda-Stoikov-style inventory-skew adapter:

- cancels stale quotes;
- reposts bid/ask quotes;
- changes offsets with inventory;
- limits quote size by remaining inventory headroom.

### Limit-order placement

The existing queue/imbalance dynamic-programming policy receives live simulator observations.

Its abstract state is reconstructed from:

- actual queue-ahead quantity;
- current book imbalance;
- time/deadline.

The selected market/rest/improve/wait action is then translated into real exchange actions.

## Run

~~~bash
python projects/limit-order-book-simulator/run_shared_environment.py
~~~

The runner evaluates execution, market making, and limit placement on separate environments with the same market configuration and seed.

Generated outputs include:

~~~text
execution_steps.csv
execution_trades.csv
market_maker_steps.csv
market_maker_trades.csv
limit_placement_steps.csv
limit_placement_trades.csv
shared_environment_summary.csv
~~~

## Project structure

~~~text
src/limit_order_book_simulator/
├── engine.py
├── environment.py
└── strategies.py
~~~

## What this changes

Before this project, the repository's models mostly had their own stylized fill or cost processes.

The simulator provides a common experimental layer:

~~~text
optimization / control model
        ↓
strategy adapter
        ↓
event-driven matching engine
        ↓
background order flow
        ↓
realized fills / queue / PnL / inventory
~~~

This lets different decision methods be compared under the same synthetic exchange mechanics.

## Limitations

This is still a research simulator.

It does not model:

- hidden/iceberg liquidity;
- exchange-specific matching rules;
- maker/taker fee schedules;
- latency races;
- order modification priority rules;
- auctions;
- halts;
- multi-venue fragmentation;
- real calibration of order-flow intensities;
- self-exciting order flow;
- adverse-selection state transitions;
- production-grade exchange throughput.

The architecture is intended as a controllable testbed, not a market emulator.
