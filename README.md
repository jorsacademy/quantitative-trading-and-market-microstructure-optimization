# Quantitative Trading and Market Microstructure Optimization
<!-- portfolio-umbrella:start -->
## Portfolio role

This repository is a primary umbrella repository in the consolidated Jors Academy portfolio. It groups related native research projects under `projects/` and serves as the main entry point for this domain.
<!-- portfolio-umbrella:end -->

Operations Research, stochastic control, and prescriptive analytics for electronic trading and market microstructure.

The repository focuses on the decision layer of quantitative trading:

> **signal / state estimates → optimize execution, quoting, routing, and inventory decisions**

It is deliberately not a collection of price-prediction notebooks. The core questions are how to trade, quote, route, size, and hedge under market impact, liquidity, execution risk, and inventory constraints.

## Project map

| Project | Decision problem | Methods | Status |
| --- | --- | --- | --- |
| [Optimal Trade Execution](projects/optimal-trade-execution/) | Schedule a parent order under impact, alpha decay, volatility scenarios, and tail risk | Convex optimization, learned impact model, stochastic execution, CVaR | Flagship |
| [Market Making & Inventory Control](projects/market-making-inventory-control/) | Optimize dealer quotes under single/multi-asset inventory risk and compare control vs RL | Dynamic programming, Avellaneda-Stoikov, multi-asset control, Q-learning | Flagship |
| [Smart Order Routing](projects/smart-order-routing/) | Allocate maker, taker, and dark child orders under latency, queue risk, and venue toxicity | MILP, toxicity-aware hybrid routing, midpoint dark pool, dynamic rerouting | Flagship |
| [Limit Order Placement](projects/limit-order-placement/) | Decide market/limit/wait actions from queue and order-book state | MDP / dynamic programming | Implemented |
| [RFQ Pricing & Dealer Optimization](projects/rfq-pricing-dealer-optimization/) | Price corporate-credit RFQs while managing multi-period IG/HY inventory with CDX-style hedges | MILP, client-response elasticity, dynamic inventory, hedge optimization | Flagship |
| [Event-Driven Limit Order Book Simulator](projects/limit-order-book-simulator/) | Run strategies under calibrated self-exciting and queue-reactive order flow | Price-time matching, Hawkes flow, queue-reactive calibration, toxicity model | Flagship |
| [Multi-Venue Fragmented Market](projects/multi-venue-fragmented-market/) | Route across lit and dark liquidity with calibrated venue flow and adverse-selection risk | Fragmented-market simulation, Hawkes flow, toxicity-aware maker/taker/dark routing | Flagship |

## Design principles

Each project is intended to include:

- an explicit mathematical decision model;
- deterministic synthetic data or market-state generator;
- executable Python implementation;
- benchmark policies;
- feasibility/invariant tests;
- scenario or sensitivity analysis;
- clear limitations.

## Installation

Python 3.10+:

```bash
pip install -e ".[dev]"
pytest
```

## Run examples

```bash
python -m trading_optimization.execution
python projects/optimal-trade-execution/run_advanced.py

python -m trading_optimization.market_making
python projects/market-making-inventory-control/run_advanced.py
python projects/limit-order-book-simulator/run_shared_environment.py
python projects/limit-order-book-simulator/run_microstructure_calibration.py
python projects/multi-venue-fragmented-market/run_fragmented_market.py
python projects/multi-venue-fragmented-market/run_hybrid_routing.py
python projects/multi-venue-fragmented-market/run_toxicity_routing.py

python -m trading_optimization.smart_order_routing
python -m trading_optimization.limit_order_placement
python -m trading_optimization.rfq_pricing
```

## Repository structure

```text
projects/
├── optimal-trade-execution/
├── market-making-inventory-control/
├── smart-order-routing/
├── limit-order-placement/
├── rfq-pricing-dealer-optimization/
├── limit-order-book-simulator/
└── multi-venue-fragmented-market/

src/
└── trading_optimization/

tests/
```

## Methodological theme

The common pattern is:

```text
market state / forecasts
        ↓
execution or quoting economics
        ↓
optimization / control model
        ↓
action policy
        ↓
benchmark + stress validation
        ↓
shared event-driven market simulation
```

## Scope

The examples are synthetic research/education models. They are not production trading systems and do not include exchange connectivity, live market data, broker integration, latency engineering, regulatory controls, transaction reporting, or capital deployment.

## Disclaimer

Educational and research use only. Nothing in this repository is investment advice, trading advice, or a recommendation to transact in any financial instrument.
