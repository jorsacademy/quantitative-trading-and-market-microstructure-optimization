# RFQ Pricing and Dealer Inventory Optimization

A discrete quote-selection model for a synthetic dealer RFQ book.

For each client request, the dealer chooses a quote spread or rejects the RFQ.

Tighter quotes have higher acceptance probability but lower spread revenue. Wider quotes earn more conditional on acceptance but win less flow.

## Decisions

For every RFQ choose one of:

- 2 bps;
- 4 bps;
- 7 bps;
- reject.

## Acceptance model

Acceptance probability decreases exponentially with quoted spread:

\[
P(accept)
=
p_0 e^{-k s}
\]

where sensitivity differs by client/RFQ.

## Expected economics

Expected quote PnL is:

\[
P(accept)
\times
notional
\times
(spread - hedge\ cost)
\]

## Portfolio constraints

The model jointly prices the RFQ book under:

- expected gross accepted-notional capacity;
- positive/negative expected dealer inventory limits.

This prevents optimizing each RFQ independently without regard to the dealer book.

## Run

~~~bash
python -m trading_optimization.rfq_pricing
~~~

## Extensions

Possible extensions include client segmentation, instrument-specific hedge cost, nonlinear inventory penalties, live mid-price uncertainty, competitor quote distributions, multi-period RFQ arrival, and reinforcement-learning quote policies.

## Limitations

All data are synthetic. The model is not a production OTC pricing system and does not represent any specific market convention or client.


## Advanced electronic-credit dealer layer

The project now includes a second model that treats RFQ pricing and inventory
hedging as one multi-period decision system.

Module:

~~~text
rfq_pricing_dealer_optimization.credit_dealer
~~~

Compatibility entry point:

~~~bash
python -m trading_optimization.credit_rfq
~~~

### Risk state

Each synthetic corporate-bond RFQ carries two factor loadings:

- IG credit beta;
- HY credit beta.

Expected accepted flow therefore changes a two-dimensional dealer inventory
state. Client-sell RFQs add dealer credit exposure; client-buy RFQs reduce it.

### Joint quote and hedge decisions

For each RFQ the MILP chooses a discrete quote tier or rejects the request.
Acceptance probability is client/RFQ specific and decreases with quoted
spread.

At the same time, every RFQ wave can trade synthetic CDX-IG and CDX-HY hedges.

The state transition is:

~~~text
inventory_t
=
inventory_(t-1)
+
expected accepted RFQ factor flow
+
CDX hedge effect
~~~

The objective balances:

- expected RFQ spread PnL;
- explicit hedge-trading cost;
- absolute IG/HY inventory penalties;
- a stronger terminal inventory penalty.

Hard factor-inventory limits and per-period expected accepted-notional capacity
remain in force.

### Why this is different from the baseline

The baseline RFQ model optimizes one synthetic dealer book with a scalar signed
inventory limit.

The advanced layer adds:

- multiple credit-risk factors;
- multi-period inventory carry;
- explicit hedge instruments;
- cross-factor hedge basis;
- dynamic hedge trades;
- a hedged-versus-unhedged benchmark.

This turns the example into a compact electronic-credit dealer optimization
problem rather than a standalone quote calculator.

### Run the benchmark

~~~bash
python projects/rfq-pricing-dealer-optimization/run_credit_dealer.py
~~~

Generated outputs include:

~~~text
outputs/
├── credit_rfq_quotes.csv
├── credit_inventory_path.csv
├── credit_hedge_trades.csv
└── hedged_vs_unhedged_credit_dealer.csv
~~~

### Validation

The tests verify:

- exactly one quote/reject action per RFQ;
- factor-inventory limits;
- hedge-trade limits;
- the hedged model cannot have a worse optimum than the same model with hedge
  trades forced to zero;
- the synthetic client set produces more than one economically selected quote
  tier.

### Limitations

The factor loadings, client response curves and hedge effects are synthetic.
The model does not include live bond marks, real CDX basis, axes, dealer
capital, RFQ competition, information leakage, settlement, venue protocols or
actual client behavior.
