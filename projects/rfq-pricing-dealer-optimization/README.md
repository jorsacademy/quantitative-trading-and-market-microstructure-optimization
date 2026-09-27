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
