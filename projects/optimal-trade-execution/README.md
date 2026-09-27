# Optimal Trade Execution

A discrete-time optimal execution model for splitting a parent order across intraday intervals.

The model trades off:

- temporary market impact;
- inventory / timing risk;
- market-volume variation;
- participation-rate limits.

## Decision variable

For interval (t):

[
q_t = 	ext{quantity executed in interval }t
]

with:

[
sum_t q_t = Q
]

where (Q) is the parent order.

## Objective

The implementation minimizes an Almgren-Chriss-style cost-risk proxy:

[
min
quad
eta
sum_t rac{q_t^2}{L_t}
+
lambda
sum_t sigma_t^2 X_t^2
]

where:

- (L_t) is relative market liquidity;
- (X_t) is remaining inventory after interval (t);
- (eta) controls temporary impact;
- (lambda) controls risk aversion.

## Constraints

- parent-order completion;
- nonnegative execution;
- interval-specific maximum participation.

## Benchmarks

The optimized schedule is compared with:

- TWAP;
- VWAP.

A sensitivity experiment varies risk aversion and reports how execution timing shifts as inventory risk becomes more expensive.

## Run

```bash
python -m trading_optimization.execution
```

## Interpretation

Low risk aversion favors spreading quantity toward liquid intervals to reduce impact.

Higher risk aversion pushes execution earlier because carrying remaining inventory becomes more expensive.

## Limitations

This is a stylized execution model. It does not model queue position, limit orders, permanent impact calibration, alpha decay, venue fragmentation, spread dynamics, order-book resilience, latency, or information leakage.
