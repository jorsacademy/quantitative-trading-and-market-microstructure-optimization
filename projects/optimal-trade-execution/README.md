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


## Advanced layer: learned impact + stochastic alpha-aware execution

The project also contains a second execution stack:

\`\`\`text
synthetic execution observations
        ↓
nonnegative impact-model calibration
        ↓
alpha-decay + volatility scenarios
        ↓
stochastic execution optimization
        ↓
expected cost + CVaR tail control
\`\`\`

### Learned market-impact model

Module:

\`\`\`text
optimal_trade_execution.impact_model
\`\`\`

Synthetic historical execution observations include:

- participation rate;
- squared participation;
- volatility;
- half-spread;
- order-book imbalance;
- realized impact in bps.

The impact model is calibrated with nonnegative least squares so the estimated
microstructure cost coefficients remain economically interpretable.

The model predicts quantity-weighted impact cost for candidate schedules and is
fed directly into the stochastic optimizer.

### Alpha decay

The stochastic execution layer assumes that a short-lived execution advantage
decays through the horizon:

\[
\alpha_t = \alpha_0 e^{-kt} + \varepsilon_t
\]

For a sell program, executing earlier captures more of this temporary alpha
advantage, while trading too aggressively increases learned market impact.

### Scenario uncertainty

Each optimization instance generates multiple paths for:

- alpha;
- volatility.

The same schedule must be feasible across all scenarios.

For scenario \(s\), the stylized path cost contains:

\[
C_s(q)
=
Impact_{\hat f}(q)
+
InventoryRisk_s(q)
-
CapturedAlpha_s(q)
\]

### CVaR objective

The advanced optimizer minimizes:

\[
E[C_s]
+
\lambda_{CVaR}CVaR_\beta(C_s)
\]

so execution is not chosen only for average performance; expensive tail paths
also influence the schedule.

### Benchmarks

The resulting stochastic schedule is compared against:

- TWAP;
- VWAP;
- the original deterministic impact/risk optimizer.

Reported metrics include:

- expected scenario cost;
- VaR threshold;
- CVaR;
- quantity-weighted execution time;
- learned-impact calibration RMSE.

### Run advanced experiment

\`\`\`bash
python projects/optimal-trade-execution/run_advanced.py
\`\`\`

Generated outputs include:

\`\`\`text
impact_training_data.csv
impact_model_coefficients.csv
execution_alpha_scenarios.csv
execution_volatility_scenarios.csv
stochastic_execution_schedule.csv
stochastic_remaining_inventory.csv
stochastic_execution_scenario_costs.csv
\`\`\`

## Project code

\`\`\`text
src/optimal_trade_execution/
├── model.py
├── impact_model.py
└── stochastic_execution.py
\`\`\`

The original deterministic model remains the transparent baseline. The
stochastic layer is intentionally separate so the incremental modeling
assumptions are inspectable.
