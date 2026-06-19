# Realistic Backtesting Engine

Situation: Does a naive EMA(5/8) crossover survive realistic trading costs? (I made an algorithm which would check that for US currency)

Short answer: No. After costs it returns +6.7% over five years while buying and holding returns 134%, with a worse drawdown, and its mean daily return is statistically indistinguishable from zero. The repository is the evidence.

The goal here is not a profitable strategy. The goal is a
backtester that I can trust to tell the truth and a worked example of it delivering bad news cleanly.

## Method

A signal is expressed as a target position, a fraction of equity in `[-1, 1]`, 
decided at the close of each bar. The engine is an explicit loop, bar by bar, that can only act on the previous bar's decision, filling at the open of the next bar. There is exactly one place in the code where a decision becomes a fill, and it is structurally incapable of seeing the future. A test mutates all future prices and asserts that past positions do not move (`tests/test_lookahead.py`).

Every fill pays frictions, applied adverse to the trader: half spread, slippage, and
commission (all in basis points, all parameters). Accounting uses an average cost basis. Cash is the single source of truth, and an accord reconstructs the
equity curve two independent ways and checks both against zero.

The strategy only trades when its target fraction changes, it does not rebalance every bar, intern trade counts and turnover reflect real signal activity rather than accounting noise.

## Assumptions (and their limits)

One asset, one regime.
- Daily AAPL, 2013-02-08 to 2018-02-07 (1,259 bars). A long biased strategy will look good here for reasons that have nothing to do with skill.
Costs
- 1.0 bps half spread, 0.5 bps slippage, 0.5 bps commission by default. Liquid large cap equity assumptions.
Sharpe
- rf = 0, daily returns annualized by √252, sample std (ddof=1).
Fills
- Fills at the bar open with no volume/impact model, dividends ignored (price return only), no borrow cost on shorts.
Results 
- Results are in sample on a single series unless the walk-forward section says otherwise. Treat every number as an illustration of method, not an estimate of edge.

## The honest conclusion
(AAPL, 2013–2018, after costs), EMA(5/8) L/S, Buy & hold
Total return:  +6.7%, +134.3%
Sharpe: 0.17, 0.85
Max drawdown: −43%, −32%
Trades/win rate: 118/30.5%, 1 / — 
Annualized turnover: 46.6×, 0

The mean daily return after costs has a t-stat of 0.38, we cannot reject that it is
zero. Sweeping the EMA lengths over a 6×6 grid gives after cost Sharpes ranging from −0.48 to +0.71 that flip sign between neighboring settings: the classic fingerprint of noise, not signal. A walk-forward test that retunes the parameters out of sample always collapses to the slowest pair available (3, 89) and its OOS return (+60.2%) merely ties buy-and-hold over the same window (+58.3%). Even the one robust-looking result is a regime artifact, not an edge.

---

## Project layout

```
backtester/
  data.py        load_csv / load_sample / load_yfinance / generate_gbm
  strategy.py    Strategy base class + EMACrossover (emits target fractions)
  engine.py      CostModel, run_backtest — the no-lookahead simulation loop
  metrics.py     Sharpe, max drawdown, turnover, PnL reconciliation
  analysis.py    buy_and_hold, parameter_grid, walk_forward
tests/           14 tests: lookahead, accounting, costs
notebooks/
  ema_backtest.ipynb   the end-to-end story, with equity curves before/after costs
data/
  sample_prices.csv    bundled AAPL daily OHLCV (so everything runs offline)
```

## How to run

```bash
pip install -r requirements.txt

# the centerpiece: a test that fails if the engine can ever see the future
pytest -q

# the full narrative with plots
jupyter notebook notebooks/ema_backtest.ipynb
```

A minimal end-to-end run in code:

```python
from backtester import load_sample, EMACrossover, CostModel, run_backtest, compute_metrics

prices = load_sample("AAPL")                 # bundled offline data
target = EMACrossover(fast=5, slow=8).generate_signals(prices)

gross = run_backtest(prices, target, CostModel.frictionless())
net   = run_backtest(prices, target, CostModel())   # realistic frictions

print(compute_metrics(gross)["total_return"], "->", compute_metrics(net)["total_return"])
```

**Using your own data.** Swap `load_sample("AAPL")` for `load_yfinance("AAPL", start="2015-01-01")` (requires `pip install yfinance` and internet), or `load_csv(path)` for any CSV with
`Date, Open, High, Low, Close` columns. Everything downstream is unchanged.

## Data caveat

The bundled sample is daily AAPL OHLCV from the public `plotly/datasets` mirror, chosen so the notebook and tests run fully offline and reproducibly. It is split but not dividend-adjusted, which understates the true buy-and-hold return strengthening, not weakening, the conclusion that the active strategy underperforms.
