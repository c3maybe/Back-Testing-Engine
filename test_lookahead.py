"""Lookahead tests -- the heart of the project.

If any of these fail, no result the engine produces can be trusted. They probe
the two ways lookahead sneaks into a backtest:

1. Same bar execution: acting on a signal during the very bar that produced it.
2. Information leakage: a "past" decision that secretly depends on future data.
"""

import numpy as np
import pandas as pd

from backtester import EMACrossover, generate_gbm, run_backtest, CostModel


def test_no_same_bar_execution():
    prices = generate_gbm(n=60, seed=1)
    t = 30
    target = pd.Series(0.0, index=prices.index)
    target.iloc[t:] = 1.0  # decide "go long" first at the close of bar t

    res = run_backtest(prices, target, CostModel.frictionless())
    pos = res.position.to_numpy()

    assert pos[t] == 0.0, "position changed on the SAME bar the signal fired"
    assert pos[t + 1] > 0.0, "position should be established at the open of bar t+1"
    assert (pos[:t + 1] == 0.0).all(), "no position should exist before execution"


def test_signal_is_causal_under_future_perturbation():
    """EMA-cross decisions through bar k must not change when the FUTURE changes."""
    prices = generate_gbm(n=400, seed=2)
    strat = EMACrossover(5, 8)
    base = strat.generate_signals(prices)

    k = 200
    perturbed = prices.copy()
    rng = np.random.default_rng(99)
    # Scramble every price strictly after bar k.
    perturbed.iloc[k + 1:, :] *= rng.uniform(0.5, 1.5, size=perturbed.iloc[k + 1:, :].shape)
    after = strat.generate_signals(perturbed)

    assert np.allclose(base.iloc[:k + 1].to_numpy(),
                       after.iloc[:k + 1].to_numpy()), \
        "signal at/through bar k changed when future prices changed (lookahead)"


def test_engine_information_barrier():
    prices = generate_gbm(n=300, seed=3)
    target = EMACrossover(5, 8).generate_signals(prices)

    k = 150
    perturbed = prices.copy()
    rng = np.random.default_rng(7)
    perturbed.iloc[k + 1:, :] *= rng.uniform(0.5, 1.5, size=perturbed.iloc[k + 1:, :].shape)

    base = run_backtest(prices, target, CostModel())
    pert = run_backtest(perturbed, target, CostModel())

    assert np.allclose(base.position.to_numpy()[:k + 1],
                       pert.position.to_numpy()[:k + 1]), \
        "positions through bar k changed when future prices changed (lookahead)"


def test_flat_target_never_trades():
    prices = generate_gbm(n=100, seed=4)
    target = pd.Series(0.0, index=prices.index)
    res = run_backtest(prices, target, CostModel())
    assert len(res.fills) == 0
    assert np.allclose(res.equity.to_numpy(), res.initial_cash)
