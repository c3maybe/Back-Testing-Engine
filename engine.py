"""The backtest engine.

Design goals, in priority order: (1) no lookahead, (2) correct cash/position
accounting, (3) realistic frictions, (4) speed. The engine is an explicit
bar-by-bar loop rather than a vectorised one-liner -- it is plenty fast for
daily data and, more importantly, it makes the order of operations auditable.

* half-spread  -- you buy at ``open*(1 + s)`` and sell at ``open*(1 - s)``.
* slippage     -- an additional adverse move in the same direction as spread.
* commission   -- per-notional (bps) plus an optional fixed charge per fill.

All three are applied so they *always* work against you, which is the honest
direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

_EPS = 1e-12


@dataclass
class CostModel:

    half_spread_bps: float = 1.0
    slippage_bps: float = 0.5
    commission_bps: float = 0.5
    commission_fixed: float = 0.0

    @classmethod
    def frictionless(cls) -> "CostModel":
        return cls(0.0, 0.0, 0.0, 0.0)

    @property
    def adverse(self) -> float:
        """Fractional adverse price move applied to every fill."""
        return (self.half_spread_bps + self.slippage_bps) * 1e-4


@dataclass
class Fill:
    bar: int
    time: pd.Timestamp
    dq: float           # signed change in units
    price: float        # fill price incl. spread + slippage
    commission: float
    notional: float     # abs(dq) * price


@dataclass
class Trade:
    """A closed round-trip (or the closed portion of one)."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int      # +1 long, -1 short
    qty: float
    entry_price: float  # avg net cost basis per unit
    exit_price: float   # net proceeds per unit
    pnl: float          # net of all frictions


@dataclass
class BacktestResult:
    equity: pd.Series
    position: pd.Series # units held (marked at close)
    cash: pd.Series
    target: pd.Series   # target fraction decided each bar
    exposure: pd.Series # position value / equity
    fills: List[Fill]
    trades: List[Trade]
    prices: pd.DataFrame
    initial_cash: float
    cost_model: CostModel
    meta: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)

    @property
    def total_traded_notional(self) -> float:
        return float(sum(f.notional for f in self.fills))


def run_backtest(prices: pd.DataFrame, target: pd.Series,
                 cost_model: Optional[CostModel] = None,
                 initial_cash: float = 100_000.0,
                 max_leverage: float = 1.0,
                 allow_short: bool = True) -> BacktestResult:
    """Simulate trading ``target`` against ``prices``.

    Parameters
    ----------
    prices : DataFrame with at least Open and Close, DatetimeIndex.
    target : Series of target position fractions, aligned to ``prices``.
             ``target[t]`` is decided at the close of bar ``t``.
    cost_model : frictions (defaults to :class:`CostModel` defaults).
    initial_cash : starting equity.
    max_leverage : hard cap on ``|target|`` -- prevents accidental infinite
                   leverage no matter what the strategy emits.
    allow_short : if False, negative targets are clipped to 0.
    """
    cost_model = cost_model or CostModel()
    if not prices.index.equals(target.index):
        target = target.reindex(prices.index)
    target = target.fillna(0.0).astype(float)

    lo = -max_leverage if allow_short else 0.0
    tgt = target.clip(lower=lo, upper=max_leverage).to_numpy()

    opens = prices["Open"].to_numpy(dtype=float)
    closes = prices["Close"].to_numpy(dtype=float)
    times = prices.index
    n = len(prices)

    cash = float(initial_cash)
    position = 0.0          # signed units
    avg_cost = 0.0          # net cost basis per unit
    open_entry_time = None  # when the current open position was established

    equity_arr = np.empty(n)
    pos_arr = np.empty(n)
    cash_arr = np.empty(n)

    fills: List[Fill] = []
    trades: List[Trade] = []
    adverse = cost_model.adverse

    pending = 0.0
    held_target = 0.0
    prev_equity = float(initial_cash)
    prev_close = closes[0]

    for i in range(n):
        if i > 0 and abs(pending - held_target) > _EPS:
            desired_units = pending * prev_equity / prev_close
            dq = desired_units - position
            held_target = pending
            if abs(dq) > _EPS:
                direction = 1.0 if dq > 0 else -1.0
                fill_price = opens[i] * (1.0 + direction * adverse)
                notional = abs(dq) * fill_price
                commission = cost_model.commission_fixed + cost_model.commission_bps * 1e-4 * notional
                # Real cash flow
                cash -= dq * fill_price
                cash -= commission
                fills.append(Fill(i, times[i], dq, fill_price, commission, notional))

                # Average-cost accounting + realised PnL (net of fees)
                fee_per_unit = commission / abs(dq)
                # Net per-unit acquisition cost (buy) / proceeds (sell).
                buy_unit = fill_price + fee_per_unit
                sell_unit = fill_price - fee_per_unit

                if position == 0.0 or (position > 0) == (dq > 0):
                    # Opening from flat, or increasing an existing position.
                    new_units = abs(position) + abs(dq)
                    unit = buy_unit if dq > 0 else sell_unit
                    avg_cost = (avg_cost * abs(position) + unit * abs(dq)) / new_units
                    if position == 0.0:
                        open_entry_time = times[i]
                    position += dq
                else:
                    # Reducing, closing, or flipping the position.
                    closing = min(abs(dq), abs(position))
                    if position > 0:      # closing a long by selling
                        pnl = (sell_unit - avg_cost) * closing
                        d = 1
                    else:                  # closing a short by buying
                        pnl = (avg_cost - buy_unit) * closing
                        d = -1
                    trades.append(Trade(open_entry_time, times[i], d, closing,
                                        avg_cost, (sell_unit if position > 0 else buy_unit),
                                        float(pnl)))
                    if abs(dq) > abs(position) + _EPS:
                        # Flip: remainder opens a fresh position on the other side.
                        remainder = abs(dq) - abs(position)
                        position = direction * remainder
                        avg_cost = buy_unit if direction > 0 else sell_unit
                        open_entry_time = times[i]
                    else:
                        position += dq
                        if abs(position) < _EPS:
                            position = 0.0
                            open_entry_time = None
                        # avg_cost of the remaining same-side units is unchanged.

        equity = cash + position * closes[i]
        equity_arr[i] = equity
        pos_arr[i] = position
        cash_arr[i] = cash

        pending = tgt[i]
        prev_equity = equity
        prev_close = closes[i]

    equity_s = pd.Series(equity_arr, index=times, name="equity")
    pos_s = pd.Series(pos_arr, index=times, name="position")
    cash_s = pd.Series(cash_arr, index=times, name="cash")
    exposure_s = (pos_s * prices["Close"] / equity_s).rename("exposure")

    return BacktestResult(
        equity=equity_s, position=pos_s, cash=cash_s,
        target=target, exposure=exposure_s,
        fills=fills, trades=trades, prices=prices,
        initial_cash=float(initial_cash), cost_model=cost_model,
        meta={"max_leverage": max_leverage, "allow_short": allow_short,
              "final_avg_cost": float(avg_cost), "final_position": float(position)},
    )
