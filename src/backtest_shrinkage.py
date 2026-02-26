#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 19:05:05 2026
Out-of-sample backtest: sample vs LW covariance (rolling)
@author: poddar
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from .constrained_optimizer import ConstrainedOptimizer


class BacktesterShrinkage:
    """
    Rolling out-of-sample backtest comparing:
      - sample covariance
      - Ledoit–Wolf covariance
    using long-only max Sharpe via frontier scan (QP).
    """

    def __init__(self, returns: pd.DataFrame, window=756, rf=0.02, trading_days=252, frontier_points=60):
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame.")
        self.returns = returns
        self.window = int(window)
        self.rf = float(rf)
        self.trading_days = int(trading_days)
        self.frontier_points = int(frontier_points)

    def _perf_metrics(self, series: pd.Series):
        mean_d = series.mean()
        vol_d = series.std()
        ann_ret = mean_d * self.trading_days
        ann_vol = vol_d * np.sqrt(self.trading_days)
        sharpe = (ann_ret - self.rf) / ann_vol
        return float(ann_ret), float(ann_vol), float(sharpe)

    def run(self):
        port_sample = []
        port_lw = []
        dates = []

        for i in range(self.window, len(self.returns) - 1):
            W = self.returns.iloc[i - self.window : i]
            mu = W.mean() * self.trading_days

            # Sample Σ
            Sigma_s = W.cov() * self.trading_days

            # LW Σ (fit on daily returns)
            lw = LedoitWolf().fit(W.values)
            Sigma_lw = pd.DataFrame(
                lw.covariance_ * self.trading_days,
                index=W.columns,
                columns=W.columns
            )

            # Optimize long-only max Sharpe (frontier scan)
            w_s = ConstrainedOptimizer(mu, Sigma_s, rf=self.rf).max_sharpe(num_points=self.frontier_points)
            w_lw = ConstrainedOptimizer(mu, Sigma_lw, rf=self.rf).max_sharpe(num_points=self.frontier_points)

            r_next = self.returns.iloc[i + 1].values

            port_sample.append(float(w_s @ r_next))
            port_lw.append(float(w_lw @ r_next))
            dates.append(self.returns.index[i + 1])

        s_sample = pd.Series(port_sample, index=dates, name="sample")
        s_lw = pd.Series(port_lw, index=dates, name="lw")

        return s_sample, s_lw, self._perf_metrics(s_sample), self._perf_metrics(s_lw)