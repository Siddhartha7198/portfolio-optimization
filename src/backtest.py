#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 12:38:38 2026
Out-of-Sample Backtesting
@author: poddar
"""


import numpy as np
import pandas as pd
from .optimizer import PortfolioOptimizer


class Backtester:
    """
    Rolling out-of-sample backtest for tangency portfolio.
    """

    def __init__(self, returns, window=756, rf=0.02, trading_days=252):
        self.returns = returns
        self.window = window
        self.rf = rf
        self.trading_days = trading_days

    def run(self):
        portfolio_returns = []
        dates = []

        for i in range(self.window, len(self.returns) - 1):
            window_returns = self.returns.iloc[i-self.window:i]

            mu = window_returns.mean() * self.trading_days
            Sigma = window_returns.cov() * self.trading_days

            optimizer = PortfolioOptimizer(mu, Sigma, self.rf)
            w = optimizer.tangency_weights()

            next_return = self.returns.iloc[i+1]
            portfolio_return = w @ next_return

            portfolio_returns.append(portfolio_return)
            dates.append(self.returns.index[i+1])

        return pd.Series(portfolio_returns, index=dates)

    def performance_metrics(self, portfolio_returns):
        mean_daily = portfolio_returns.mean()
        vol_daily = portfolio_returns.std()

        annual_return = mean_daily * self.trading_days
        annual_vol = vol_daily * np.sqrt(self.trading_days)
        sharpe = (annual_return - self.rf) / annual_vol

        return annual_return, annual_vol, sharpe