#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 12:15:53 2026
Weight Instability (Rolling Window Analysis)
@author: poddar
"""

import numpy as np
import pandas as pd
from .optimizer import PortfolioOptimizer


class StabilityAnalyzer:
    """
    Rolling window tangency portfolio estimation.
    """

    def __init__(self, returns, window=756, rf=0.02, trading_days=252):
        self.returns = returns
        self.window = window
        self.rf = rf
        self.trading_days = trading_days

    def rolling_weights(self):
        weights_history = []
        dates = []

        for i in range(self.window, len(self.returns)):
            window_returns = self.returns.iloc[i-self.window:i]

            mu = window_returns.mean() * self.trading_days
            Sigma = window_returns.cov() * self.trading_days

            optimizer = PortfolioOptimizer(mu, Sigma, self.rf)
            w = optimizer.tangency_weights()

            weights_history.append(w)
            dates.append(self.returns.index[i])

        weights_df = pd.DataFrame(
            weights_history,
            index=dates,
            columns=self.returns.columns
        )

        return weights_df