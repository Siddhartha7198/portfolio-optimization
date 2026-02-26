#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 15:14:19 2026
Log Return Calculation
@author: poddar
"""

import numpy as np
import pandas as pd


class ReturnEstimator:
    """
    Computes log returns and annualized statistics.
    """

    def __init__(self, prices, trading_days=252):
        self.prices = prices
        self.trading_days = trading_days

    def compute_log_returns(self):
        """
        R_t = log(P_t / P_{t-1})
        """
        log_prices = np.log(self.prices)
        returns = log_prices.diff().dropna()
        return returns

    def annualized_mean(self, returns):
        """
        μ_ann = 252 * μ_daily
        """
        return returns.mean() * self.trading_days

    def annualized_covariance(self, returns):
        """
        Σ_ann = 252 * Σ_daily
        """
        return returns.cov() * self.trading_days