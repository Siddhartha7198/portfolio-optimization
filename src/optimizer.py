#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 16:19:39 2026

@author: poddar
"""

import numpy as np


class PortfolioOptimizer:
    """
    Computes tangency portfolio and Sharpe ratio.
    """

    def __init__(self, mu, Sigma, rf=0.02):
        self.mu = mu.values
        self.Sigma = Sigma.values
        self.rf = rf
        self.ones = np.ones(len(mu))

    def tangency_weights(self):
        """
        w* = Σ^{-1}(μ - rf 1) / (1^T Σ^{-1}(μ - rf 1))
        """

        excess_returns = self.mu - self.rf * self.ones

        Sigma_inv_excess = np.linalg.solve(self.Sigma, excess_returns)

        normalization = self.ones @ Sigma_inv_excess

        w = Sigma_inv_excess / normalization

        return w

    def portfolio_performance(self, w):
        """
        Returns (expected return, volatility, Sharpe ratio)
        """

        expected_return = w @ self.mu
        volatility = np.sqrt(w @ self.Sigma @ w)
        sharpe = (expected_return - self.rf) / volatility

        return expected_return, volatility, sharpe