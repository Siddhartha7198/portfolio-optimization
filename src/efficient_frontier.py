#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 15:31:54 2026
Efficient Frontier Construction
@author: poddar
"""


import numpy as np


class EfficientFrontier:
    """
    Closed-form Markowitz efficient frontier.
    """

    def __init__(self, mu, Sigma):
        self.mu = mu.values
        self.Sigma = Sigma.values
        self.ones = np.ones(len(mu))

        # Precompute inverse-related quantities using solve
        self.Sigma_inv_ones = np.linalg.solve(self.Sigma, self.ones)
        self.Sigma_inv_mu = np.linalg.solve(self.Sigma, self.mu)

        self.A = self.ones @ self.Sigma_inv_ones
        self.B = self.ones @ self.Sigma_inv_mu
        self.C = self.mu @ self.Sigma_inv_mu
        self.D = self.A * self.C - self.B**2

    def weights(self, target_return):
        """
        Compute portfolio weights for a target return.
        """

        term1 = (self.C - self.B * target_return) / self.D
        term2 = (self.A * target_return - self.B) / self.D

        w = (
            term1 * self.Sigma_inv_ones
            +
            term2 * self.Sigma_inv_mu
        )

        return w

    def portfolio_variance(self, w):
        return w.T @ self.Sigma @ w

    def frontier(self, num_points=50):
        """
        Generate efficient frontier curve.
        """
        target_returns = np.linspace(
            min(self.mu),
            max(self.mu),
            num_points
        )

        volatilities = []
        weights_list = []

        for r in target_returns:
            w = self.weights(r)
            vol = np.sqrt(self.portfolio_variance(w))

            volatilities.append(vol)
            weights_list.append(w)

        return target_returns, np.array(volatilities), weights_list