#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 17:37:57 2026
Long-only optimization (Quadratic Programming)
@author: poddar
"""

import numpy as np
import cvxpy as cp


class ConstrainedOptimizer:
    """
    Long-only Markowitz optimization using convex programming (cvxpy).
    Provides:
      - Efficient frontier with w >= 0
      - Max Sharpe (long-only) via bisection + SOCP feasibility
    """

    def __init__(self, mu, Sigma, rf=0.02):
        # accept pandas Series/DataFrame or numpy
        self.mu = np.asarray(getattr(mu, "values", mu)).reshape(-1)
        self.Sigma = np.asarray(getattr(Sigma, "values", Sigma))
        self.rf = float(rf)
        self.n = self.mu.shape[0]
        self.ones = np.ones(self.n)

        # small numerical jitter can help PSD issues from estimation noise
        self.Sigma_psd = self._make_psd(self.Sigma)

    @staticmethod
    def _make_psd(Sigma, eps=1e-10):
        # Symmetrize + jitter on diagonal to avoid numerical non-PSD
        Sigma = 0.5 * (Sigma + Sigma.T)
        return Sigma + eps * np.eye(Sigma.shape[0])

    def efficient_portfolio(self, target_return):
        """
        Long-only minimum variance portfolio for a target return:
            min w' Σ w
            s.t. μ'w >= target, 1'w = 1, w >= 0
        """
        w = cp.Variable(self.n)

        objective = cp.Minimize(cp.quad_form(w, self.Sigma_psd))
        constraints = [
            self.mu @ w >= float(target_return),
            self.ones @ w == 1.0,
            w >= 0
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)  # OSQP is great for QPs

        if w.value is None:
            raise ValueError("QP infeasible or solver failed for target_return=" + str(target_return))

        return np.asarray(w.value).reshape(-1)

    def frontier(self, num_points=50, ret_min=None, ret_max=None):
        """
        Generate the long-only efficient frontier (returns, vols, weights_list).
        """
        # feasible return range under long-only is roughly [min(mu), max(mu)]
        mu_min = self.mu.min() if ret_min is None else float(ret_min)
        mu_max = self.mu.max() if ret_max is None else float(ret_max)

        targets = np.linspace(mu_min, mu_max, num_points)
        vols = []
        weights_list = []

        for tr in targets:
            try:
                w = self.efficient_portfolio(tr)
                var = w.T @ self.Sigma_psd @ w
                vols.append(np.sqrt(var))
                weights_list.append(w)
            except ValueError:
                # skip infeasible points near extremes
                continue

        return np.array(targets[:len(vols)]), np.array(vols), weights_list

    # ---------- Max Sharpe (long-only) via bisection + SOCP ----------

    def _is_sharpe_feasible(self, s):
        """
        Feasibility check for Sharpe >= s:
            exists w,t:
              mu_excess' w >= s * t
              ||L w||_2 <= t
              1'w = 1, w >= 0
        """
        mu_excess = self.mu - self.rf * self.ones

        # Cholesky may fail if Sigma not strictly PD; add jitter already handled
        L = np.linalg.cholesky(self.Sigma_psd).T  # so that ||L w||_2^2 = w' Σ w

        w = cp.Variable(self.n)
        t = cp.Variable(nonneg=True)

        constraints = [
            mu_excess @ w >= float(s) * t,
            cp.norm(L @ w, 2) <= t,
            self.ones @ w == 1.0,
            w >= 0
        ]

        prob = cp.Problem(cp.Minimize(0), constraints)
        prob.solve(solver=cp.SCS, verbose=False)  # SCS handles SOCP well

        return w.value is not None

    def max_sharpe(self, num_points=60, ret_min=None, ret_max=None):
        """
        Long-only max Sharpe via scanning the long-only efficient frontier:
          1) Generate frontier portfolios (QP)
          2) Compute Sharpe for each
          3) Return weights with maximum Sharpe
    
        This is robust and avoids SOCP recovery edge cases.
        """
        target_returns, vols, weights_list = self.frontier(
            num_points=num_points, ret_min=ret_min, ret_max=ret_max
        )
    
        if len(weights_list) == 0:
            raise ValueError("Frontier generation failed; cannot compute max Sharpe.")
    
        best_idx = None
        best_sharpe = -np.inf
    
        for i, w in enumerate(weights_list):
            r, v, s = self.performance(w)
            if np.isfinite(s) and s > best_sharpe:
                best_sharpe = s
                best_idx = i
    
        if best_idx is None:
            raise ValueError("Could not find a valid Sharpe ratio on the frontier scan.")
    
        return weights_list[best_idx]

    def performance(self, w):
        """
        Return (expected_return, volatility, sharpe) using annualized mu/Sigma inputs.
        """
        w = np.asarray(w).reshape(-1)
        ret = w @ self.mu
        vol = np.sqrt(w @ self.Sigma_psd @ w)
        sharpe = (ret - self.rf) / vol
        return ret, vol, sharpe