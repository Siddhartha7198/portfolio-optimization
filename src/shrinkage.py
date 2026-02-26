#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 19:01:15 2026
Covariance shrinkage (Ledoit–Wolf)
@author: poddar
"""


import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class LedoitWolfShrinkage:
    """
    Ledoit–Wolf covariance shrinkage.

    Input: returns DataFrame (T x N) of daily log returns.
    Output: annualized covariance matrix (N x N) as DataFrame.
    """

    def __init__(self, returns: pd.DataFrame, trading_days: int = 252):
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame with columns as assets.")
        self.returns = returns
        self.trading_days = trading_days

    def fit(self):
        # sklearn expects shape (n_samples, n_features) = (T, N)
        X = self.returns.values

        lw = LedoitWolf().fit(X)

        # lw.covariance_ is DAILY covariance
        Sigma_daily = lw.covariance_

        # annualize
        Sigma_ann = Sigma_daily * self.trading_days

        Sigma_ann_df = pd.DataFrame(
            Sigma_ann, index=self.returns.columns, columns=self.returns.columns
        )

        return Sigma_ann_df, float(lw.shrinkage_)