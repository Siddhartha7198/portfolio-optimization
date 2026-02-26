#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 11:48:22 2026
Covariance Diagnostics & Stability
@author: poddar
"""


import numpy as np


class CovarianceDiagnostics:
    """
    Provides diagnostics for covariance matrix stability.
    """

    def __init__(self, Sigma):
        self.Sigma = Sigma.values

    def eigenvalues(self):
        eigvals = np.linalg.eigvalsh(self.Sigma)
        return eigvals

    def condition_number(self):
        eigvals = self.eigenvalues()
        return max(eigvals) / min(eigvals)

    def is_positive_definite(self):
        eigvals = self.eigenvalues()
        return np.all(eigvals > 0)