# 📈 Mean–Variance Portfolio Optimization (Markowitz)


## Executive Summary

This project implements and analyzes the classical mean–variance portfolio optimization framework with real ETF market data. Beyond constructing optimal portfolios, it rigorously examines numerical instability, sensitivity to estimation error, effects of realistic constraints (long-only), and the benefits of covariance shrinkage (Ledoit–Wolf). The analysis demonstrates why naive mean–variance optimization often produces unstable results and how practical remedies improve robustness and out-of-sample performance.

The project constructs:
- **Efficient frontier** for risk-return tradeoffs
- **Tangency portfolio** (maximum Sharpe ratio)
- **Long-only optimization** reflecting real constraints
- **Shrinkage covariance estimator** (Ledoit–Wolf)
- **Rolling stability and out-of-sample backtest**
  
It provides diagnostic metrics such as condition number, eigenvalue spectra, concentration indices, and turnover measures to deepen understanding of model behavior.
  
---

## 1. Motivation and Practical Relevance

Portfolio optimization is a foundational tool in quantitative finance used to allocate capital across assets to balance risk and return. While textbook Markowitz optimization provides elegant closed-form solutions, real financial data introduce estimation error, ill-conditioned covariance matrices, and instability in optimal weights — all of which are critical in practice. This project bridges theory and practice, illustrating both strengths and limitations of classical methods and introducing remedies used in professional asset management.

---

## 2. Model Overview

The classical mean–variance optimization problem is:

$$
\min_{w}\; w^T \Sigma w
$$

Subject to:

$$
w^T \mu = \mu^*, 
\qquad 
\mathbf{1}^T w = 1
$$

---

## 3. Mathematical Framework

The tangency portfolio with risk-free rate $r_f$ is:

$$
w^* =
\frac{\Sigma^{-1}(\mu - r_f \mathbf{1})}
{\mathbf{1}^T \Sigma^{-1}(\mu - r_f \mathbf{1})}
$$

Long-only optimization adds:

$$
w \ge 0
$$

Covariance shrinkage uses:

$$
\hat{\Sigma}_{LW} = (1-\delta)\hat{\Sigma}_{sample} + \delta \bar{\sigma}^2 I
$$

where $\delta$ is the Ledoit–Wolf shrinkage intensity and $\bar{\sigma}^2$ is the average sample variance.

---

## 4. Data Requirements

The analysis uses daily adjusted price data for a universe of highly liquid ETFs, including:

- SPY (US large cap)
- QQQ (US growth/tech)
- IWM (US small cap)
- TLT (long-duration Treasuries)
- GLD (gold)
- EFA (international equities)

Data are retrieved using the `yfinance` API. Log returns are computed and annualized for model inputs.

---

## 5. Implementation Architecture

All modeling logic is implemented in Python functions and classes, and the analysis is documented in a single comprehensive Jupyter notebook (`analysis.ipynb`).

---

## 6. Empirical Analysis

The project conducts rigorous empirical tests, including:

- Diagnostic evaluation of sample vs shrunk covariance
- Eigenvalue spectrum and condition number comparison
- Rolling-window re-estimation of portfolio weights
- Out-of-sample performance backtest
- Comparison of unconstrained vs long-only portfolios
- Concentration (HHI, effective number of positions)
- Turnover (weight change across periods)

These analyses quantify the stability and economic performance of optimized portfolios.

---

## 7. Assumptions

Key modeling assumptions include:

- Returns are approximately i.i.d. over estimation windows  
- Daily returns can be annualized by scaling (252 trading days)  
- Ledoit–Wolf shrinkage target is the scaled identity  
- No transaction costs are considered  
- Rebalancing is monthly in backtests

These assumptions are standard in portfolio research but should be revisited for real trading systems.

---

## 8. Computational Considerations

- Quadratic programming (QP) with long-only constraints is solved via `cvxpy` with OSQP.
- Ledoit–Wolf shrinkage is computed with `scikit-learn`.
- Computation scales with the number of assets and length of rolling windows; performance is sufficient for portfolios with up to ~100 assets.
- Handling covariance near singularity is addressed with shrinkage and numerical jitter.

---

## 9. Results and Interpretation

Empirical results typically show:

- **Sample covariance** often has high condition numbers and small eigenvalues, leading to unstable weights.
- **Long-only constrained portfolios** produce more interpretable allocations with fewer shorts.
- **Ledoit–Wolf shrinkage** reduces condition number and stabilizes weights.
- **Out-of-sample performance** reveals that naive in-sample optimization can produce misleading Sharpe ratios.
- **Concentration and turnover metrics** confirm that shrinkage and constraints improve diversification and reduce excessive trading.

Charts and tables in the notebook illustrate these findings clearly.

---

## 10. Conclusion

This project demonstrates that classical mean–variance optimization, while elegant, has practical limitations due to estimation error and ill-conditioned covariance matrices. With realistic constraints and statistical shrinkage, portfolio allocations become more stable and robust. The methodology and analysis provide a strong foundation for both academic understanding and practical implementation of portfolio optimization.

---

## Reproducibility

To reproduce results:

1. Clone the repository.
2. Install dependencies:
pip install -r requirements.txt
3. Run `analysis.ipynb` end to end.
jupyter notebook notebooks/analysis.ipynb
5. All data is fetched dynamically via `yfinance`; no external data downloads are required.


