# BC3 — Portfolio Replica: "Cracking the Black Box"

## Project Overview

This module (PoliMI Fintech, Module #3) tackles **portfolio replication under information asymmetry**: given only the published return stream of a "secret" portfolio, infer a combination of liquid, observable instruments that tracks it.

The notebook (`Portfolio_ReplicaPoliMIv2.ipynb`) builds a self-constructed **"Monster Index"** as its black-box target — deliberately made hard to replicate — and then trains a regularized linear model over a rolling-window backtest, subject to a regulatory Value-at-Risk constraint.

**Target (Monster Index):**

| Component                      | Weight |
| ------------------------------ | :----: |
| HFRX Global Hedge Fund         |  50%   |
| MSCI World (Developed Equity)  |  25%   |
| Bloomberg Global Aggregate Bond|  25%   |

**Replication instruments:** 11 Bloomberg futures (equity, bond, commodity, FX — see Dataset below).

**Risk constraint:** Gaussian Value-at-Risk at 1% confidence over a 1-month horizon must stay at or below 8% (a prudential proxy for the UCITS cap of 20%). If exceeded, weights are rescaled down.

---

## Dataset Description

- **Source:** Bloomberg (`Dataset3_PortfolioReplicaStrategy.xlsx`).
- **Period:** 23 Oct 2007 – April 2021 (≈705 weekly observations, local currency).
- **Frequency:** Weekly.
- **Layout (local file):** Row 1 = ticker symbols; first column = date; rows 2+ = closing levels.

### Target Indices

| Ticker     | Description                                    |
| ---------- | ---------------------------------------------- |
| HFRXGL     | HFRX Global Hedge Fund Index                   |
| MXWO       | MSCI World (Developed)                         |
| MXWD       | MSCI World All Country (Developed + Emerging)  |
| LEGATRUU   | Bloomberg Global Aggregate Bond                |

### Futures Universe

| Ticker | Instrument                          |
| ------ | ----------------------------------- |
| RX1    | Bund (10Y German Govt)              |
| TY1    | 10Y US Treasury                     |
| DU1    | Schatz (2Y German Govt)             |
| TU2    | 2Y US Treasury                      |
| ES1    | S&P 500 E-mini                      |
| NQ1    | Nasdaq 100                          |
| VG1    | Eurostoxx 50                        |
| TP1    | Topix (Japanese Equity)             |
| LLL1   | MSCI Emerging Markets               |
| GC1    | Gold                                |
| CO1    | Brent Crude Oil                     |

---

## Implemented Pipeline

The 33-cell notebook implements an end-to-end replication pipeline. Cell ranges below refer to the on-disk notebook after execution.

| Step | Cells | Description |
| ---- | ----- | ----------- |
| 1. **Data ingestion** | 1–2 | Load the Excel file, attach `' Index'` / `' Comdty'` suffixes to tickers, index by weekly date, and build a `variable_info` label dictionary. |
| 2. **EDA on target indices** | 3–4 | Rebased historical price lines (base 100) for `MXWO`, `MXWD`, `LEGATRUU`, `HFRXGL`; annualized return, volatility, Sharpe, max drawdown, skewness, kurtosis. |
| 3. **Correlation & distribution** | 5–6 | Correlation heatmap of returns; histograms with KDE; cumulative-return curves; rolling 52-week correlation versus HFRX. |
| 4. **Monster Index construction** | 7–9 | Synthesize the 50/25/25 target from component weekly returns; compute per-futures correlation with the target; QQ plots vs. normal for the target and the top-3 correlated futures. |
| 5. **Time-series diagnostics** | 10–16 | ACF / PACF (lags ≤ 20); Ljung-Box tests at lags 5/10/15/20; ACF of squared and absolute returns (volatility clustering); rolling correlation with top-5 futures. |
| 6. **Elastic Net regressor** | 18–23 | Rolling-window backtest with MinMax-normalized features and target, coefficients rescaled back to original units; per-step Gaussian 1-month 1% VaR with 8% cap and proportional rescaling. |
| 7. **Grid search** | 24–26 | 54 configurations: `l1_ratio ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}` × `rolling_window ∈ {52, 104, 156}` × `alpha ∈ {1e-4, 1e-3, 1e-2}`. Ranked by out-of-sample information ratio. |
| 8. **Best-config diagnostics** | 27–30 | Cumulative returns (target vs. replica), drawdowns, gross-exposure path, VaR path, scaling-factor path, top-10 weights trajectory, weekly-returns scatter, rolling 26-week correlation. |
| 9. **HINTs** (not implemented) | 31–32 | Transaction costs, OLS / Lasso / Ridge benchmarks, alternative rebalancing cadences, futures pre-selection, Kalman Filter, neural-network weight-generator. Tracked in `TODO.md`. |

### Key modelling choices

- **Linear regression + regularization** rather than a flexible predictor: the fitted coefficients are *portfolio weights*, so interpretability is non-negotiable.
- **MinMax normalization on X and y** before the penalized fit — scale invariance matters for Lasso / Elastic Net. Coefficients are rescaled by `1 / scaler_X.scale_` before use.
- **No intercept** (`fit_intercept=False`): alphas should explain the target without a free cash term.
- **Rolling walk-forward backtest**, 1-week horizon: avoids look-ahead and respects the non-stationarity of financial data.

---

## Elastic Net Backtest — Key Results

Grid search ranks **54** configurations by out-of-sample information ratio. The top-3 (full top-10 is in cell 26):

| Rank | `l1_ratio` | `rolling_window` | `alpha` | IR     | Correlation | Tracking Error | Avg Gross Exposure | Avg VaR |
| :--: | :-----: | :----------: | :---: | :----: | :---------: | :------------: | :----------------: | :-----: |
| 1    | 0.0 (Ridge) | 156 wk (3Y) | 0.010 | -0.257 | 0.724       | 3.86%          | 0.216              | 2.11%   |
| 2    | 0.2     | 156 wk       | 0.010 | -0.336 | 0.759       | 3.79%          | 0.200              | 1.83%   |
| 3    | 0.0     | 156 wk       | 0.001 | -0.339 | 0.780       | 3.62%          | 0.224              | 1.95%   |

Observation: the best configurations all favour the **longest rolling window** (3 years) — a stable signal matters more than recent adaptation on this weekly data set.

### Best configuration — full metrics

> `l1_ratio = 0.0` (pure Ridge), `rolling_window = 156` weeks, `alpha = 0.01`

| Metric                  | Target | Replica |
| ----------------------- | :----: | :-----: |
| Annualized return       | 3.78%  | 2.79%   |
| Annualized volatility   | 5.55%  | 3.52%   |
| Sharpe ratio            | 0.68   | 0.79    |
| Max Drawdown            | 5.56%  | 5.56%   |
| Tracking Error          |   –    | 3.86%   |
| Information Ratio       |   –    | -0.26   |
| Correlation             |   –    | 0.7235  |
| Average Gross Exposure  |   –    | 0.2157  |
| Average VaR (1%, 1M)    |   –    | 2.11%   |

**Reading the numbers.** The replica tracks the Monster Index with correlation ~0.72 while staying well below the VaR cap (2.11% vs. an 8% limit). The negative information ratio reflects a ~1 pp/y return shortfall against the target: the VaR rescaling and the sparsity of the 11-futures universe cost some upside. A higher Sharpe than the target (0.79 vs. 0.68) comes from lower replica volatility, not alpha.

These are the *uncovered* baselines — none of the HINTs extensions (transaction costs, Kalman filter, NN) have been incorporated yet.

---

## File Organization

```
BC3/
├── README.md                              # this file
├── TODO.md                                # HINTs to-do list
├── Portfolio_ReplicaPoliMIv2.ipynb        # full notebook
├── Dataset3_PortfolioReplicaStrategy.xlsx # weekly Bloomberg data
└── Zenti_Business_Case_3.pdf              # original assignment brief
```

### How to re-run the notebook

```bash
# from repo root; assumes the project uv/venv at .venv
cd BC3
../.venv/bin/jupyter nbconvert --to notebook --execute Portfolio_ReplicaPoliMIv2.ipynb --inplace
```

The loader prefers the local Excel file next to the notebook and falls back to the Colab Drive path only when the local file is missing.

> Extra dependency: `statsmodels` (ACF/PACF, Ljung-Box). Install with `.venv/bin/python -m pip install statsmodels` if not already present.

---

## See Also

- [`TODO.md`](TODO.md) — extensions drawn from the notebook's HINTs cells.
- [Root `README.md`](../README.md) — theoretical framework, regulatory context (UCITS/MIFID), and literature references.
- [`Zenti_Business_Case_3.pdf`](Zenti_Business_Case_3.pdf) — original assignment brief.
