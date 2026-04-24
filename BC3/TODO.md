# BC3 — Portfolio Replica: Extensions To-Do (HINTs)

The notebook's final two cells list extensions that are **not yet implemented**. They are split here into **five balanced tracks**, one per group member, each ending in a self-contained deliverable that plugs into a final consolidated comparison.

> Baseline reference: the Elastic Net rolling backtest in `Portfolio_ReplicaPoliMIv2.ipynb` (cells 22–30). All tracks share the same target index
>
> $$
> r^{\text{target}}_t = 0.50 \cdot r^{\text{HFRX}}_t + 0.25 \cdot r^{\text{MSCI World}}_t + 0.25 \cdot r^{\text{Global Bond}}_t
> $$
>
> and the same reporting metric set so results are directly comparable.

---

## Team allocation at a glance

| Member | Track | Theme | Shipped deliverable |
| :----: | :---- | :---- | :----- |
| **M1** | [Harness + Transaction Costs](#member-1--harness--transaction-costs) | Frictionless → cost-aware backtest | Reusable `run_rolling_backtest()` harness + net-of-cost report |
| **M2** | [Classical Linear Bench](#member-2--classical-linear-benchmark) | OLS / Ridge / Lasso / Huber + pre-selection | Linear-model comparison grid |
| **M3** | [Rebalancing & Portfolio Constraints](#member-3--rebalancing--portfolio-constraints) | Trading cadences, leverage caps, stress windows | Constraint-aware B6 table + crisis-window evaluation |
| **M4** | [Kalman Filter / State-Space](#member-4--kalman-filter--state-space) | Dynamic weights via state-space model | Kalman baseline + EM noise tuning |
| **M5** | [Deep Learning + Synthesis](#member-5--deep-learning--synthesis) | NN weight generator + final writeup | Meta-model + consolidated comparison + presentation |

**Shared contract** (everyone agrees on this before splitting):

- **Inputs:** `X` = 11 futures weekly returns, `y` = target Monster Index weekly return.
- **Output of every model:** a `weights_history : DataFrame[T × 11]` (rows = rebalance dates, cols = futures).
- **Output of the harness:** `replica_returns : Series[T]` and a metrics dict `{IR, TE, ρ, GE, VaR, turnover, net_IR, net_TE}`.
- **Persistence:** each member commits a `results/<track>.pkl` with the weights + metrics so M5 can load them into the final comparison without re-running.

---

## Member 1 — Harness + Transaction Costs

> **Role.** Infrastructure owner. Ships the shared harness that M2–M5 call into, and the transaction-cost accounting that every model is re-scored against.

### M1.1 Refactor the rolling backtest

- [ ] Extract `run_elastic_net_normalized` into `run_rolling_backtest(model_factory, X, y, rolling_window, rebalance_every=1, cost_bps=0.0, var_cap=0.08)`.
- [ ] `model_factory` must accept `(X_train, y_train)` and return an object with `.predict_weights() → np.ndarray[n_features]`. OLS, Ridge, Lasso, Elastic Net, Kalman, and NN all conform to this contract.

### M1.2 Per-trade transaction costs

- [ ] Subtract the cost term at every rebalance date $t$:

  $$
  c_t = \tau \cdot \sum_{j=1}^{n} \lvert w_{t,j} - w_{t-1,j} \rvert, \qquad \tau = 0.0005
  $$

  with the first-rebalance term $c_0 = \tau \cdot \sum_{j} \lvert w_{0,j} \rvert$ (entering from cash).

### M1.3 Net-of-cost reporting

- [ ] Add a second results table alongside the gross table showing net annualized return, Sharpe, $\operatorname{TE}$, and $\operatorname{IR}$. Include the average weekly turnover and the annualized cost drag:

  $$
  \bar{T} = \mathbb{E}_t \!\left[\, \sum_{j} \lvert \Delta w_{t,j} \rvert \,\right], \qquad \text{drag} = 52 \cdot \tau \cdot \bar{T}
  $$

### M1.4 Cost sensitivity sweep

- [ ] Re-run M1.2/M1.3 with $\tau \in \{0.0002,\, 0.0005,\, 0.0010\}$ and plot $\operatorname{IR}(\tau)$ for the Elastic Net baseline.

### M1.5 Extension — Turnover-penalised optimisation

- [ ] Add a turnover term directly inside the optimisation, not just in post-scoring:

  $$
  \hat{\beta}_t = \arg\min_{\beta} \; \tfrac{1}{2N}\lVert y - X\beta \rVert_2^{2} + \alpha\!\left(\rho\lVert\beta\rVert_1 + (1-\rho)\lVert\beta\rVert_2^{2}\right) + \gamma\lVert \beta - \hat{\beta}_{t-1} \rVert_1
  $$

  Tune $\gamma$ so that net $\operatorname{IR}$ is maximised and compare with the post-hoc cost subtraction from M1.2.

**Done when:** M2–M5 can call the harness without touching its internals, and the net-of-cost table + $\operatorname{IR}(\tau)$ plot live in the notebook.

---

## Member 2 — Classical Linear Benchmark

> **Role.** Build the full linear-model leaderboard. Each sweep reuses M1's harness unchanged.

### M2.1 OLS baseline

- [ ] `LinearRegression(fit_intercept=False)` on MinMax-normalised features, coefficients rescaled back.

### M2.2 Pure Lasso sweep

- [ ] `Lasso(alpha=`$\alpha$`)` over $\alpha \in \{10^{-5},\, 10^{-4},\, 5\!\cdot\!10^{-4},\, 10^{-3},\, 5\!\cdot\!10^{-3},\, 10^{-2}\}$.
- [ ] Plot sparsity $s(\alpha) = \lvert\{j : \beta_j \neq 0\}\rvert$ versus tracking error $\operatorname{TE}(\alpha)$.

### M2.3 Pure Ridge sweep

- [ ] `Ridge(alpha=`$\alpha$`)` over the same grid; compare $\operatorname{TE}$ and average gross exposure versus Elastic Net at matched $\alpha$.

### M2.4 Top-$K$ futures pre-selection

- [ ] At each rebalance, keep only the $K$ futures with highest $\lvert \operatorname{corr}(r_j, y) \rvert$ inside the training window; run the best linear model on the reduced set.
- [ ] Ablation $K \in \{3, 5, 7, 11\}$; plot $\operatorname{TE}(K)$ and $\bar{T}(K)$.

### M2.5 Extension — Huber-robust regression

- [ ] Replace the squared loss with the Huber loss to test robustness under crisis tails:

  $$
  \mathcal{L}_\delta(\beta) = \sum_t \rho_\delta\!\bigl( y_t - x_t^{\top}\beta \bigr), \qquad
  \rho_\delta(u) =
  \begin{cases}
    \tfrac{1}{2} u^{2} & \lvert u \rvert \leq \delta, \\
    \delta\!\left(\lvert u \rvert - \tfrac{1}{2}\delta\right) & \lvert u \rvert > \delta.
  \end{cases}
  $$

  Use `sklearn.linear_model.HuberRegressor`; tune $\delta \in \{1.0, 1.35, 2.0\}$.

**Done when:** a single table {OLS, Ridge, Lasso, Elastic Net, Huber} × {all 11, top-5} with $\operatorname{IR}$, $\operatorname{TE}$, $\rho$, $\overline{\operatorname{GE}}$, $\bar{T}$, ranked by net $\operatorname{IR}$.

---

## Member 3 — Rebalancing & Portfolio Constraints

> **Role.** Make the baseline realistic: trading cadence, leverage limits, and crisis-period evaluation.

### M3.1 Lower rebalancing frequency

- [ ] Hold weights between rebalances for $\Delta t \in \{1, 4, 12\}$ weeks: re-fit only on rebalance dates, in-between weeks apply the previously fit weights to the current futures returns,

  $$
  r^{\text{replica}}_t = w_{t^{\star}}^{\top}\, r_t, \qquad t^{\star} = \Delta t \cdot \lfloor t / \Delta t \rfloor.
  $$

### M3.2 Gross-exposure / leverage cap

- [ ] Add a projection onto the UCITS-like feasible set after each fit:

  $$
  w_t \mapsto \Pi_{\mathcal{W}}(w_t), \qquad
  \mathcal{W} = \Bigl\{ w : \sum_j \lvert w_j \rvert \leq L \Bigr\}, \qquad L \in \{1.0,\, 1.5,\, 2.0\}.
  $$

  Report how $\operatorname{IR}$ and $\operatorname{TE}$ trade off against the cap.

### M3.3 Long-only variant

- [ ] Implement a long-only Ridge (`sklearn.linear_model.Ridge` + non-negativity via `scipy.optimize.nnls` post-fit, or use `cvxpy`). Report the tracking-error penalty paid for removing shorts.

### M3.4 Stress-window evaluation

- [ ] Re-score every model (yours and the baselines) on two isolated sub-samples:
  - **2008 crisis:** 2008-09-01 → 2009-06-30.
  - **COVID crash:** 2020-02-15 → 2020-06-30.
- [ ] Report per-window $\operatorname{TE}$, max drawdown, correlation, and $\operatorname{VaR}_{\text{empirical}}$ breaches.

### M3.5 Consolidated comparison (B6 in the original plan)

- [ ] Build the comparison table: rows = $\{\text{OLS},\, \text{Ridge},\, \text{Lasso},\, \text{ElasticNet},\, \text{Huber}\}$ × $\{1\text{w},\, 4\text{w},\, 12\text{w}\}$ × $\{\text{no cap},\, L = 1.5,\, L = 2.0\}$; columns = $\{\operatorname{IR}, \operatorname{TE}, \rho, \overline{\operatorname{GE}}, \bar{T}\}$.

**Done when:** the table is published + a "Findings" markdown cell explains which configuration survives the two stress windows best.

---

## Member 4 — Kalman Filter / State-Space

> **Role.** Replace the fixed rolling window with a dynamic weight model.

State-space form specified in the notebook's HINTs cell:

$$
\begin{aligned}
x_t &= A\, x_{t-1} + B\, u_t, \qquad & u_t &\sim \mathcal{N}(0,\, I)     && \text{(state: portfolio weights, random walk)} \\
y_t &= C_t\, x_t + D\, \varepsilon_t, \qquad & \varepsilon_t &\sim \mathcal{N}(0,\, 1) && \text{(observation: target return)}
\end{aligned}
$$

### M4.1 Scaffolding

- [ ] Install/verify `pykalman` or hand-roll a filter. Dimensions: state $\dim(x_t) = 11$, observation $\dim(y_t) = 1$.

### M4.2 Parameter choices

- [ ] $A = I$, $B = \sigma_w \cdot I$, $C_t = r_t^{\top}$, $D = \hat\sigma_y$ over the training window.
- [ ] Grid: $\sigma_w \in \{10^{-4},\, 10^{-3},\, 10^{-2}\}$.

### M4.3 Initialisation

- [ ] Ridge regression on the first training window gives $x_0$ and $P_0 = \lambda_0 \cdot I$.

### M4.4 Filter recursion

- [ ] Propagate the filter forward through the full sample:

  $$
  \begin{aligned}
  \hat{x}_{t \mid t-1} &= A\, \hat{x}_{t-1 \mid t-1}, \\
  P_{t \mid t-1}       &= A\, P_{t-1 \mid t-1}\, A^{\top} + B B^{\top}, \\
  K_t                  &= P_{t \mid t-1}\, C_t^{\top} \bigl( C_t\, P_{t \mid t-1}\, C_t^{\top} + D^{2} \bigr)^{-1}, \\
  \hat{x}_{t \mid t}   &= \hat{x}_{t \mid t-1} + K_t \bigl( y_t - C_t\, \hat{x}_{t \mid t-1} \bigr).
  \end{aligned}
  $$

### M4.5 VaR guardrail

- [ ] Apply $\operatorname{VaR}_{1\%,\,1\text{M}}(\hat x_t) \leq 8\%$ scaling before computing replica returns (same rule as the Elastic Net pipeline).

### M4.6 Extension — EM for the noise scalars

- [ ] Replace the $\sigma_w$ grid with EM estimation of $(B, D)$ using `pykalman.KalmanFilter.em()`. Compare EM-tuned against grid-tuned on $\operatorname{IR}$ and CPU time.

### M4.7 Extension (bonus) — Regime-switching KF

- [ ] Fit a two-state HMM on realised target volatility to classify weeks as *calm* / *stressed*; run two Kalman filters with different $\sigma_w$ and switch between them by the HMM posterior.

**Done when:** Kalman (+ EM and regime variants) rows appear in the consolidated comparison with matched metrics.

---

## Member 5 — Deep Learning + Synthesis

> **Role.** End-to-end weight-generating network **and** owner of the final consolidated report. This member integrates everyone else's artefacts.

End-to-end: the network $f_\theta$ *outputs the weights*, not the returns:

$$
w_t = f_\theta(\phi_t), \qquad r^{\text{replica}}_t = w_t^{\top} r_t.
$$

### M5.1 Feature engineering

- [ ] Build $\phi_t$ from trailing $\{4, 12, 52\}$-week futures returns, trailing realised volatility, interest-rate / MSCI World regime indicators, and the last Elastic Net weight $w^{\text{EN}}_{t-1}$ as a warm start.

### M5.2 Architecture

- [ ] Small MLP: $\phi_t \in \mathbb{R}^{d_\phi} \to 64 \to 32 \to w_t \in \mathbb{R}^{11}$ (no softmax — long/short allowed).
- [ ] Optional final projection $\Pi_{\mathcal{W}}$ onto the VaR feasible set (reusing M3.2's projector).

### M5.3 Loss

- [ ] Tracking-error volatility plus an optional turnover penalty:

  $$
  \mathcal{L}(\theta) = \operatorname{Var}_{t:\,t+H}\!\bigl( w_t^{\top} r_s - y_s \bigr) + \lambda \lVert \Delta w_t \rVert_1, \qquad H = 12,\; \lambda \in \{0,\, 10^{-3},\, 10^{-2}\}.
  $$

### M5.4 Training regime

- [ ] Rolling train / out-of-sample predict with retrain cadence $\Delta t \in \{1, 4\}$ weeks (reuse M3.1's harness).
- [ ] Device: `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`; early stopping on validation $\operatorname{TE}$.

### M5.5 Extension — PCA-compressed or attention-based features

- [ ] **PCA variant:** compute top-$k$ principal components of the trailing futures-returns window and feed the score vector into the MLP.
- [ ] **Attention variant (optional, stretch):** a small transformer that attends over the trailing 52-week feature matrix $\Phi_t \in \mathbb{R}^{52 \times n}$ instead of a flattened vector.

### M5.6 Final consolidated comparison

- [ ] Load each member's `results/<track>.pkl` and produce one master table:

  $$
  \{\text{OLS}, \text{Ridge}, \text{Lasso}, \text{ElasticNet}, \text{Huber}, \text{Kalman}, \text{KF+EM}, \text{NN}, \text{NN+Attn}\}\times\{\text{gross},\, \text{net-of-cost}\}
  $$

- [ ] Plots for the best model of each family: cumulative returns (target vs. replica), rolling correlation, weights heatmap over time.
- [ ] "Findings" markdown cell: which model wins on $\operatorname{IR}$, which wins on turnover, where the ranking flips when $\tau > \tau^{\star}$, and survival in the 2008/2020 stress windows.

### M5.7 Slides / report

- [ ] 10-slide deck summarising problem, method, leaderboard, stress-window result, recommended configuration.

**Done when:** the master table, final plots, Findings cell, and slide deck are committed.

---

## Metric definitions (shared across all tracks)

| Symbol | Definition |
| :----: | :--------- |
| $\operatorname{TE}$ | $\sqrt{52 \cdot \operatorname{Var}(r^{\text{replica}}_t - y_t)}$ — annualized tracking error |
| $\operatorname{IR}$ | $\dfrac{52 \cdot \mathbb{E}[r^{\text{replica}}_t - y_t]}{\operatorname{TE}}$ — information ratio |
| $\rho$ | $\operatorname{corr}(r^{\text{replica}}_t,\, y_t)$ |
| $\overline{\operatorname{GE}}$ | $\mathbb{E}_t\!\bigl[\, \sum_j \lvert w_{t,j} \rvert \,\bigr]$ — average gross exposure |
| $\bar T$ | $\mathbb{E}_t\!\bigl[\, \sum_j \lvert \Delta w_{t,j} \rvert \,\bigr]$ — average weekly turnover |
| $\operatorname{VaR}_{1\%,\,1\text{M}}$ | $-\Phi^{-1}(0.01) \cdot \hat\sigma_w \cdot \sqrt{4}$ — Gaussian 1-month 1% VaR |
| $c_t$ | $\tau \cdot \sum_j \lvert w_{t,j} - w_{t-1,j} \rvert$ — per-rebalance transaction cost |

---

## Milestone plan (suggested)

| Week | M1 | M2 | M3 | M4 | M5 |
| :--: | :- | :- | :- | :- | :- |
| 1 | Harness refactor | OLS + sparsity plots | Cadence sweep 1w/4w/12w | Scaffolding + Ridge init | Feature design + MLP skeleton |
| 2 | Cost accounting + $\operatorname{IR}(\tau)$ | Ridge + Lasso sweeps | Gross cap + long-only | KF recursion + VaR guard | Train loop + val metric |
| 3 | Turnover-penalised opt. | Huber + top-$K$ pre-selection | Stress windows (2008, 2020) | EM tuning of $(B, D)$ | Turnover penalty ablation |
| 4 | Final cost-drag table | Consolidated linear table | Consolidated B6 table | Regime-switching KF | PCA / attention variant |
| 5 | — | — | — | — | Master table + slides |

---

## References (from the notebook HINTs)

- Kalman filter intro — [MathWorks](https://it.mathworks.com/help/econ/what-is-the-kalman-filter.html)
- Replicating hedge funds with Kalman filters — [SSRN 1325190](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1325190)
