The pipeline takes hourly close prices, transforms them into strategy signals, turns those into portfolio weights, simulates trading with costs, and then computes annualized performance metrics – every step can be written as explicit formulas.

***

## Data and returns

Let \(P_t^i\) be the close price of asset \(i\) at time \(t\), and collect them in a price matrix \(P \in \mathbb{R}^{T \times N}\) (rows = times, columns = symbols).
Simple asset returns are  
\[
r_t^i = \frac{P_t^i - P_{t-1}^i}{P_{t-1}^i}
\]
implemented as \(r_t^i = \text{pct\_change}(P^i)_t\).

For forward “targets” with horizon \(f\) (e.g. \(f=1\) hour or day), the target for time \(t\) is  
\[
y_t^i = r_{t+f}^i
\]
which is obtained by shifting the return series backward by \(f\) steps.
In backtesting, the realized portfolio return at time \(t\) is always computed using past weights:  
\[
R_t^{\text{port}} = \sum_{i=1}^N w_{t-1}^i\, r_t^i.
\] 

***

## Signal mathematics

All three strategies work per asset and per time, using only past information (via shifts).

- **Momentum (lookback \(L\))**

  For each asset,
  \[
  s^{\text{mom}}_t{}^i
  = \frac{P_{t}^i - P_{t-L}^i}{P_{t-L}^i}
  \]
  implemented as \(\text{pct\_change}(P^i, \text{periods}=L)\), then shifted by one step so the signal at \(t\) only uses data up to \(t-1\).

- **Mean reversion (rolling z‑score of returns, window \(L\))**

  First compute one‑step returns \(r_t^i\).
  Rolling mean and standard deviation:
  \[
  \mu_t^i = \frac{1}{L}\sum_{k=0}^{L-1} r_{t-k}^i,
  \quad
  \sigma_t^i = \sqrt{\frac{1}{L-1}\sum_{k=0}^{L-1} (r_{t-k}^i - \mu_t^i)^2}.
  \]
  The z‑score and signal are
  \[
  z_t^i = \frac{r_t^i - \mu_t^i}{\sigma_t^i},\quad
  s^{\text{mr}}_t{}^i = -\,z_{t-1}^i,
  \]
  so assets with unusually low recent returns (negative \(z\)) get positive mean‑reversion signals.

- **EWMA crossover (fast span \(F\), slow span \(S\), volatility window \(W\))**

  Define fast and slow exponential moving averages
  \[
  \text{EMA}^{\text{fast}}_t{}^i = \text{EWMA}_F(P^i)_t,
  \quad
  \text{EMA}^{\text{slow}}_t{}^i = \text{EWMA}_S(P^i)_t,
  \]
  and a rolling standard deviation of price
  \[
  \sigma^{\text{EWMA}}_t{}^i
  = \sqrt{\frac{1}{W-1}\sum_{k=0}^{W-1}\bigl(P_{t-k}^i-\bar P_t^i\bigr)^2},
  \]
  where \(\bar P_t^i\) is the rolling mean over the same window.
  The raw signal is
  \[
  s^{\text{ewma}}_t{}^i
  = \frac{\text{EMA}^{\text{fast}}_t{}^i - \text{EMA}^{\text{slow}}_t{}^i}
         {\sigma^{\text{EWMA}}_t{}^i},
  \]
  then shifted by one step so \(s^{\text{ewma}}_t{}^i = s^{\text{ewma}}_{t-1,\text{raw}}{}^i\).

These strategy functions are applied column‑wise to the price matrix, producing three signal matrices \(S^{\text{mom}}, S^{\text{mr}}, S^{\text{ewma}} \in \mathbb{R}^{T \times N}\).

***

## Z‑scoring and filtering

At any time \(t\), the predictor converts raw signals for all assets into cross‑sectional z‑scores to put them on a comparable scale.
Given raw signals \(s_t^i\) for \(i=1,\dots,N\), define
\[
\bar s_t = \frac{1}{N}\sum_{i=1}^N s_t^i,
\quad
\sigma_t = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (s_t^i - \bar s_t)^2},
\]
and the **unfiltered** z‑score
\[
z_t^i = \frac{s_t^i - \bar s_t}{\sigma_t},
\]
with \(\sigma_t\) set to NaN when zero and later filled with 0 after division.

Filtering uses cross‑sectional quantiles at each time \(t\).

- Let \(q_t^{\text{long}}\) be the \(\text{top\_q}\) quantile (e.g. 0.8) of \(\{z_t^i\}_i\), and \(q_t^{\text{short}}\) the \(\text{bottom\_q}\) quantile (e.g. 0.2).
- The filtered signal \(f_t^i\) is
  \[
  f_t^i =
  \begin{cases}
  z_t^i & \text{if } z_t^i \ge q_t^{\text{long}} \quad\text{(candidate long)}\\
  -z_t^i & \text{if long\_short and } z_t^i \le q_t^{\text{short}} \quad\text{(candidate short, continuous mode)}\\
  0 & \text{otherwise,}
  \end{cases}
  \]
  or \(+1/-1/0\) if “discrete” mode is enabled.

Re‑z‑scoring is then done only on active (non‑zero) names at each time \(t\).

- Let \(A_t = \{i : f_t^i \ne 0\}\).  
- Compute mean and standard deviation on this active set,
  \[
  \bar f_t = \frac{1}{|A_t|}\sum_{i\in A_t} f_t^i,
  \quad
  \sigma^{\text{act}}_t = \sqrt{\frac{1}{|A_t|-1}\sum_{i\in A_t}(f_t^i-\bar f_t)^2},
  \]
  and set
  \[
  \tilde f_t^i =
  \begin{cases}
  \dfrac{f_t^i - \bar f_t}{\sigma^{\text{act}}_t} & i\in A_t,\ \sigma^{\text{act}}_t>0,\\[4pt]
  0 & \text{otherwise.}
  \end{cases}
  \] [attached_file:1]

For a **single strategy**, \(\tilde f_t^i\) is the final signal used as weights in its own backtest. [attached_file:1]  
For the **multi‑strategy portfolio**, each strategy provides its **unfiltered** z‑scores \(z_{k,t}^i\), which are combined before a single round of filtering and re‑z‑scoring. [attached_file:1]

***

## Portfolio combination and backtest

Suppose there are \(K\) predictors (strategies) with allocation weights \(a_k \ge 0\) and \(\sum_k a_k = 1\) after normalization. [attached_file:1]  
Let \(z_{k,t}^i\) be the unfiltered z‑score from predictor \(k\) for asset \(i\) at time \(t\); the combined unfiltered signal is
\[
u_t^i = \sum_{k=1}^K a_k\, z_{k,t}^i.
\] [attached_file:1]

This matrix \(U = (u_t^i)\) is passed through the same quantile filter and active‑only re‑z‑scoring as above, yielding final portfolio weights \(w_t^i\). [attached_file:1]  
Volatility targeting is currently disabled, so the final weight matrix is simply \(W = (\tilde f_t^i)\). [attached_file:1]

The backtester enforces a discrete rebalancing schedule: if the rebalance frequency is \(R\) periods, target weights are updated every \(R\) steps and forward‑filled in between. [attached_file:1]

- Let \(w_t^i\) be the target weights after this schedule. [attached_file:1]
- Trading convention: weights chosen at close of \(t-1\) apply to returns from \(t-1\) to \(t\), so backtest uses
  \[
  \tilde w_t^i = w_{t-1}^i
  \quad\Rightarrow\quad
  R_t^{\text{port}} = \sum_i \tilde w_t^i\, r_t^i.
  \] [attached_file:1]

Turnover and costs per period \(t\) are
\[
\text{turnover}_t = \frac{1}{2}\sum_{i=1}^N |\tilde w_t^i - \tilde w_{t-1}^i|,
\quad
c_t = \text{turnover}_t\,(c_{\text{tc}} + c_{\text{slip}}),
\]
where \(c_{\text{tc}}\) and \(c_{\text{slip}}\) are transaction‑cost and slippage rates in decimal (e.g. \(10\) bps \(\Rightarrow 0.001\)). [attached_file:1]  
Net portfolio return and cumulative return are
\[
\hat R_t = R_t^{\text{port}} - c_t,
\quad
C_t = \prod_{u\le t}(1 + \hat R_u).
\] [attached_file:1]

***

## Performance metrics

The metrics module turns the net return series \(\{\hat R_t\}_{t=1}^T\) into annualized statistics, assuming \(N_{\text{year}} = 8760\) periods per year for hourly data. [attached_file:1]

- **Annualized return**

  Let total compounded return be
  \[
  R_{\text{total}} = \prod_{t=1}^T (1 + \hat R_t) - 1.
  \]
  With \(T\) periods, the number of years is \(Y = T / N_{\text{year}}\), and
  \[
  R_{\text{ann}} = (1 + R_{\text{total}})^{1/Y} - 1.
  \] [attached_file:1]

- **Annualized volatility and Sharpe**

  \[
  \sigma_{\text{ann}} = \text{std}(\hat R_t)\,\sqrt{N_{\text{year}}},
  \quad
  \text{Sharpe} = \frac{R_{\text{ann}} - r_f}{\sigma_{\text{ann}}},
  \]
  with \(r_f\) the annual risk‑free rate (set to 0 by default). [attached_file:1]

- **Max drawdown and Calmar**

  From cumulative curve \(C_t\), define running peak \(M_t = \max_{u\le t} C_u\) and drawdown
  \[
  D_t = \frac{C_t - M_t}{M_t}.
  \]
  The maximum drawdown is \(\min_t D_t\), and the Calmar ratio is
  \[
  \text{Calmar} = \frac{R_{\text{ann}}}{| \min_t D_t |}.
  \] [attached_file:1]

- **Win rate and profit factor**

  Win rate is
  \[
  \text{WinRate} = \frac{\#\{t : \hat R_t > 0\}}{T},
  \]
  while profit factor is
  \[
  \text{PF} = \frac{\sum_{\hat R_t>0} \hat R_t}
                   {\left|\sum_{\hat R_t<0} \hat R_t\right|},
  \]
  with edge cases handled when gains or losses are zero. [attached_file:1]

