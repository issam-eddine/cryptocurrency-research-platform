# Cryptocurrency Research Platform <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [Overview](#overview)
- [How to run](#how-to-run)
  - [Try Online](#try-online)
  - [Run Locally](#run-locally)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
    - [Running the Application](#running-the-application)
- [Architecture](#architecture)
  - [Directory Structure](#directory-structure)
  - [Data Flow Pipeline](#data-flow-pipeline)
- [Three Core Strategies](#three-core-strategies)
  - [1. Momentum Strategy](#1-momentum-strategy)
    - [Philosophy](#philosophy)
    - [Signal Calculation](#signal-calculation)
    - [Signal Interpretation](#signal-interpretation)
    - [Portfolio Construction from Momentum Signals](#portfolio-construction-from-momentum-signals)
    - [Parameters](#parameters)
  - [2. Mean Reversion Strategy](#2-mean-reversion-strategy)
    - [Philosophy](#philosophy-1)
    - [Signal Calculation](#signal-calculation-1)
    - [Signal Interpretation](#signal-interpretation-1)
    - [Portfolio Construction](#portfolio-construction)
    - [Parameters](#parameters-1)
  - [3. EWMA Crossover Strategy](#3-ewma-crossover-strategy)
    - [Philosophy](#philosophy-2)
    - [Signal Calculation](#signal-calculation-2)
    - [Signal Interpretation](#signal-interpretation-2)
    - [Portfolio Construction](#portfolio-construction-1)
    - [Parameters](#parameters-2)
- [Mathematical Formulations](#mathematical-formulations)
  - [General Framework](#general-framework)
  - [Cross-Sectional Ranking](#cross-sectional-ranking)
  - [Portfolio Weight Construction](#portfolio-weight-construction)
- [Portfolio Construction](#portfolio-construction-2)
  - [Multi-Strategy Combination](#multi-strategy-combination)
  - [Gross vs Net Exposure](#gross-vs-net-exposure)
- [Cost Calculations](#cost-calculations)
  - [Transaction Costs](#transaction-costs)
  - [Turnover Calculation](#turnover-calculation)
  - [Cost Impact on Returns](#cost-impact-on-returns)
  - [Typical Cost Breakdown](#typical-cost-breakdown)
- [Returns Calculations](#returns-calculations)
  - [Gross Daily Returns](#gross-daily-returns)
  - [Cost-Adjusted Returns](#cost-adjusted-returns)
  - [Cumulative Returns](#cumulative-returns)
  - [Period Returns](#period-returns)
- [Backtesting Methodology](#backtesting-methodology)
  - [Core Backtesting Algorithm](#core-backtesting-algorithm)
  - [Implementation (Python)](#implementation-python)
  - [Weight Generation](#weight-generation)
  - [Return Computation with Costs](#return-computation-with-costs)
  - [Rebalancing Schedule](#rebalancing-schedule)
- [Performance Metrics](#performance-metrics)
  - [Annualized Return](#annualized-return)
  - [Annualized Volatility](#annualized-volatility)
  - [Sharpe Ratio](#sharpe-ratio)
  - [Maximum Drawdown](#maximum-drawdown)
  - [Information Ratio (Optional)](#information-ratio-optional)
- [Implementation Details](#implementation-details)
  - [Data Fetching \& Caching](#data-fetching--caching)
  - [Data Alignment](#data-alignment)
  - [Signal Computation](#signal-computation)
  - [Universe Selection](#universe-selection)
  - [Handling Edge Cases](#handling-edge-cases)
- [Configuration \& Parameters](#configuration--parameters)
  - [Interactive Controls (Streamlit UI)](#interactive-controls-streamlit-ui)
  - [Default Parameter Recommendations](#default-parameter-recommendations)
- [Advanced Topics](#advanced-topics)
  - [Correlation Analysis](#correlation-analysis)
  - [Portfolio Exposure Metrics](#portfolio-exposure-metrics)
  - [Turnover Analysis](#turnover-analysis)
- [Conclusion](#conclusion)

---

## Overview

The **Cryptocurrency Research Platform** is a backtesting and analysis framework for cryptocurrency trading strategies. It implements three complementary factor-based strategies:

1. **Momentum Strategy** - Captures trending behavior
2. **Mean Reversion Strategy** - Exploits price extremes
3. **EWMA Crossover Strategy** - Identifies trend changes via exponential moving average divergences

These strategies operate on a universe of top-performing cryptocurrencies (by trading volume) and can be combined into a diversified multi-strategy portfolio. The platform supports:

- **Live data fetching** from Binance via CCXT
- **Cross-sectional signal ranking** for portfolio construction
- **Dynamic rebalancing** with configurable frequencies
- **Transaction cost modeling** for realistic P&L simulation
- **Comprehensive performance analytics** including Sharpe ratios, maximum drawdowns, and correlation analysis

---

## How to run

### Try Online

Access the live application directly:

[https://cryptocurrency-research-platform.streamlit.app/](https://cryptocurrency-research-platform.streamlit.app/)

### Run Locally

#### Prerequisites

Ensure you have Python and `uv` installed on your system.

#### Installation Steps

1. **Clone the repository**
   ```
   git clone https://github.com/issam-eddine/cryptocurrency-research-platform.git
   cd cryptocurrency-research-platform
   ```

2. **Create a virtual environment**
   ```
   uv venv
   ```

3. **Activate the virtual environment**
   ```
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```
   uv sync
   ```

#### Running the Application

Each time you want to run the application:

```
source .venv/bin/activate
streamlit run streamlit/app.py
```

The application will start in your default browser at `http://localhost:8501`.

---

## Architecture

### Directory Structure
```
cryptocurrency-research-platform/
├── app/
│   └── streamlit_app.py              # Interactive Streamlit interface
├── src/
│   ├── backtest.py                   # Core backtesting engine
│   ├── data_fetch.py                 # Data retrieval from Binance
│   ├── factors.py                    # Factor/signal computations
│   ├── metrics.py                    # Performance metrics
│   └── preprocessing.py              # Data cleaning & alignment
├── requirements.txt                  # Python dependencies
└── pyproject.toml                    # Project configuration
```

### Data Flow Pipeline

```
Binance Exchange → CCXT API → Raw OHLCV Data → Caching (Parquet)
         ↓
    Preprocessing → Data Alignment → Price Matrix
         ↓
    Factor Computation → Signal Generation → Cross-sectional Ranking
         ↓
    Portfolio Construction → Backtesting Engine
         ↓
    Performance Metrics → Visualization & Analysis
```

---

## Three Core Strategies

### 1. Momentum Strategy

#### Philosophy
The momentum strategy is built on the **principle of trend persistence**: assets that have performed well recently tend to continue outperforming in the near term. This is a well-documented market anomaly across asset classes.

#### Signal Calculation

The momentum signal for asset $i$ at time $t$ is computed as:

$$ \text{MOM}_{i,t} = \frac{P_{i,t} - P_{i,t-L}}{P_{i,t-L}} $$

Where:
- $P_{i,t}$ = closing price of asset $i$ at time $t$
- $L$ = lookback period in days (default: 21 trading days)
- The signal is **shifted forward by 1 bar** to ensure no look-ahead bias

**Implementation** (Python):
```python
def momentum(close: pd.Series, lookback: int = 21) -> pd.Series:
    return close.pct_change(periods=lookback).shift(1)
```

#### Signal Interpretation

- **Positive signal** ($\text{MOM} > 0$): Asset has appreciated; rank it as a potential **long**
- **Negative signal** ($\text{MOM} < 0$): Asset has depreciated; rank it as a potential **short**
- **Magnitude**: Larger absolute values indicate stronger momentum

#### Portfolio Construction from Momentum Signals

1. **Cross-sectional ranking**: On rebalance dates, rank all assets by their momentum signal $(0, 1)$ percentile scale

2. **Quantile cutoffs**:
   - **Top quantile cutoff** $q_{\text{top}}$ (default: 0.80): Assets ranked above this percentile form the **long portfolio**
   - **Bottom quantile cutoff** $q_{\text{bottom}}$ (default: 0.20): Assets ranked below this percentile form the **short portfolio**

3. **Equal weighting**: Within each basket (long/short), assign equal weight:

   $w_{i,t}^{\text{long}} = \frac{1}{n_{\text{long}}} \quad \text{if asset } i \in \text{long basket}$

   $w_{i,t}^{\text{short}} = -\frac{1}{n_{\text{short}}} \quad \text{if asset } i \in \text{short basket}$

   $w_{i,t} = 0 \quad \text{otherwise}$

4. **Rebalancing**: Weights are held constant for a **rebalance period** (default: 21 days), then recomputed

#### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `momentum_lookback` | 21 | 5-120 | Days to look back for momentum calculation |
| `momentum_rebalance` | 21 | 1-63 | Days between rebalancing |
| `momentum_top_q` | 0.20 | 0.10-0.50 | Fraction of assets in long basket |
| `momentum_bottom_q` | 0.20 | 0.10-0.50 | Fraction of assets in short basket |

---

### 2. Mean Reversion Strategy

#### Philosophy
The mean reversion strategy exploits the **mean-reverting behavior** of asset returns: when returns deviate significantly from their recent historical average, they tend to revert back. This is particularly pronounced in cryptocurrency markets due to technical trading and liquidation cascades.

#### Signal Calculation

The mean reversion signal uses a **z-score** (standardized score) approach:

$$ r_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}} $$

$$ \mu_{i,t}^{(L)} = \frac{1}{L} \sum_{j=0}^{L-1} r_{i,t-j} $$

$$ \sigma_{i,t}^{(L)} = \sqrt{\frac{1}{L} \sum_{j=0}^{L-1} (r_{i,t-j} - \mu_{i,t}^{(L)})^2} $$

$$ Z_{i,t}^{\text{MR}} = \frac{r_{i,t} - \mu_{i,t}^{(L)}}{\sigma_{i,t}^{(L)}} $$

The **mean reversion signal** is the **negative** of the z-score:

$$ \text{MR}_{i,t} = -Z_{i,t}^{\text{MR}} $$

The negative sign reflects the mean reversion logic: when the z-score is **high** (positive, i.e., return is above average), we expect **mean reversion downward**, so we want a **negative signal** to short the asset.

Where:
- $r_{i,t}$ = daily return
- $\mu_{i,t}^{(L)}$ = rolling mean of returns over lookback window $L$ (default: 14 days)
- $\sigma_{i,t}^{(L)}$ = rolling standard deviation of returns
- The signal is **shifted forward by 1 bar** to prevent look-ahead bias

**Implementation** (Python):
```python
def mean_reversion_zscore(close: pd.Series, lookback: int = 21) -> pd.Series:
    ret = close.pct_change().fillna(0)
    mu = ret.rolling(lookback).mean()
    sigma = ret.rolling(lookback).std().replace(0, np.nan)
    z = (ret - mu) / sigma
    return -z.shift(1)  # Negative for mean-reversion
```

#### Signal Interpretation

- **Positive MR signal** (large negative z-score): Returns are significantly **below** historical mean → expect **upward reversion** → **long**
- **Negative MR signal** (large positive z-score): Returns are significantly **above** historical mean → expect **downward reversion** → **short**

#### Portfolio Construction

Identical to Momentum strategy:
- Rank assets by mean reversion signal
- Long the top quantile, short the bottom quantile
- Equal-weight within baskets
- Rebalance on fixed schedule

#### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `mr_lookback` | 14 | 5-120 | Days for rolling mean/std computation |
| `mr_rebalance` | 7 | 1-63 | Days between rebalancing |
| `mr_top_q` | 0.30 | 0.10-0.50 | Fraction of assets in long basket |
| `mr_bottom_q` | 0.30 | 0.10-0.50 | Fraction of assets in short basket |

---

### 3. EWMA Crossover Strategy

#### Philosophy
The EWMA (Exponentially Weighted Moving Average) crossover strategy is a **trend-following** approach that identifies trend shifts via the divergence between fast and slow exponential moving averages. When the fast EMA (recent prices) crosses above the slow EMA (historical average), it signals an **uptrend**. Conversely, when it crosses below, a **downtrend** is signaled.

#### Signal Calculation

**Step 1: Compute exponential moving averages**

The exponential moving average with span $\alpha$ is defined recursively as:

$$ \text{EMA}_{i,t}^{(\alpha)} = \lambda \cdot P_{i,t} + (1 - \lambda) \cdot \text{EMA}_{i,t-1}^{(\alpha)} $$

Where the smoothing factor is:

$$ \lambda = \frac{2}{\alpha + 1} $$

Fast EWMA uses $\alpha_{\text{fast}}$ (default: 12), emphasizing recent prices.
Slow EWMA uses $\alpha_{\text{slow}}$ (default: 26), emphasizing longer-term price trends.

**Step 2: Compute the EWMA crossover signal**

$$ \text{EMA Spread}_{i,t} = \text{EMA}_{i,t}^{(12)} - \text{EMA}_{i,t}^{(26)} $$

**Step 3: Normalize by volatility**

To make the signal comparable across different price ranges and volatility regimes, normalize by the rolling standard deviation:

$$ \text{EWMA Signal}_{i,t} = \frac{\text{EMA}_{i,t}^{(12)} - \text{EMA}_{i,t}^{(26)}}{\sigma_{i,t}^{(20)}} $$

Where:
- $\sigma_{i,t}^{(20)}$ = rolling standard deviation of closing prices over 20 days

The signal is **shifted forward by 1 bar** to eliminate look-ahead bias.

**Implementation** (Python):
```python
def ewma_crossover(close: pd.Series, fast_window: int = 12, 
                   slow_window: int = 26, std_window: int = 20) -> pd.Series:
    fast_ewma = close.ewm(span=fast_window).mean()
    slow_ewma = close.ewm(span=slow_window).mean()
    rolling_std = close.rolling(window=std_window).std().replace(0, np.nan)
    signal = (fast_ewma - slow_ewma) / rolling_std
    return signal.shift(1).fillna(0)
```

#### Signal Interpretation

- **Positive signal** ($\text{EWMA Signal} > 0$): Fast EMA above slow EMA → **uptrend** → rank as potential **long**
- **Negative signal** ($\text{EWMA Signal} < 0$): Fast EMA below slow EMA → **downtrend** → rank as potential **short**
- **Magnitude**: Larger values indicate stronger trend conviction (after accounting for volatility)

#### Portfolio Construction

Same as Momentum and Mean Reversion:
- Rank assets by EWMA crossover signal
- Long the top quantile, short the bottom quantile
- Equal-weight within baskets
- Rebalance periodically

#### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `ewma_fast_window` | 12 | 5-50 | Span for fast exponential moving average |
| `ewma_slow_window` | 26 | 20-200 | Span for slow exponential moving average |
| `ewma_std_window` | 20 | 10-100 | Window for volatility normalization |
| `ewma_rebalance` | 21 | 1-63 | Days between rebalancing |
| `ewma_top_q` | 0.20 | 0.10-0.50 | Fraction of assets in long basket |
| `ewma_bottom_q` | 0.20 | 0.10-0.50 | Fraction of assets in short basket |

---

## Mathematical Formulations

### General Framework

For a universe of $N$ assets observed over $T$ trading days:

**Price matrix**:

$$ P \in \mathbb{R}^{T \times N} $$

where $P_{t,i}$ is the closing price of asset $i$ on day $t$.

**Returns matrix**:

$$ r \in \mathbb{R}^{T \times N}, \quad r_{t,i} = \frac{P_{t,i} - P_{t-1,i}}{P_{t-1,i}} $$

**Signal matrix** (for each strategy $s \in \{\text{Mom}, \text{MR}, \text{EWMA}\}$):

$$ S_s \in \mathbb{R}^{T \times N} $$

where $S_s^{t,i}$ is the signal value at time $t$ for asset $i$.

### Cross-Sectional Ranking

On each rebalancing date $t_r$, convert signals to percentile ranks $\in [0, 1]$:

$$ \text{Rank}_{i,t_r} = \frac{\text{number of assets with signal} \leq S_s^{t_r,i}}{N} $$

This ranks assets from lowest (0) to highest (1).

### Portfolio Weight Construction

Given quantile thresholds $q_{\text{top}}$ and $q_{\text{bottom}}$:

Long basket $\mathcal{L}_{t_r}$: 

$$ \mathcal{L}_{t_r} = \{i : \text{Rank}_{i,t_r} \geq q_{\text{top}} \} $$

Short basket $\mathcal{S}_{t_r}$: 

$$ \mathcal{S}_{t_r} = \{i : \text{Rank}_{i,t_r} \leq q_{\text{bottom}} \} $$

**Weights on rebalance date**:

$$ w_{i,t_r}^{\text{long}} = \begin{cases}
\frac{1}{|\mathcal{L}_{t_r}|} & \text{if } i \in \mathcal{L}_{t_r} \\
0 & \text{otherwise}
\end{cases} $$

$$ w_{i,t_r}^{\text{short}} = \begin{cases}
-\frac{1}{|\mathcal{S}_{t_r}|} & \text{if } i \in \mathcal{S}_{t_r} \\
0 & \text{otherwise}
\end{cases} $$

$$ w_{i,t_r} = w_{i,t_r}^{\text{long}} + w_{i,t_r}^{\text{short}} $$

**Between rebalances**, weights are **held constant** (not rebalanced):

$$ w_{i,t} = w_{i,t_r} \quad \forall t \in [t_r, t_{r+1}) $$

---

## Portfolio Construction

### Multi-Strategy Combination

The platform supports combining all three strategies via **weighted average**:

$$ w_{i,t}^{\text{combined}} = \alpha_{\text{Mom}} \cdot w_{i,t}^{\text{Mom}} + \alpha_{\text{MR}} \cdot w_{i,t}^{\text{MR}} + \alpha_{\text{EWMA}} \cdot w_{i,t}^{\text{EWMA}} $$

Where:
- $\alpha_{\text{Mom}}, \alpha_{\text{MR}}, \alpha_{\text{EWMA}} \geq 0$
- $\alpha_{\text{Mom}} + \alpha_{\text{MR}} + \alpha_{\text{EWMA}} = 1$ (weights normalized)

**Default allocation**: Equal-weight ($\alpha = 1/3$ each)

The combined portfolio's daily returns are:

$$ r_{t}^{\text{combined}} = \alpha_{\text{Mom}} \cdot r_{t}^{\text{Mom}} + \alpha_{\text{MR}} \cdot r_{t}^{\text{MR}} + \alpha_{\text{EWMA}} \cdot r_{t}^{\text{EWMA}} $$

### Gross vs Net Exposure

At each time $t$, compute:

**Gross long exposure**:

$$ \text{Gross}_{\text{long},t} = \sum_{i: w_{i,t} > 0} w_{i,t} $$

**Gross short exposure** (absolute value):

$$ \text{Gross}_{\text{short},t} = \left| \sum_{i: w_{i,t} < 0} w_{i,t} \right| $$

**Net exposure**:

$$ \text{Net}_{t} = \text{Gross}_{\text{long},t} - \text{Gross}_{\text{short},t} $$

**Gross exposure**:

$$ \text{Gross}_{t} = \text{Gross}_{\text{long},t} + \text{Gross}_{\text{short},t} $$

For a standard long-short portfolio with equal-weight longs and shorts:

$$ \text{Gross}_{t} = 2.0 \quad \text{(fully invested)} $$

---

## Cost Calculations

### Transaction Costs

Transaction costs are modeled as a **linear function of portfolio turnover**:

$$ \text{Cost}_{t} = \text{Turnover}_{t} \times \text{TC}_\text{bps} $$

Where $\text{TC}_\text{bps}$ is the transaction cost rate expressed in **basis points** (bps), where 1 bps = 0.0001 = 0.01%.

### Turnover Calculation

Turnover measures the **magnitude of portfolio changes** from one day to the next:

$$ \text{Turnover}_{t} = \frac{1}{2} \sum_{i=1}^{N} |w_{i,t} - w_{i,t-1}| $$

The factor $\frac{1}{2}$ is applied because we count each trade twice (sell and buy):
- For each unit of net buying, there is an equivalent unit of net selling
- Therefore, total turnover is half the sum of absolute weight changes

**Example**:
- At $t-1$: long portfolio has weights $w = [0.5, 0.5, 0, ...]$
- At $t$: long portfolio has weights $w = [0.33, 0.33, 0.33, ...]$
- Weight changes: $\Delta w = [-0.17, -0.17, +0.33, ...]$
- Absolute changes: $|\Delta w| = [0.17, 0.17, 0.33, ...]$
- Sum: $0.17 + 0.17 + 0.33 = 0.67$
- Turnover: $0.67 / 2 = 0.335 \approx 33.5\%$ of AUM

### Cost Impact on Returns

Daily portfolio returns are reduced by transaction costs:

$$ r_{t}^{\text{net}} = r_{t}^{\text{gross}} - \text{Cost}_{t} = r_{t}^{\text{gross}} - (\text{Turnover}_{t} \times \text{TC}_\text{bps}) $$

Where:
- $r_{t}^{\text{gross}}$ = return before costs
- $\text{TC}_\text{bps}$ = transaction cost rate (e.g., 10 bps = 0.001)

**Example with 10 bps transaction costs**:
- If turnover is 33.5%, then costs = 0.335 × 0.001 = 0.0335 = **3.35 basis points** or **0.0335% of AUM**
- If gross daily return is 0.50%, net return = 0.50% - 0.0335% ≈ **0.4966%**

### Typical Cost Breakdown

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Maker fees (trading) | 1-2 bps | Paid to exchange for providing liquidity |
| Taker fees (trading) | 2-5 bps | Paid to exchange for taking liquidity |
| Borrowing costs (short) | 0.01-1% annually | Cost to borrow assets for shorting |
| Slippage | 1-10 bps | Impact of execution on price |
| **Total round-trip** | **10-20 bps** | Typical for this model |

In the platform, the user specifies **round-trip costs in basis points**, which is applied proportionally to turnover.

---

## Returns Calculations

### Gross Daily Returns

For a given day $t$, the portfolio's **gross return** (before transaction costs) is:

$$ r_{t}^{\text{gross}} = \sum_{i=1}^{N} w_{i,t-1}^{\text{previous}} \times r_{i,t} $$

Where:
- $w_{i,t-1}^{\text{previous}}$ = the **weight set on the previous day** $t-1$
- $r_{i,t}$ = the **daily return of asset $i$** on day $t$

This is a **lagged-weight** approach: we use yesterday's weights to compute today's return.

**Implementation detail**: The weight applied on day $t$ is the weight from **yesterday** ($t-1$):
```python
shifted_weights = weights.shift(1).fillna(0)  # Yesterday's weights
daily_returns = (shifted_weights * returns).sum(axis=1)  # Weighted sum of returns
```

### Cost-Adjusted Returns

$$ r_{t}^{\text{net}} = r_{t}^{\text{gross}} - \text{Turnover}_{t} \times \text{TC}_\text{bps} $$

### Cumulative Returns

The cumulative return series is computed via **chained compounding**:

$$ \text{Wealth}_{t} = \prod_{\tau=1}^{t} (1 + r_{\tau}^{\text{net}}) $$

Or equivalently:

$$ \text{Wealth}_{t} = (1 + r_1^{\text{net}}) \times (1 + r_2^{\text{net}}) \times \cdots \times (1 + r_t^{\text{net}}) $$

This represents the **total compound growth** of \$1 invested at the start.

**Example**:
- Day 1: $r_1^{\text{net}} = 0.1\%$ → Wealth: 1.001
- Day 2: $r_2^{\text{net}} = 0.05\%$ → Wealth: $1.001 \times 1.0005 = 1.001505$
- Day 3: $r_3^{\text{net}} = -0.2\%$ → Wealth: $1.001505 \times 0.998 = 0.999508$

### Period Returns

For any interval $[t_1, t_2]$:

$$ r_{t_1,t_2}^{\text{total}} = \frac{\text{Wealth}_{t_2}}{\text{Wealth}_{t_1}} - 1 $$

**Example**: If Wealth on day 1 is 1.001 and on day 3 is 0.999508:

$$ r_{1,3} = \frac{0.999508}{1.001} - 1 \approx -0.1491\% $$

---

## Backtesting Methodology

### Core Backtesting Algorithm

The backtesting engine follows this procedure:

```
1. INPUT:
   - Signal matrix S_s (computed for each strategy)
   - Price matrix P
   - Top/bottom quantile thresholds
   - Rebalancing frequency

2. FOR each rebalancing date t_r:
   a. Rank signals cross-sectionally to [0, 1]
   b. Identify long and short baskets based on quantiles
   c. Compute equal-weighted portfolio weights
   d. Set weights to 0 on non-rebalancing dates (hold previous weights)

3. FOR each trading day t:
   a. Apply previous day's weights to today's returns
   b. Compute gross portfolio return
   c. Calculate turnover (weight change from previous day)
   d. Apply transaction cost deduction
   e. Compute net return

4. OUTPUT:
   - Daily returns series
   - Cumulative returns series
   - Portfolio weights matrix
   - Performance metrics
```

### Implementation (Python)

```python
def backtest_signals(signal_df: pd.DataFrame, price_df: pd.DataFrame,
                     top_q: float = 0.9, bottom_q: float = 0.1,
                     rebalance_every: int = 21, transaction_cost_bps: float = 0.0,
                     long_short: bool = True) -> Dict:
    """High-level backtest routine."""

    # Step 1: Rank signals cross-sectionally
    ranks = signal_df.rank(axis=1, pct=True)

    # Step 2: Initialize weight matrix
    weights = pd.DataFrame(0.0, index=signal_df.index, columns=signal_df.columns)

    # Step 3: Compute weights on rebalance dates
    for i, date in enumerate(signal_df.index):
        if i % rebalance_every == 0:
            w_row = generate_weights_from_ranks(
                ranks.loc[[date]], top_q, bottom_q, long_short
            )
            weights.loc[date] = w_row.loc[date]

    # Step 4: Forward-fill weights (hold between rebalances)
    weights = weights.ffill().fillna(0)

    # Step 5: Compute portfolio returns
    daily_returns, metrics = compute_portfolio_returns(
        weights, price_df, transaction_cost_bps
    )

    # Step 6: Compute cumulative returns
    cumulative = (1 + daily_returns).cumprod()

    return {
        "daily_returns": daily_returns,
        "cumulative": cumulative,
        "metrics": metrics,
        "weights": weights
    }
```

### Weight Generation

```python
def generate_weights_from_ranks(ranks: pd.DataFrame, top_q: float = 0.9,
                                bottom_q: float = 0.1, long_short: bool = True) -> pd.DataFrame:
    """Convert ranked signals to portfolio weights."""
    weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)

    for date in ranks.index:
        row = ranks.loc[date]

        # Long basket: top quantile
        longs = row[row >= row.quantile(top_q)].index.tolist()
        # Short basket: bottom quantile
        shorts = row[row <= row.quantile(bottom_q)].index.tolist() if long_short else []

        # Equal weight within baskets
        if longs:
            weights.loc[date, longs] = 1.0 / len(longs)
        if shorts:
            weights.loc[date, shorts] = -1.0 / len(shorts)

    return weights
```

### Return Computation with Costs

```python
def compute_portfolio_returns(weights: pd.DataFrame, price_df: pd.DataFrame,
                              transaction_cost_bps: float = 0.0) -> Tuple[pd.Series, Dict]:
    """Calculate portfolio returns with transaction costs."""

    # Align prices and compute returns
    prices = price_df.reindex(index=weights.index).ffill()
    returns = prices.pct_change().fillna(0)

    # Lagged weights (apply yesterday's weights to today's returns)
    shifted_weights = weights.shift(1).fillna(0)

    # Compute turnover (weight changes)
    w_prev = shifted_weights.shift(1).fillna(0)
    turnover = (shifted_weights - w_prev).abs().sum(axis=1) / 2.0

    # Transaction cost impact
    cost_impact = turnover * transaction_cost_bps

    # Compute daily returns
    daily_returns = (shifted_weights * returns).sum(axis=1) - cost_impact

    return daily_returns, {"turnover_mean": turnover.mean()}
```

### Rebalancing Schedule

Rebalancing dates are determined by a **modulo operation**:

$$ \text{Rebalance}_t = \begin{cases}
\text{True} & \text{if } t \bmod \text{rebalance\\_every} = 0 \\
\text{False} & \text{otherwise}
\end{cases} $$

**Example** with `rebalance_every = 21`:
- Day 0: Rebalance (0 mod 21 = 0)
- Day 21: Rebalance (21 mod 21 = 0)
- Day 42: Rebalance (42 mod 21 = 0)
- Days 1-20, 22-41, etc.: Hold weights from previous rebalance date

---

## Performance Metrics

All metrics are computed on the **net daily returns** (after transaction costs).

### Annualized Return

$$ r_{\text{annual}} = (1 + \bar{r}_{\text{daily}})^{252} - 1 $$

Where:
- $\bar{r}_{\text{daily}} = \frac{1}{T} \sum_{t=1}^{T} {r_t}^{\text{net}}$ = mean daily return
- **252** = number of trading days per year (standard for financial markets)

**Example**: If average daily return is 0.05% (0.0005):

$$ r_{\text{annual}} = (1.0005)^{252} - 1 \approx 12.7\% $$

### Annualized Volatility

$$ \sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252} $$

Where:
- $\sigma_{\text{daily}} = \sqrt{\frac{1}{T-1} \sum_{t=1}^{T} (r_t - \bar{r})^2}$ = sample standard deviation of daily returns

**Example**: If daily volatility is 1.5%:

$$ \sigma_{\text{annual}} = 0.015 \times \sqrt{252} \approx 0.015 \times 15.87 \approx 23.8\% $$

### Sharpe Ratio

The Sharpe ratio measures **risk-adjusted return**:

$$ \text{Sharpe} = \frac{r_{\text{annual}} - r_f}{\sigma_{\text{annual}}} $$

Where:
- $r_f$ = risk-free rate (typically 0% in this platform; can be adjusted to 2-5% annually)
- $\text{Sharpe}$ = excess return per unit of risk

**Interpretation**:
- $\text{Sharpe} < 0$: Strategy underperforms risk-free rate
- $0 < \text{Sharpe} < 1$: Below-average risk-adjusted returns
- $1 < \text{Sharpe} < 2$: Good risk-adjusted returns
- $\text{Sharpe} > 2$: Excellent risk-adjusted returns

**Example**: If annual return is 20%, volatility is 15%, and risk-free rate is 0%:

$$ \text{Sharpe} = \frac{0.20 - 0}{0.15} \approx 1.33 $$

### Maximum Drawdown

The maximum drawdown is the **largest peak-to-trough decline** in cumulative wealth:

$$ \text{MDD} = \min_t \left( \frac{\text{Wealth}_t - \text{Peak}_{\text{up to } t}}{\text{Peak}_{\text{up to } t}} \right) $$

Where:
- $\text{Peak}_{\text{up to } t} = \max_{\tau \leq t} \text{Wealth}_{\tau}$ = running maximum wealth

This measures the **worst-case loss** from a previous high point.

**Algorithm**:
```python
def max_drawdown(cum_returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    wealth = cum_returns if cum_returns.iloc[0] == 1.0 else (1 + cum_returns).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    return drawdown.min()
```

**Example**:
- Starting wealth: 1.0
- Peak wealth: 1.2 (on day 30)
- Trough wealth: 0.95 (on day 50)
- Drawdown from peak: (0.95 - 1.2) / 1.2 = -0.2083 = **-20.83%**

### Information Ratio (Optional)

For comparing to a benchmark, the Information Ratio is:

$$ \text{IR} = \frac{r_{\text{annual}} - r_{\text{benchmark}}}{\sigma_{\text{tracking}}} $$

Where:
- $r_{\text{benchmark}}$ = annual return of benchmark (e.g., buy-and-hold Bitcoin)
- $\sigma_{\text{tracking}}$ = standard deviation of excess returns

---

## Implementation Details

### Data Fetching & Caching

The platform uses **CCXT** (CryptoCurrency eXchange Trading) to fetch OHLCV data from Binance:

```python
def fetch_ohlcv_ccxt(symbol: str, timeframe: str = "1d",
                     since: Optional[int] = None, limit: int = 365*4) -> pd.DataFrame:
    """Fetch OHLCV from Binance using ccxt."""
    exchange = ccxt.binance({"enableRateLimit": True})
    all_rows = []
    since_param = since

    while True:
        try:
            chunk = exchange.fetch_ohlcv(symbol, timeframe=timeframe,
                                        since=since_param, limit=limit)
        except ccxt.BaseError as e:
            print(f"Error: {e}, retrying...")
            time.sleep(2)
            continue

        if not chunk or len(chunk) < limit:
            all_rows.extend(chunk if chunk else [])
            break

        all_rows.extend(chunk)
        since_param = chunk[-1][0] + 1

    # Convert to DataFrame
    df = pd.DataFrame(all_rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('datetime').drop(columns=['ts'])
    return df.sort_index()
```

**Features**:
- **Pagination**: Automatically handles API pagination for large date ranges
- **Error handling**: Retries on rate limit errors
- **Caching**: Saves data to Parquet format for faster subsequent access

### Data Alignment

The `align_universe()` function ensures all cryptocurrencies are aligned to the same timestamps:

```python
def align_universe(dfs: Dict[str, pd.DataFrame], freq: str = "1D") -> pd.DataFrame:
    """Align timestamps across symbols and build close prices matrix."""
    closes = {}

    for symbol, df in dfs.items():
        dfc = clean_price_df(df)
        if dfc.empty or "close" not in dfc.columns:
            continue

        # Resample to frequency and forward-fill
        dfc = dfc.resample(freq).last().ffill()
        closes[symbol] = dfc["close"]

    # Concatenate into matrix
    closes_df = pd.concat(closes.values(), axis=1, keys=closes.keys())
    return closes_df.dropna(axis=1, how="all")
```

**Process**:
1. Clean each symbol's data (remove duplicates, convert types)
2. Resample to trading frequency (daily)
3. Forward-fill missing values (e.g., days with no trading)
4. Concatenate all symbols into a single DataFrame
5. Drop symbols with completely missing data

### Signal Computation

Signals are computed **independently for each asset**, then stacked into a matrix:

```python
def compute_momentum_signals_df(closes: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Compute momentum signals for all symbols."""
    signals = pd.DataFrame(index=closes.index, columns=closes.columns)

    for sym in closes.columns:
        signals[sym] = momentum(closes[sym], lookback=lookback)

    return signals.fillna(0)
```

### Universe Selection

The platform selects the top-N cryptocurrencies by **24-hour trading volume**:

```python
def get_top_symbols(n: int = 10, quote: str = "USDT") -> List[str]:
    """Get top-n trading symbols by volume."""
    exchange = ccxt.binance()
    markets = exchange.fetch_markets()
    filtered = [m for m in markets
                if m.get("quote") == quote and m.get("active")]
    filtered.sort(
        key=lambda m: float(m.get("info", {}).get("quoteVolume", 0) or 0),
        reverse=True
    )
    return [m["symbol"] for m in filtered][:n]
```

This ensures:
- **Sufficient liquidity**: High-volume assets have tighter bid-ask spreads
- **Diversification**: Captures multiple segments (large-cap, mid-cap, alt-coins)
- **Tradability**: Avoids illiquid tokens where transaction costs are prohibitive

### Handling Edge Cases

1. **NaN values in signals**: Filled with 0 (neutral signal)
2. **Division by zero** (e.g., zero volatility): Handled by `.replace(0, np.nan)` to avoid misleading signals
3. **Missing price data**: Forward-filled with last observed price (realistic for daily data)
4. **Zero turnover**: On rebalancing dates when weights don't change, turnover is 0

---

## Configuration & Parameters

### Interactive Controls (Streamlit UI)

The Streamlit interface provides real-time parameter adjustment:

**Global controls**:
- Universe size (10, 20, 30 cryptocurrencies)
- Date range (start & end dates)
- Transaction cost (in basis points)

**Strategy-specific controls**:

| Strategy | Parameters |
|----------|-----------|
| Momentum | Lookback, rebalance frequency, top/bottom quantiles |
| Mean Reversion | Lookback, rebalance frequency, top/bottom quantiles |
| EWMA | Fast/slow windows, std window, rebalance frequency, top/bottom quantiles |

**Portfolio controls**:
- Strategy weights (Momentum, Mean Reversion, EWMA)
- Automatic normalization to sum to 1.0

### Default Parameter Recommendations

| Strategy | Parameter | Recommended | Rationale |
|----------|-----------|-------------|-----------|
| Momentum | Lookback | 21 days | 1 trading month |
| Momentum | Rebalance | 21 days | Aligns with lookback |
| Momentum | Quantiles | 20% (top & bottom) | Balanced long-short |
| Mean Reversion | Lookback | 14 days | 2 weeks of history |
| Mean Reversion | Rebalance | 7 days | Shorter for faster mean reversion |
| Mean Reversion | Quantiles | 30% (top & bottom) | Broader extremes |
| EWMA | Fast window | 12 | MACD-inspired |
| EWMA | Slow window | 26 | MACD-inspired |
| EWMA | Std window | 20 | ~1 month |
| EWMA | Rebalance | 21 days | ~1 trading month |
| EWMA | Quantiles | 20% (top & bottom) | Balanced |

---

## Advanced Topics

### Correlation Analysis

The platform computes **Pearson correlation** of daily returns between strategies:

$$ \rho_{s_1, s_2} = \frac{\text{Cov}(r_1, r_2)}{\sigma_1 \times \sigma_2} $$

Where:
- $r_1, r_2$ = daily return series of two strategies
- $\text{Cov}(r_1, r_2)$ = covariance
- $\sigma_1, \sigma_2$ = standard deviations

**Interpretation**:
- $\rho \approx 1$: Strategies move together (low diversification benefit)
- $\rho \approx 0$: Strategies independent (good diversification)
- $\rho \approx -1$: Strategies move opposite (hedge each other)

Low correlation is desirable in a multi-strategy portfolio.

### Portfolio Exposure Metrics

**Gross Long + Short Exposure**:

$$ \text{Exposure}_{t} = \sum_{i: w_{i,t} > 0} w_{i,t} + \left| \sum_{i: w_{i,t} < 0} w_{i,t} \right| $$

For equal-weight long-short: $\text{Exposure} = 2.0$ (100% long, 100% short)

**Net Exposure**:

$$ \text{Net}_{t} = \sum_{i: w_{i,t} > 0} w_{i,t} + \sum_{i: w_{i,t} < 0} w_{i,t} $$

For equal-weight long-short: $\text{Net} = 0.0$ (market neutral)

### Turnover Analysis

Track **turnover over time** to understand:
- Strategy churn rate
- Impact of rebalancing frequency
- Implicit costs due to trading

Average turnover (from backtest):

$$ \overline{\text{TO}} = \frac{1}{T} \sum_{t=1}^{T} \text{Turnover}_t $$

---

## Conclusion

This platform provides a comprehensive framework for:

1. **Signal generation**: Three distinct alpha factors (momentum, mean reversion, EWMA)
2. **Portfolio construction**: Cross-sectional ranking and equal-weighted baskets
3. **Risk-aware backtesting**: Transaction costs, turnover, realistic P&L
4. **Multi-strategy combination**: Diversification across approaches
5. **Performance analysis**: Sharpe, max drawdown, correlation, exposure metrics

The mathematical foundations ensure **reproducibility**, **no look-ahead bias**, and **realistic transaction cost modeling** for a fair assessment of strategy performance on cryptocurrency markets.

