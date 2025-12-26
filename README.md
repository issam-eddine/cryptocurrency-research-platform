# Cryptocurrency Research Platform

A sophisticated multi-strategy backtesting framework for cryptocurrency trading. This platform combines three factor-based quantitative strategies—**Momentum**, **Mean Reversion**, and **EWMA Crossover**—into a diversified portfolio with live data fetching from Binance.

---

## Table of Contents

- [Cryptocurrency Research Platform](#cryptocurrency-research-platform)
  - [Table of Contents](#table-of-contents)
  - [Live Demo](#live-demo)
  - [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Running the Application](#running-the-application)
  - [Project Structure](#project-structure)
  - [Architecture Overview](#architecture-overview)
  - [Pipeline Components](#pipeline-components)
    - [1. DataPipeline](#1-datapipeline)
    - [2. SignalStrategy](#2-signalstrategy)
    - [3. Predictor](#3-predictor)
    - [4. Portfolio](#4-portfolio)
    - [5. TargetEngineer](#5-targetengineer)
    - [6. Backtester](#6-backtester)
    - [7. MetricsCalculator](#7-metricscalculator)
  - [Signal Processing Pipeline](#signal-processing-pipeline)
    - [Step 1: Raw Signal Computation](#step-1-raw-signal-computation)
    - [Step 2: Cross-Sectional Z-Scoring](#step-2-cross-sectional-z-scoring)
    - [Step 3: Quantile Filtering](#step-3-quantile-filtering)
    - [Step 4: Re-Z-Scoring Active Positions](#step-4-re-z-scoring-active-positions)
  - [Timing Convention](#timing-convention)
  - [Trading Strategies](#trading-strategies)
    - [Momentum Strategy](#momentum-strategy)
    - [Mean Reversion Strategy](#mean-reversion-strategy)
    - [EWMA Crossover Strategy](#ewma-crossover-strategy)
  - [Streamlit Web Interface](#streamlit-web-interface)
    - [Configuration Sidebar](#configuration-sidebar)
    - [Visualizations](#visualizations)
  - [Configuration \& Parameters](#configuration--parameters)
    - [Default Parameters (main.py)](#default-parameters-mainpy)
  - [Dependencies](#dependencies)
    - [Core Libraries](#core-libraries)
    - [Installation](#installation-1)
  - [License](#license)

---

## Live Demo

Access the live application directly without any installation:

🔗 **[https://cryptocurrency-research-platform.streamlit.app/](https://cryptocurrency-research-platform.streamlit.app/)**

---

## Quick Start

### Prerequisites

- Python 3.11 or higher
- [`uv`](https://github.com/astral-sh/uv) package manager (recommended) or `pip`

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/issam-eddine/cryptocurrency-research-platform.git
   cd cryptocurrency-research-platform
   ```

2. **Create and activate virtual environment**

   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   uv sync
   ```

### Running the Application

**Option 1: Streamlit Web Interface**

```bash
streamlit run streamlit/app.py
```

The application will open in your browser at `http://localhost:8501`.

**Option 2: Command Line Pipeline**

```bash
python main.py
```

This runs the full backtesting pipeline with default parameters and prints performance metrics to the console.

---

## Project Structure

```
cryptocurrency-research-platform/
├── main.py                     # CLI entry point - full pipeline example
├── pyproject.toml              # Project configuration & dependencies
├── requirements.txt            # Alternative pip dependencies
├── README.md
│
├── src/
│   ├── __init__.py
│   └── pipeline/               # Core backtesting framework
│       ├── __init__.py         # Public API exports
│       ├── data_pipeline.py    # Data fetching & preprocessing
│       ├── signal_strategy.py  # Trading strategy implementations
│       ├── predictor.py        # Signal processing & filtering
│       ├── portfolio.py        # Multi-strategy combination
│       ├── target_engineer.py  # Return computation
│       ├── backtester.py       # Backtest execution engine
│       ├── metrics.py          # Performance metrics calculator
│       ├── signal_utils.py     # Z-scoring & quantile utilities
│       └── feature_engineer.py # Feature computation (extensible)
│
└── streamlit/
    └── app.py                  # Interactive web interface
```

---

## Architecture Overview

The platform follows a modular pipeline architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
│  ┌─────────────┐                                                            │
│  │ DataPipeline │ ──► Fetch OHLCV from Binance ──► Cache to Parquet         │
│  │              │ ──► Preprocess & Align ──► Build Price Matrix             │
│  └─────────────┘                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SIGNAL LAYER                                      │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐        │
│  │ MomentumStrategy │   │MeanReversionStrat│   │ EWMACrossoverStr │        │
│  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘        │
│           │                      │                      │                   │
│           ▼                      ▼                      ▼                   │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐        │
│  │    Predictor     │   │    Predictor     │   │    Predictor     │        │
│  │ (z-score+filter) │   │ (z-score+filter) │   │ (z-score+filter) │        │
│  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘        │
│           │                      │                      │                   │
│           └──────────────────────┼──────────────────────┘                   │
│                                  ▼                                          │
│                         ┌──────────────────┐                                │
│                         │    Portfolio     │                                │
│                         │ (weighted combo) │                                │
│                         └────────┬─────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXECUTION LAYER                                     │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐        │
│  │  TargetEngineer  │   │    Backtester    │   │ MetricsCalculator│        │
│  │ (compute returns)│──►│ (simulate trades)│──►│ (performance KPIs)│        │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Components

### 1. DataPipeline

**Location:** `src/pipeline/data_pipeline.py`

Handles all data acquisition and preprocessing operations.

**Key Features:**

| Feature | Description |
|---------|-------------|
| **Data Source** | Binance US exchange via CCXT library |
| **Timeframe** | Configurable (default: 1-hour candles) |
| **Caching** | Automatic Parquet file caching to avoid redundant API calls |
| **Retry Logic** | Exponential backoff (2s → 4s → 8s → 16s → 32s) on failures |
| **Historical Depth** | Fetches up to 10 years of historical data |

**Data Flow:**

```
Binance API ──► Raw OHLCV DataFrames ──► Parquet Cache
                                              │
                                              ▼
                            Clean & Normalize (remove duplicates, sort)
                                              │
                                              ▼
                            Resample to common frequency (1h)
                                              │
                                              ▼
                            Forward-fill missing values
                                              │
                                              ▼
                            Build unified Price Matrix (dates × symbols)
```

**Usage:**

```python
from src.pipeline import DataPipeline

pipeline = DataPipeline(cache_dir="data/raw", exchange_id="binanceus")

# Fetch data
raw_data = pipeline.fetch(
    symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)

# Preprocess
cleaned = pipeline.preprocess()

# Get aligned price matrix
prices = pipeline.get_price_matrix(freq="1h")
```

---

### 2. SignalStrategy

**Location:** `src/pipeline/signal_strategy.py`

Abstract base class for implementing trading signals. Each concrete strategy computes a signal from price data where **higher values indicate more bullish outlook**.

**Class Hierarchy:**

```
SignalStrategy (ABC)
    ├── MomentumStrategy
    ├── MeanReversionStrategy
    └── EWMACrossoverStrategy
```

**Interface:**

```python
class SignalStrategy(ABC):
    def compute(self, close: pd.Series) -> pd.Series:
        """Compute signal for a single asset."""
        pass
    
    def compute_universe(self, price_matrix: pd.DataFrame) -> pd.DataFrame:
        """Compute signals for all assets."""
        pass
```

---

### 3. Predictor

**Location:** `src/pipeline/predictor.py`

Wraps a `SignalStrategy` and applies a three-stage signal processing pipeline.

**Processing Pipeline:**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Signal    │───►│   Z-Score       │───►│ Quantile Filter │
│  (from strategy)│    │ (cross-section) │    │  (top/bottom %) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  Re-Z-Score     │
                                              │ (active only)   │
                                              └─────────────────┘
```

**Key Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | SignalStrategy | required | The signal strategy to use |
| `top_q` | float | 0.8 | Quantile threshold for longs (0.8 = top 20%) |
| `bottom_q` | float | 0.2 | Quantile threshold for shorts (0.2 = bottom 20%) |
| `long_short` | bool | True | Enable short positions |
| `discrete` | bool | False | Use +1/0/-1 instead of continuous values |

**Usage:**

```python
from src.pipeline import Predictor, MomentumStrategy

predictor = Predictor(
    strategy=MomentumStrategy(lookback=168),  # 1 week
    top_q=0.8,
    bottom_q=0.2,
    long_short=True
)

# Get unfiltered signal (for portfolio combination)
unfiltered = predictor.compute_unfiltered_signal(price_matrix)

# Or get fully processed signal (for single-strategy backtest)
processed = predictor.predict(price_matrix)
```

---

### 4. Portfolio

**Location:** `src/pipeline/portfolio.py`

Combines multiple predictors using weighted average at the **unfiltered signal level**.

**Why Combine at Unfiltered Level?**

Combining z-scored (but not yet filtered) signals ensures:
- Different strategies are on comparable scales before merging
- The final filtering is applied to the combined signal, not individual strategies
- Proper diversification benefits are achieved

**Combination Workflow:**

```
Predictor 1 ──► Unfiltered Signal (z-scored) ──┐
                                               │
Predictor 2 ──► Unfiltered Signal (z-scored) ──┼──► Weighted Sum ──► Filter ──► Re-Z-Score ──► Weights
                                               │
Predictor 3 ──► Unfiltered Signal (z-scored) ──┘
```

**Key Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictor_weights` | Dict[str, float] | None | Weight allocation per predictor |
| `top_q` | float | 0.8 | Quantile threshold for final long positions |
| `bottom_q` | float | 0.2 | Quantile threshold for final short positions |
| `enable_vol_target` | bool | False | Enable volatility targeting |
| `vol_target` | float | 0.20 | Target annualized volatility (20%) |

**Usage:**

```python
from src.pipeline import Portfolio

portfolio = Portfolio(
    predictor_weights={
        "momentum": 0.4,
        "mean_reversion": 0.3,
        "ewma_crossover": 0.3
    },
    top_q=0.8,
    bottom_q=0.2
)

# Add unfiltered signals from each predictor
portfolio.add_predictor_unfiltered_signal("momentum", momentum_unfiltered)
portfolio.add_predictor_unfiltered_signal("mean_reversion", mr_unfiltered)
portfolio.add_predictor_unfiltered_signal("ewma_crossover", ewma_unfiltered)

# Process: combine → filter → re-zscore → weights
weights = portfolio.combine_and_process()
```

---

### 5. TargetEngineer

**Location:** `src/pipeline/target_engineer.py`

Computes return series for backtesting and evaluation.

**Timing Convention:**

```
returns[t] = (price[t] - price[t-1]) / price[t-1]

This represents the return from close of period t-1 to close of period t.
```

**Methods:**

| Method | Description |
|--------|-------------|
| `compute_returns()` | Standard percentage returns |
| `compute_targets()` | Forward returns (for prediction tasks) |
| `compute_binary_targets()` | Binary classification (up/down) |
| `compute_tercile_targets()` | Tercile classification (bottom/middle/top) |

---

### 6. Backtester

**Location:** `src/pipeline/backtester.py`

Orchestrates the full backtest simulation with realistic trading mechanics.

**Simulation Features:**

| Feature | Description |
|---------|-------------|
| **Rebalancing** | Configurable frequency (default: 24 periods = daily for hourly data) |
| **Transaction Costs** | Proportional to turnover (default: 10 bps) |
| **Slippage** | Estimated execution slippage (default: 5 bps) |
| **Turnover Calculation** | Half the sum of absolute weight changes |

**Backtest Flow:**

```
1. Align weights and prices to common dates/symbols
           │
           ▼
2. Apply rebalancing schedule (forward-fill between rebalance periods)
           │
           ▼
3. Compute asset returns using TargetEngineer
           │
           ▼
4. Shift weights by 1 period (trade at close, realize returns next period)
           │
           ▼
5. Calculate turnover and transaction costs
           │
           ▼
6. Compute portfolio returns: shifted_weights × asset_returns
           │
           ▼
7. Subtract transaction costs → net returns
           │
           ▼
8. Compound net returns → cumulative return series
```

**Output: BacktestResult**

```python
@dataclass
class BacktestResult:
    daily_returns: pd.Series       # Period-by-period net returns
    cumulative_returns: pd.Series  # Compounded wealth curve
    weights: pd.DataFrame          # Applied portfolio weights
    metrics: PerformanceMetrics    # Performance statistics
    turnover: pd.Series            # Portfolio turnover per period
```

---

### 7. MetricsCalculator

**Location:** `src/pipeline/metrics.py`

Computes comprehensive performance statistics.

**Available Metrics:**

| Metric | Formula | Description |
|--------|---------|-------------|
| **Annual Return** | Geometric mean annualized | Expected yearly return |
| **Annual Volatility** | σ × √(periods_per_year) | Annualized standard deviation |
| **Sharpe Ratio** | (Annual Return - Rf) / Annual Vol | Risk-adjusted return |
| **Max Drawdown** | Largest peak-to-trough decline | Worst loss from peak |
| **Calmar Ratio** | Annual Return / \|Max Drawdown\| | Return per unit of drawdown |
| **Win Rate** | % of positive returns | Consistency measure |
| **Profit Factor** | Gross Profit / Gross Loss | Gain-to-loss ratio |
| **Average Win** | Mean of positive returns | Typical winning period |
| **Average Loss** | Mean of negative returns | Typical losing period |

**Note:** For hourly cryptocurrency data, `periods_per_year = 8760` (365 × 24).

---

## Signal Processing Pipeline

The platform uses a standardized signal processing flow to ensure comparability across strategies:

### Step 1: Raw Signal Computation

Each strategy produces a raw signal based on its logic (momentum, mean reversion, etc.).

### Step 2: Cross-Sectional Z-Scoring

```python
z_score[t, asset] = (signal[t, asset] - mean_across_assets[t]) / std_across_assets[t]
```

This normalizes signals across all assets at each timestamp, producing zero mean and unit standard deviation.

### Step 3: Quantile Filtering

```python
if z_score >= quantile(top_q):     # e.g., top 20%
    position = +z_score            # Long
elif z_score <= quantile(bottom_q): # e.g., bottom 20%
    position = -z_score            # Short (if long_short=True)
else:
    position = 0                   # No position
```

### Step 4: Re-Z-Scoring Active Positions

Only the non-zero (active) positions are re-z-scored to maintain proper scaling:

```python
active_z_score = (position - mean_of_active) / std_of_active
```

---

## Timing Convention

Understanding the timing convention is critical for correct backtesting:

```
Time:     t-1          t           t+1
           │           │            │
           ▼           ▼            ▼
Price:   P[t-1]      P[t]        P[t+1]
           │           │
           │◄─────────►│
           │  Return[t] = (P[t] - P[t-1]) / P[t-1]
           │
           │
      Decide weights
       weights[t-1]
           │
           └────────────► Apply during period t
                          Realize Return[t]

Portfolio Return[t] = weights[t-1] × returns[t]
```

**Key Points:**

1. Weights are determined at the **close of period t-1**
2. These weights are **held during period t**
3. Returns from **close t-1 to close t** are realized
4. No lookahead bias: decisions are made before returns are known

---

## Trading Strategies

### Momentum Strategy

**Concept:** Assets that have performed well recently will continue to perform well.

**Signal Computation:**

```python
signal = close.pct_change(periods=lookback).shift(1)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 168 hours (1 week) | Return calculation period |

**Interpretation:**
- Positive signal → asset has appreciated → go long
- Negative signal → asset has depreciated → go short

---

### Mean Reversion Strategy

**Concept:** Assets that have deviated from their mean will revert back.

**Signal Computation:**

```python
returns = close.pct_change()
z_score = (returns - rolling_mean) / rolling_std
signal = -z_score.shift(1)  # Negative for mean-reversion
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 72 hours (3 days) | Rolling window for mean/std |

**Interpretation:**
- Negative z-score (oversold) → positive signal → go long
- Positive z-score (overbought) → negative signal → go short

---

### EWMA Crossover Strategy

**Concept:** Trend-following using fast vs slow exponential moving average crossover.

**Signal Computation:**

```python
fast_ewma = close.ewm(span=fast_window).mean()
slow_ewma = close.ewm(span=slow_window).mean()
signal = ((fast_ewma - slow_ewma) / rolling_std).shift(1)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fast_window` | 24 hours | Fast EMA span |
| `slow_window` | 168 hours (1 week) | Slow EMA span |
| `std_window` | 72 hours | Volatility normalization window |

**Interpretation:**
- Fast EWMA > Slow EWMA → uptrend → positive signal → go long
- Fast EWMA < Slow EWMA → downtrend → negative signal → go short

---

## Streamlit Web Interface

The interactive web interface (`streamlit/app.py`) provides:

### Configuration Sidebar

| Section | Controls |
|---------|----------|
| **Global** | Universe size (10/20 assets), date range |
| **Momentum** | Lookback, rebalance frequency, quantile thresholds |
| **Mean Reversion** | Lookback, rebalance frequency, quantile thresholds |
| **EWMA Crossover** | Fast/slow/std windows, rebalance frequency, quantiles |
| **Portfolio** | Strategy weights, combined filtering thresholds |
| **Costs** | Transaction cost in basis points |

### Visualizations

| Chart | Description |
|-------|-------------|
| **Asset Cumulative Returns** | Individual cryptocurrency performance |
| **Strategy Cumulative Returns** | Comparison of all strategies + combined |
| **Performance Metrics Cards** | Key KPIs for each strategy |
| **Detailed Metrics Table** | Full metrics comparison |
| **Correlation Matrix** | Strategy return correlations |
| **Signal Heatmaps** | Z-scored signals by asset and time |
| **Returns Distribution** | Histogram of period returns |
| **Drawdown Chart** | Underwater equity curves |

---

## Configuration & Parameters

### Default Parameters (main.py)

| Category | Parameter | Value | Description |
|----------|-----------|-------|-------------|
| **Universe** | Symbols | 7 major coins | BTC, ETH, BNB, XRP, ADA, SOL, DOGE |
| **Date Range** | Period | 2 months | Nov 1 - Dec 31, 2024 |
| **Momentum** | Lookback | 168 hours | 1 week |
| **Mean Reversion** | Lookback | 72 hours | 3 days |
| **EWMA** | Fast/Slow | 24/168 hours | 1 day / 1 week |
| **Filtering** | Top/Bottom | 80% / 20% | Top and bottom quintile |
| **Portfolio** | Weights | 40/30/30 | Momentum / MR / EWMA |
| **Execution** | Rebalance | 24 hours | Daily |
| **Costs** | Transaction | 10 bps | 0.10% round-trip |
| **Costs** | Slippage | 5 bps | 0.05% execution |

---

## Dependencies

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥2.0.0 | Data manipulation & analysis |
| `numpy` | ≥1.24.0 | Numerical computing |
| `ccxt` | ≥4.0.0 | Cryptocurrency exchange connectivity |
| `streamlit` | ≥1.28.0 | Interactive web interface |
| `plotly` | ≥5.15.0 | Interactive visualizations |
| `scipy` | ≥1.11.0 | Scientific computing |
| `scikit-learn` | ≥1.3.0 | Machine learning utilities |
| `matplotlib` | ≥3.7.0 | Static plotting |
| `pyarrow` | ≥14.0.0 | Parquet file support |

### Installation

**Using uv (recommended):**

```bash
uv sync
```

**Using pip:**

```bash
pip install -r requirements.txt
```

---

## License

This project is provided as-is for educational and research purposes.

---

<p align="center">
  <strong>Built for quantitative cryptocurrency research</strong>
</p>
