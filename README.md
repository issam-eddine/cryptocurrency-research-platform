# Cryptocurrency Research Platform

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

## Overview

This is a **Cryptocurrency Research Platform** that implements a multi-strategy backtesting framework for cryptocurrency trading. The project combines three factor-based strategies (Momentum, Mean Reversion, and EWMA Crossover) into a diversified portfolio with live data fetching from Binance.

## Project Architecture

The platform follows a modular pipeline architecture with eight core components organized in the `src/pipeline` directory. Each module handles a specific stage of the backtesting workflow, allowing strategies to be tested individually or combined into multi-strategy portfolios.

### Data Pipeline Construction

The **DataPipeline** class manages data acquisition and preprocessing. It fetches OHLCV (Open, High, Low, Close, Volume) data from Binance using the CCXT library, implementing automatic caching to parquet files to avoid redundant API calls. The pipeline fetches data in chunks of 1000 candles, starting from 10 years ago, and continues until all available historical data is retrieved. After fetching, it aligns all symbols to a common time grid through resampling, forward-filling missing values, and building a unified price matrix where rows represent timestamps and columns represent different cryptocurrency symbols.

### Signal Strategy Sequential Construction

Each strategy inherits from the **SignalStrategy** base class and implements distinct signal computation logic. The **MomentumStrategy** calculates past returns over a lookback period, shifted by one period to prevent lookahead bias, producing positive signals for assets showing upward trends. The **MeanReversionStrategy** computes rolling z-scores of returns and negates them, generating positive signals when prices are oversold relative to their recent behavior. The **EWMACrossoverStrategy** calculates the difference between fast and slow exponential weighted moving averages, normalized by rolling standard deviation, producing positive signals when the fast EWMA crosses above the slow EWMA.

### Predictor Signal Processing Pipeline

The **Predictor** class wraps each strategy and applies a three-stage processing pipeline. First, it computes raw signals using the strategy's transformation logic. Second, it applies cross-sectional z-scoring across all assets at each timestamp, standardizing signals to have zero mean and unit standard deviation across the universe. Third, it filters signals by quantile thresholds—only assets in the top quantile (e.g., top 20%) receive long signals, while those in the bottom quantile (e.g., bottom 20%) receive short signals. Finally, it re-z-scores only the active (non-zero) positions to ensure the filtered signals maintain proper scaling.

### Portfolio Combination Workflow

The **Portfolio** class combines multiple predictors using a weighted average approach at the unfiltered signal level. This is crucial because combining z-scored but unfiltered signals ensures different strategies are on comparable scales before merging. The workflow collects unfiltered signals (z-scored but not filtered) from each predictor, combines them using configurable allocation weights, applies quantile filtering to the combined signal, and re-z-scores the active positions. Optional volatility targeting can scale weights to achieve a target portfolio volatility, though this feature is disabled by default.

### Target Engineering and Timing Convention

The **TargetEngineer** computes forward returns for training or evaluation. The timing convention is critical: returns[t] represents the return from close of day t-1 to close of day t, calculated as `(prices[t] - prices[t-1]) / prices[t-1]`. For backtesting, `portfolio_returns[t] = weights[t-1] * returns[t]`, meaning weights determined at close of day t-1 are held during day t, and returns realized from t-1 close to t close are applied. This prevents lookahead bias by ensuring position decisions are made before returns are realized.

### Backtesting Execution Flow

The **Backtester** orchestrates the full simulation with rebalancing schedules, transaction costs, and performance tracking. It aligns weights and price matrices to common dates and symbols, applies the rebalancing schedule by forward-filling weights between rebalance periods, computes asset returns using the TargetEngineer, shifts weights by one period to match the timing convention, calculates turnover as half the sum of absolute weight changes, applies transaction costs and slippage proportional to turnover, computes portfolio returns by multiplying shifted weights with asset returns, and subtracts transaction costs to get net returns. The cumulative return series is built by compounding net returns: `(1 + returns).cumprod()`.

### Metrics Calculation Framework

The **MetricsCalculator** computes comprehensive performance statistics. It calculates annualized return using geometric mean over the backtest period, annualized volatility by scaling return standard deviation by the square root of periods per year (8760 for hourly crypto data), Sharpe ratio as excess return over volatility, maximum drawdown by tracking the largest peak-to-trough decline, Calmar ratio as annualized return divided by absolute maximum drawdown, and win rate, profit factor, and average win/loss statistics.

### Main Execution Sequence

The `main.py` file demonstrates the complete pipeline workflow. It fetches and preprocesses data for specified symbols and date ranges, computes unfiltered signals (raw → z-score) for each of the three strategies with hourly parameters (168 hours for momentum lookback = 1 week, 72 hours for mean reversion = 3 days, 24/168 hours for EWMA fast/slow windows), adds unfiltered signals to a Portfolio instance with custom allocation weights (e.g., 40% momentum, 30% mean reversion, 30% EWMA), processes the combined portfolio to generate final weights (combine unfiltered → filter → re-zscore), runs the backtest with configurable rebalancing frequency (default 24 hours = daily) and transaction costs (default 10 bps), and computes all performance metrics.

### Interactive Frontend

The Streamlit app (`streamlit/app.py`) provides a web interface for parameter tuning and visualization. Users can configure universe size, date ranges, strategy-specific parameters (lookbacks, rebalancing frequencies, quantile thresholds), portfolio allocation weights, and transaction costs through sidebar controls. The app displays individual asset cumulative returns, strategy cumulative returns comparison, performance metrics tables, correlation matrices, signal heatmaps showing unfiltered z-scores, returns distributions, and drawdown charts. The interface is accessible online or can be run locally.

