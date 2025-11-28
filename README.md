# Cryptocurrency Research Platform

## Overview

The **Cryptocurrency Research Platform** is a backtesting and analysis framework for cryptocurrency trading strategies. It implements three complementary factor-based strategies:

1. **Momentum Strategy** - Captures trending behavior
2. **Mean Reversion Strategy** - Exploits price extremes
3. **EWMA Crossover Strategy** - Identifies trend changes via exponential moving average divergences

These strategies operate on a universe of top-performing cryptocurrencies (by trading volume) and can be combined into a diversified multi-strategy portfolio. The platform supports:

- **Live data fetching** from Binance via CCXT
- **Cross-sectional signal ranking** for predictor construction
- **Portfolio construction** combining the predictors
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

