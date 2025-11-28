"""
Streamlit application for cryptocurrency research platform with multi-strategy backtesting.
Includes Momentum, Mean Reversion, and EWMA Crossover strategies.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Import from existing modules
from src.data_fetch import get_top_symbols, fetch_ohlcv_date_range
from src.preprocessing import align_universe
from src.factors import momentum, mean_reversion_zscore
from src.backtest import backtest_signals
from src.metrics import compute_metrics


# ===== NEW EWMA CROSSOVER FACTOR =====
def ewma_crossover(close: pd.Series, fast_window: int = 12, 
                   slow_window: int = 26, std_window: int = 20) -> pd.Series:
    """
    EWMA Crossover Strategy

    Signal = (fast_ewma - slow_ewma) / rolling_std

    Args:
        close: Series of closing prices
        fast_window: Window for fast EWMA
        slow_window: Window for slow EWMA
        std_window: Window for rolling standard deviation

    Returns:
        Series of normalized signals
    """
    fast_ewma = close.ewm(span=fast_window).mean()
    slow_ewma = close.ewm(span=slow_window).mean()
    rolling_std = close.rolling(window=std_window).std().replace(0, np.nan)

    signal = (fast_ewma - slow_ewma) / rolling_std
    return signal.shift(1).fillna(0)


# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Cryptocurrency Research Platform",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Cryptocurrency Research Platform - Multi-Strategy Backtester")

# ===== SIDEBAR: GLOBAL CONTROLS =====
st.sidebar.header("🔧 Global Configuration")

universe_size = st.sidebar.selectbox(
    "Universe (top N symbols by volume)",
    [10, 20, 30],
    index=0
)

# Date range selection
st.sidebar.subheader("Date Range")
col1, col2 = st.sidebar.columns(2)

with col1:
    start_date = st.date_input(
        "Start date",
        value=datetime(2023, 1, 1),
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2024, 12, 31)
    )

with col2:
    end_date = st.date_input(
        "End date",
        value=datetime(2024, 12, 31),
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2024, 12, 31)
    )

# Validate date range
start_date_ts = pd.Timestamp(start_date)
end_date_ts = pd.Timestamp(end_date)

if start_date_ts >= end_date_ts:
    st.sidebar.error("Start date must be before end date")
    st.stop()

st.sidebar.markdown("---")

# ===== MOMENTUM STRATEGY PARAMETERS =====
st.sidebar.subheader("🔵 Momentum Strategy")
momentum_lookback = st.sidebar.slider("Momentum Lookback", 5, 120, 21, key="mom_lookback")
momentum_rebalance = st.sidebar.slider("Momentum Rebalance (days)", 1, 63, 21, key="mom_rebal")
momentum_top_q_inv = st.sidebar.slider("Momentum Top quantile (long)", 0.10, 0.50, 0.20, key="mom_top")
momentum_top_q = 1 - momentum_top_q_inv
momentum_bottom_q = st.sidebar.slider("Momentum Bottom quantile (short)", 0.10, 0.50, 0.20, key="mom_bot")

st.sidebar.markdown("---")

# ===== MEAN REVERSION STRATEGY PARAMETERS =====
st.sidebar.subheader("🟠 Mean Reversion Strategy")
mr_lookback = st.sidebar.slider("Mean Reversion Lookback", 5, 120, 14, key="mr_lookback")
mr_rebalance = st.sidebar.slider("Mean Reversion Rebalance (days)", 1, 63, 7, key="mr_rebal")
mr_top_q_inv = st.sidebar.slider("Mean Reversion Top quantile (long)", 0.10, 0.50, 0.30, key="mr_top")
mr_top_q = 1 - mr_top_q_inv
mr_bottom_q = st.sidebar.slider("Mean Reversion Bottom quantile (short)", 0.10, 0.50, 0.30, key="mr_bot")

st.sidebar.markdown("---")

# ===== EWMA CROSSOVER STRATEGY PARAMETERS =====
st.sidebar.subheader("🟢 EWMA Crossover Strategy")
ewma_fast_window = st.sidebar.slider("EWMA Fast Window", 5, 50, 12, key="ewma_fast")
ewma_slow_window = st.sidebar.slider("EWMA Slow Window", 20, 200, 26, key="ewma_slow")
ewma_std_window = st.sidebar.slider("EWMA Std Dev Window", 10, 100, 20, key="ewma_std")
ewma_rebalance = st.sidebar.slider("EWMA Rebalance (days)", 1, 63, 21, key="ewma_rebal")
ewma_top_q_inv = st.sidebar.slider("EWMA Top quantile (long)", 0.10, 0.50, 0.20, key="ewma_top")
ewma_top_q = 1 - ewma_top_q_inv
ewma_bottom_q = st.sidebar.slider("EWMA Bottom quantile (short)", 0.10, 0.50, 0.20, key="ewma_bot")

st.sidebar.markdown("---")

# ===== COMBINED PORTFOLIO WEIGHTS =====
st.sidebar.subheader("🎯 Combined Portfolio Weights")
momentum_weight = st.sidebar.slider("Momentum Weight", 0.0, 1.0, 0.333, step=0.01, key="mom_wgt")
mr_weight = st.sidebar.slider("Mean Reversion Weight", 0.0, 1.0, 0.333, step=0.01, key="mr_wgt")
ewma_weight = st.sidebar.slider("EWMA Crossover Weight", 0.0, 1.0, 0.334, step=0.01, key="ewma_wgt")

# Normalize weights
total_weight = momentum_weight + mr_weight + ewma_weight
if total_weight > 0:
    momentum_weight /= total_weight
    mr_weight /= total_weight
    ewma_weight /= total_weight

st.sidebar.markdown("---")

# ===== TRANSACTION COSTS =====
st.sidebar.subheader("💰 Transaction Costs")
transaction_cost_bps = st.sidebar.number_input(
    "Transaction cost (bps)",
    min_value=0.0,
    value=10.0,
    key="trans_cost"
) / 10000.0

# ===== RUN BACKTEST BUTTON =====
run = st.sidebar.button("▶️ Run Backtest")


# ===== DATA FETCHING =====
@st.cache_data
def get_data(universe_size: int, start_date: datetime, end_date: datetime):
    """Fetch data for top symbols in date range."""
    symbols = get_top_symbols(universe_size)
    raw = {}

    for sym in symbols:
        try:
            data = fetch_ohlcv_date_range(sym, start_date, end_date)
            if not data.empty:
                raw[sym] = data
        except Exception as e:
            st.warning(f"Failed to fetch {sym}: {str(e)}")
            continue

    if not raw:
        return symbols, raw, pd.DataFrame()

    closes = align_universe(raw)
    return symbols, raw, closes


# ===== SIGNAL COMPUTATION =====
def compute_momentum_signals_df(closes: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Compute momentum signals for all symbols."""
    signals = pd.DataFrame(index=closes.index, columns=closes.columns)

    for sym in closes.columns:
        signals[sym] = momentum(closes[sym], lookback=lookback)

    return signals.fillna(0)


def compute_mean_reversion_signals_df(closes: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Compute mean reversion signals for all symbols."""
    signals = pd.DataFrame(index=closes.index, columns=closes.columns)

    for sym in closes.columns:
        signals[sym] = mean_reversion_zscore(closes[sym], lookback=lookback)

    return signals.fillna(0)


def compute_ewma_signals_df(closes: pd.DataFrame, fast_window: int, 
                            slow_window: int, std_window: int) -> pd.DataFrame:
    """Compute EWMA crossover signals for all symbols."""
    signals = pd.DataFrame(index=closes.index, columns=closes.columns)

    for sym in closes.columns:
        signals[sym] = ewma_crossover(closes[sym], fast_window, slow_window, std_window)

    return signals.fillna(0)


# ===== MAIN EXECUTION =====
if run:
    with st.spinner("Loading data..."):
        symbols, raw_data, closes = get_data(universe_size, start_date, end_date)

    if closes.empty:
        st.error(f"No data available for {start_date} to {end_date}")
        st.stop()

    # Display universe overview
    st.subheader("📊 Universe Overview")
    
    st.metric("Symbols Loaded", len(closes.columns))
    st.metric("Data Range", f"{closes.index.min().date()} to {closes.index.max().date()}")
    st.metric("Trading Days", len(closes))

    # Compute signals
    with st.spinner("Computing signals..."):
        momentum_signals = compute_momentum_signals_df(closes, momentum_lookback)
        mr_signals = compute_mean_reversion_signals_df(closes, mr_lookback)
        ewma_signals = compute_ewma_signals_df(closes, ewma_fast_window, ewma_slow_window, ewma_std_window)

    # Run backtests
    with st.spinner("Running backtests..."):
        bt_momentum = backtest_signals(
            momentum_signals, closes,
            top_q=momentum_top_q, bottom_q=momentum_bottom_q,
            rebalance_every=momentum_rebalance,
            transaction_cost_bps=transaction_cost_bps
        )

        bt_mr = backtest_signals(
            mr_signals, closes,
            top_q=mr_top_q, bottom_q=mr_bottom_q,
            rebalance_every=mr_rebalance,
            transaction_cost_bps=transaction_cost_bps
        )

        bt_ewma = backtest_signals(
            ewma_signals, closes,
            top_q=ewma_top_q, bottom_q=ewma_bottom_q,
            rebalance_every=ewma_rebalance,
            transaction_cost_bps=transaction_cost_bps
        )

        # Combined portfolio
        bt_combined = {
            "weights": (
                momentum_weight * bt_momentum["weights"] +
                mr_weight * bt_mr["weights"] +
                ewma_weight * bt_ewma["weights"]
            ),
            "daily_returns": (
                momentum_weight * bt_momentum["daily_returns"] +
                mr_weight * bt_mr["daily_returns"] +
                ewma_weight * bt_ewma["daily_returns"]
            ),
        }
        bt_combined["cumulative"] = (1 + bt_combined["daily_returns"]).cumprod()

    # ===== CUMULATIVE RETURNS CHART =====
    st.subheader("📈 Cumulative Returns: All Strategies")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bt_momentum["cumulative"].index,
        y=bt_momentum["cumulative"].values,
        mode='lines',
        name='Momentum',
        line=dict(width=2, color='#0066CC')
    ))

    fig.add_trace(go.Scatter(
        x=bt_mr["cumulative"].index,
        y=bt_mr["cumulative"].values,
        mode='lines',
        name='Mean Reversion',
        line=dict(width=2, color='#FF8C00')
    ))

    fig.add_trace(go.Scatter(
        x=bt_ewma["cumulative"].index,
        y=bt_ewma["cumulative"].values,
        mode='lines',
        name='EWMA Crossover',
        line=dict(width=2, color='#00AA00')
    ))

    fig.add_trace(go.Scatter(
        x=bt_combined["cumulative"].index,
        y=bt_combined["cumulative"].values,
        mode='lines',
        name='Combined Portfolio',
        line=dict(width=3, color='#DD0000', dash='dash')
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500
    )

    st.plotly_chart(fig, width='stretch')

    # ===== PERFORMANCE METRICS =====
    st.subheader("📊 Performance Metrics")

    metrics_momentum = compute_metrics(bt_momentum["daily_returns"])
    metrics_mr = compute_metrics(bt_mr["daily_returns"])
    metrics_ewma = compute_metrics(bt_ewma["daily_returns"])
    metrics_combined = compute_metrics(bt_combined["daily_returns"])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### 🔵 Momentum")
        st.metric("Annual Return", f"{metrics_momentum['annual_return']:.2%}")
        st.metric("Annual Vol", f"{metrics_momentum['annual_vol']:.2%}")
        st.metric("Sharpe", f"{metrics_momentum['sharpe']:.2f}")
        st.metric("Max DD", f"{metrics_momentum['max_drawdown']:.2%}")

    with col2:
        st.markdown("#### 🟠 Mean Reversion")
        st.metric("Annual Return", f"{metrics_mr['annual_return']:.2%}")
        st.metric("Annual Vol", f"{metrics_mr['annual_vol']:.2%}")
        st.metric("Sharpe", f"{metrics_mr['sharpe']:.2f}")
        st.metric("Max DD", f"{metrics_mr['max_drawdown']:.2%}")

    with col3:
        st.markdown("#### 🟢 EWMA Crossover")
        st.metric("Annual Return", f"{metrics_ewma['annual_return']:.2%}")
        st.metric("Annual Vol", f"{metrics_ewma['annual_vol']:.2%}")
        st.metric("Sharpe", f"{metrics_ewma['sharpe']:.2f}")
        st.metric("Max DD", f"{metrics_ewma['max_drawdown']:.2%}")

    with col4:
        st.markdown("#### 🎯 Combined")
        st.metric("Annual Return", f"{metrics_combined['annual_return']:.2%}")
        st.metric("Annual Vol", f"{metrics_combined['annual_vol']:.2%}")
        st.metric("Sharpe", f"{metrics_combined['sharpe']:.2f}")
        st.metric("Max DD", f"{metrics_combined['max_drawdown']:.2%}")

    # ===== DETAILED METRICS TABLE =====
    st.subheader("📋 Detailed Metrics Comparison")

    metrics_table = pd.DataFrame({
        'Momentum': [
            f"{metrics_momentum['annual_return']:.2%}",
            f"{metrics_momentum['annual_vol']:.2%}",
            f"{metrics_momentum['sharpe']:.2f}",
            f"{metrics_momentum['max_drawdown']:.2%}"
        ],
        'Mean Reversion': [
            f"{metrics_mr['annual_return']:.2%}",
            f"{metrics_mr['annual_vol']:.2%}",
            f"{metrics_mr['sharpe']:.2f}",
            f"{metrics_mr['max_drawdown']:.2%}"
        ],
        'EWMA Crossover': [
            f"{metrics_ewma['annual_return']:.2%}",
            f"{metrics_ewma['annual_vol']:.2%}",
            f"{metrics_ewma['sharpe']:.2f}",
            f"{metrics_ewma['max_drawdown']:.2%}"
        ],
        'Combined': [
            f"{metrics_combined['annual_return']:.2%}",
            f"{metrics_combined['annual_vol']:.2%}",
            f"{metrics_combined['sharpe']:.2f}",
            f"{metrics_combined['max_drawdown']:.2%}"
        ]
    }, index=['Annual Return', 'Annual Vol', 'Sharpe', 'Max Drawdown'])

    st.dataframe(metrics_table, width='stretch')

    # ===== CORRELATION MATRIX =====
    st.subheader("🔗 Strategy Correlation Matrix")

    returns_df = pd.DataFrame({
        'Momentum': bt_momentum["daily_returns"],
        'Mean Reversion': bt_mr["daily_returns"],
        'EWMA Crossover': bt_ewma["daily_returns"],
        'Combined': bt_combined["daily_returns"]
    })

    corr_matrix = returns_df.corr()

    fig_corr = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        text_auto='.3f',
        labels=dict(x="Strategy", y="Strategy", color="Correlation")
    )

    fig_corr.update_layout(
        title="Correlation of Daily Returns",
        width=600,
        height=500
    )

    st.plotly_chart(fig_corr, width='stretch')

    # ===== PORTFOLIO WEIGHTS =====
    st.subheader("🎯 Combined Portfolio Composition")

    weights_info = pd.DataFrame({
        'Strategy': ['Momentum', 'Mean Reversion', 'EWMA Crossover'],
        'Weight': [
            f"{momentum_weight:.1%}",
            f"{mr_weight:.1%}",
            f"{ewma_weight:.1%}"
        ]
    })

    st.dataframe(weights_info, width='stretch')

    st.info(
        f"""
        **Combined Portfolio:**
        - Allocation: {momentum_weight:.1%} Momentum + {mr_weight:.1%} Mean Reversion + {ewma_weight:.1%} EWMA
        - Return: {metrics_combined['annual_return']:.2%} annually
        - Volatility: {metrics_combined['annual_vol']:.2%} annually
        - Sharpe: {metrics_combined['sharpe']:.2f}
        - Max Drawdown: {metrics_combined['max_drawdown']:.2%}
        """
    )

    # ===== SIGNAL HEATMAPS =====
    st.subheader("🔥 Strategy Signals (Latest 50 Bars)")

    # Calculate global min/max across all signals
    all_signals = pd.concat([
        momentum_signals.iloc[-50:],
        mr_signals.iloc[-50:],
        ewma_signals.iloc[-50:]
    ])
    vmin = all_signals.min().min()
    vmax = all_signals.max().max()

    tabs = st.tabs(["Momentum", "Mean Reversion", "EWMA Crossover"])

    with tabs[0]:
        recent_momentum = momentum_signals.iloc[-50:].T
        fig_hm = px.imshow(
            recent_momentum,
            color_continuous_scale='RdBu_r',
            zmin=vmin,
            zmax=vmax,
            text_auto='.2f',
            labels=dict(x="Date", y="Symbol", color="Signal")
        )
        fig_hm.update_layout(height=400)
        st.plotly_chart(fig_hm, width='stretch')

    with tabs[1]:
        recent_mr = mr_signals.iloc[-50:].T
        fig_hm = px.imshow(
            recent_mr,
            color_continuous_scale='RdBu_r',
            zmin=vmin,
            zmax=vmax,
            text_auto='.2f',
            labels=dict(x="Date", y="Symbol", color="Signal")
        )
        fig_hm.update_layout(height=400)
        st.plotly_chart(fig_hm, width='stretch')

    with tabs[2]:
        recent_ewma = ewma_signals.iloc[-50:].T
        fig_hm = px.imshow(
            recent_ewma,
            color_continuous_scale='RdBu_r',
            zmin=vmin,
            zmax=vmax,
            text_auto='.2f',
            labels=dict(x="Date", y="Symbol", color="Signal")
        )
        fig_hm.update_layout(height=400)
        st.plotly_chart(fig_hm, width='stretch')

    # ===== DAILY RETURNS DISTRIBUTION =====
    st.subheader("📊 Daily Returns Distribution")

    fig_dist = go.Figure()

    fig_dist.add_trace(go.Histogram(
        x=bt_momentum["daily_returns"],
        name='Momentum',
        opacity=0.7,
        nbinsx=50
    ))

    fig_dist.add_trace(go.Histogram(
        x=bt_mr["daily_returns"],
        name='Mean Reversion',
        opacity=0.7,
        nbinsx=50
    ))

    fig_dist.add_trace(go.Histogram(
        x=bt_ewma["daily_returns"],
        name='EWMA Crossover',
        opacity=0.7,
        nbinsx=50
    ))

    fig_dist.add_trace(go.Histogram(
        x=bt_combined["daily_returns"],
        name='Combined',
        opacity=0.7,
        nbinsx=50
    ))

    fig_dist.update_layout(
        barmode='overlay',
        xaxis_title='Daily Returns',
        yaxis_title='Frequency',
        height=400
    )

    st.plotly_chart(fig_dist, width='stretch')

else:
    st.info("Adjust parameters and press ▶️ Run Backtest to start.")
