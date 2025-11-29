"""
Streamlit application for cryptocurrency research platform with multi-strategy backtesting.
Uses the src.pipeline classes for signal construction and backtesting.
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

# Import from pipeline module
from src.pipeline import (
    DataPipeline,
    MomentumStrategy,
    MeanReversionStrategy,
    EWMACrossoverStrategy,
    Predictor,
    Portfolio,
    Backtester,
    MetricsCalculator
)


# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Cryptocurrency Research Platform",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Cryptocurrency Research Platform - Multi-Strategy Backtester")

# ===== SIDEBAR: GLOBAL CONTROLS =====
st.sidebar.header("Global Configuration")

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
        value=datetime(2024, 11, 1),
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 12, 31)
    )

with col2:
    end_date = st.date_input(
        "End date",
        value=datetime(2024, 12, 31),
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 12, 31)
    )

# Validate date range
start_date_ts = pd.Timestamp(start_date)
end_date_ts = pd.Timestamp(end_date)

if start_date_ts >= end_date_ts:
    st.sidebar.error("Start date must be before end date")
    st.stop()

st.sidebar.markdown("---")

# ===== MOMENTUM STRATEGY PARAMETERS (Hourly) =====
st.sidebar.subheader("Momentum Strategy")
momentum_lookback = st.sidebar.slider("Momentum Lookback (hours)", 12, 336, 168, key="mom_lookback")  # 168h = 1 week
momentum_rebalance = st.sidebar.slider("Momentum Rebalance (hours)", 1, 168, 24, key="mom_rebal")  # 24h = daily
momentum_top_q_inv = st.sidebar.slider("Momentum Top quantile (long)", 0.10, 0.50, 0.20, key="mom_top")
momentum_top_q = 1 - momentum_top_q_inv
momentum_bottom_q = st.sidebar.slider("Momentum Bottom quantile (short)", 0.10, 0.50, 0.20, key="mom_bot")

st.sidebar.markdown("---")

# ===== MEAN REVERSION STRATEGY PARAMETERS (Hourly) =====
st.sidebar.subheader("Mean Reversion Strategy")
mr_lookback = st.sidebar.slider("Mean Reversion Lookback (hours)", 12, 336, 72, key="mr_lookback")  # 72h = 3 days
mr_rebalance = st.sidebar.slider("Mean Reversion Rebalance (hours)", 1, 168, 12, key="mr_rebal")  # 12h
mr_top_q_inv = st.sidebar.slider("Mean Reversion Top quantile (long)", 0.10, 0.50, 0.30, key="mr_top")
mr_top_q = 1 - mr_top_q_inv
mr_bottom_q = st.sidebar.slider("Mean Reversion Bottom quantile (short)", 0.10, 0.50, 0.30, key="mr_bot")

st.sidebar.markdown("---")

# ===== EWMA CROSSOVER STRATEGY PARAMETERS (Hourly) =====
st.sidebar.subheader("EWMA Crossover Strategy")
ewma_fast_window = st.sidebar.slider("EWMA Fast Window (hours)", 6, 72, 24, key="ewma_fast")  # 24h = 1 day
ewma_slow_window = st.sidebar.slider("EWMA Slow Window (hours)", 48, 336, 168, key="ewma_slow")  # 168h = 1 week
ewma_std_window = st.sidebar.slider("EWMA Std Dev Window (hours)", 12, 168, 72, key="ewma_std")  # 72h = 3 days
ewma_rebalance = st.sidebar.slider("EWMA Rebalance (hours)", 1, 168, 24, key="ewma_rebal")  # 24h = daily
ewma_top_q_inv = st.sidebar.slider("EWMA Top quantile (long)", 0.10, 0.50, 0.20, key="ewma_top")
ewma_top_q = 1 - ewma_top_q_inv
ewma_bottom_q = st.sidebar.slider("EWMA Bottom quantile (short)", 0.10, 0.50, 0.20, key="ewma_bot")

st.sidebar.markdown("---")

# ===== COMBINED PORTFOLIO WEIGHTS =====
st.sidebar.subheader("Combined Portfolio Weights")
momentum_weight = st.sidebar.slider("Momentum Weight", 0.0, 1.0, 0.333, step=0.01, key="mom_wgt")
mr_weight = st.sidebar.slider("Mean Reversion Weight", 0.0, 1.0, 0.333, step=0.01, key="mr_wgt")
ewma_weight = st.sidebar.slider("EWMA Crossover Weight", 0.0, 1.0, 0.334, step=0.01, key="ewma_wgt")

# Normalize weights
total_weight = momentum_weight + mr_weight + ewma_weight
if total_weight > 0:
    momentum_weight /= total_weight
    mr_weight /= total_weight
    ewma_weight /= total_weight

# Combined portfolio quantile filtering
st.sidebar.subheader("Combined Portfolio Filtering")
combined_top_q_inv = st.sidebar.slider("Combined Top quantile (long)", 0.10, 0.50, 0.20, key="comb_top")
combined_top_q = 1 - combined_top_q_inv
combined_bottom_q = st.sidebar.slider("Combined Bottom quantile (short)", 0.10, 0.50, 0.20, key="comb_bot")

st.sidebar.markdown("---")

# ===== TRANSACTION COSTS =====
st.sidebar.subheader("Transaction Costs")
transaction_cost_bps = st.sidebar.number_input(
    "Transaction cost (bps)",
    min_value=0.0,
    value=10.0,
    key="trans_cost"
)

# ===== RUN BACKTEST BUTTON =====
run = st.sidebar.button("Run Backtest")


# ===== DATA FETCHING =====
@st.cache_data
def get_data(universe_size: int, start_date: datetime, end_date: datetime):
    """Fetch data for top symbols in date range using DataPipeline."""
    data_pipeline = DataPipeline()
    
    # Get top symbols
    symbols = data_pipeline.get_top_symbols(universe_size)
    
    # Fetch data
    raw_data = data_pipeline.fetch(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    # Preprocess
    data_pipeline.preprocess()
    
    # Get price matrix
    closes = data_pipeline.get_price_matrix()
    
    return symbols, raw_data, closes


# ===== MAIN EXECUTION =====
if run:
    with st.spinner("Loading data..."):
        symbols, raw_data, closes = get_data(universe_size, start_date, end_date)

    if closes.empty:
        st.error(f"No data available for {start_date} to {end_date}")
        st.stop()

    # Display universe overview
    st.subheader("Universe Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Symbols Loaded", len(closes.columns))
    col2.metric("Data Range", f"{closes.index.min().date()} to {closes.index.max().date()}")
    col3.metric("Hourly Bars", len(closes))

    # ===== COMPUTE SIGNALS USING PIPELINE =====
    with st.spinner("Computing signals..."):
        # Create predictors for each strategy
        momentum_predictor = Predictor(
            strategy=MomentumStrategy(lookback=momentum_lookback),
            top_q=momentum_top_q,
            bottom_q=momentum_bottom_q,
            long_short=True,
            discrete=False
        )
        
        mr_predictor = Predictor(
            strategy=MeanReversionStrategy(lookback=mr_lookback),
            top_q=mr_top_q,
            bottom_q=mr_bottom_q,
            long_short=True,
            discrete=False
        )
        
        ewma_predictor = Predictor(
            strategy=EWMACrossoverStrategy(
                fast_window=ewma_fast_window,
                slow_window=ewma_slow_window,
                std_window=ewma_std_window
            ),
            top_q=ewma_top_q,
            bottom_q=ewma_bottom_q,
            long_short=True,
            discrete=False
        )
        
        # Compute UNFILTERED signals (raw → z-score)
        momentum_unfiltered = momentum_predictor.compute_unfiltered_signal(closes)
        mr_unfiltered = mr_predictor.compute_unfiltered_signal(closes)
        ewma_unfiltered = ewma_predictor.compute_unfiltered_signal(closes)
        
        # Process signals for individual strategy backtests
        momentum_signals = momentum_predictor.process_signal()
        mr_signals = mr_predictor.process_signal()
        ewma_signals = ewma_predictor.process_signal()

    # ===== RUN BACKTESTS =====
    with st.spinner("Running backtests..."):
        metrics_calc = MetricsCalculator()
        
        # Momentum backtest
        bt_momentum = Backtester(
            rebalance_frequency=momentum_rebalance,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=5.0
        )
        result_momentum = bt_momentum.run(momentum_signals, closes)
        
        # Mean Reversion backtest
        bt_mr = Backtester(
            rebalance_frequency=mr_rebalance,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=5.0
        )
        result_mr = bt_mr.run(mr_signals, closes)
        
        # EWMA backtest
        bt_ewma = Backtester(
            rebalance_frequency=ewma_rebalance,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=5.0
        )
        result_ewma = bt_ewma.run(ewma_signals, closes)
        
        # Combined portfolio using unfiltered signals (z-scored)
        portfolio = Portfolio(
            predictor_weights={
                "momentum": momentum_weight,
                "mean_reversion": mr_weight,
                "ewma": ewma_weight
            },
            top_q=combined_top_q,
            bottom_q=combined_bottom_q,
            long_short=True,
            discrete=False
        )
        
        portfolio.add_predictor_unfiltered_signal("momentum", momentum_unfiltered)
        portfolio.add_predictor_unfiltered_signal("mean_reversion", mr_unfiltered)
        portfolio.add_predictor_unfiltered_signal("ewma", ewma_unfiltered)
        
        combined_weights = portfolio.combine_and_process()
        
        # Use average rebalance frequency for combined
        avg_rebalance = int((momentum_rebalance + mr_rebalance + ewma_rebalance) / 3)
        bt_combined = Backtester(
            rebalance_frequency=avg_rebalance,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=5.0
        )
        result_combined = bt_combined.run(combined_weights, closes)

    # ===== INDIVIDUAL ASSET CUMULATIVE RETURNS =====
    st.subheader("Individual Asset Cumulative Returns")
    
    # Compute daily returns and cumulative returns starting at 1
    asset_returns = closes.pct_change().fillna(0)
    asset_cumulative = (1 + asset_returns).cumprod()
    
    fig_assets = go.Figure()
    
    # Color palette for assets
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
    
    for i, col in enumerate(asset_cumulative.columns):
        fig_assets.add_trace(go.Scatter(
            x=asset_cumulative.index,
            y=asset_cumulative[col].values,
            mode='lines',
            name=col.replace('/USDT', ''),
            line=dict(width=2, color=colors[i % len(colors)])
        ))
    
    fig_assets.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500
    )
    
    st.plotly_chart(fig_assets, use_container_width=True)

    # ===== CUMULATIVE RETURNS CHART =====
    st.subheader("Cumulative Returns: All Strategies")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result_momentum.cumulative_returns.index,
        y=result_momentum.cumulative_returns.values,
        mode='lines',
        name='Momentum',
        line=dict(width=2, color='#0066CC')
    ))

    fig.add_trace(go.Scatter(
        x=result_mr.cumulative_returns.index,
        y=result_mr.cumulative_returns.values,
        mode='lines',
        name='Mean Reversion',
        line=dict(width=2, color='#FF8C00')
    ))

    fig.add_trace(go.Scatter(
        x=result_ewma.cumulative_returns.index,
        y=result_ewma.cumulative_returns.values,
        mode='lines',
        name='EWMA Crossover',
        line=dict(width=2, color='#00AA00')
    ))

    fig.add_trace(go.Scatter(
        x=result_combined.cumulative_returns.index,
        y=result_combined.cumulative_returns.values,
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

    st.plotly_chart(fig, use_container_width=True)

    # ===== PERFORMANCE METRICS =====
    st.subheader("Performance Metrics")

    metrics_momentum = result_momentum.metrics
    metrics_mr = result_mr.metrics
    metrics_ewma = result_ewma.metrics
    metrics_combined = result_combined.metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### Momentum")
        st.metric("Annual Return", f"{metrics_momentum.annual_return:.2%}")
        st.metric("Annual Vol", f"{metrics_momentum.annual_volatility:.2%}")
        st.metric("Sharpe", f"{metrics_momentum.sharpe_ratio:.2f}")
        st.metric("Max DD", f"{metrics_momentum.max_drawdown:.2%}")

    with col2:
        st.markdown("#### Mean Reversion")
        st.metric("Annual Return", f"{metrics_mr.annual_return:.2%}")
        st.metric("Annual Vol", f"{metrics_mr.annual_volatility:.2%}")
        st.metric("Sharpe", f"{metrics_mr.sharpe_ratio:.2f}")
        st.metric("Max DD", f"{metrics_mr.max_drawdown:.2%}")

    with col3:
        st.markdown("#### EWMA Crossover")
        st.metric("Annual Return", f"{metrics_ewma.annual_return:.2%}")
        st.metric("Annual Vol", f"{metrics_ewma.annual_volatility:.2%}")
        st.metric("Sharpe", f"{metrics_ewma.sharpe_ratio:.2f}")
        st.metric("Max DD", f"{metrics_ewma.max_drawdown:.2%}")

    with col4:
        st.markdown("#### Combined")
        st.metric("Annual Return", f"{metrics_combined.annual_return:.2%}")
        st.metric("Annual Vol", f"{metrics_combined.annual_volatility:.2%}")
        st.metric("Sharpe", f"{metrics_combined.sharpe_ratio:.2f}")
        st.metric("Max DD", f"{metrics_combined.max_drawdown:.2%}")

    # ===== DETAILED METRICS TABLE =====
    st.subheader("Detailed Metrics Comparison")

    metrics_table = pd.DataFrame({
        'Momentum': [
            f"{metrics_momentum.annual_return:.2%}",
            f"{metrics_momentum.annual_volatility:.2%}",
            f"{metrics_momentum.sharpe_ratio:.2f}",
            f"{metrics_momentum.max_drawdown:.2%}",
            f"{metrics_momentum.calmar_ratio:.2f}",
            f"{metrics_momentum.win_rate:.2%}",
            f"{metrics_momentum.profit_factor:.2f}"
        ],
        'Mean Reversion': [
            f"{metrics_mr.annual_return:.2%}",
            f"{metrics_mr.annual_volatility:.2%}",
            f"{metrics_mr.sharpe_ratio:.2f}",
            f"{metrics_mr.max_drawdown:.2%}",
            f"{metrics_mr.calmar_ratio:.2f}",
            f"{metrics_mr.win_rate:.2%}",
            f"{metrics_mr.profit_factor:.2f}"
        ],
        'EWMA Crossover': [
            f"{metrics_ewma.annual_return:.2%}",
            f"{metrics_ewma.annual_volatility:.2%}",
            f"{metrics_ewma.sharpe_ratio:.2f}",
            f"{metrics_ewma.max_drawdown:.2%}",
            f"{metrics_ewma.calmar_ratio:.2f}",
            f"{metrics_ewma.win_rate:.2%}",
            f"{metrics_ewma.profit_factor:.2f}"
        ],
        'Combined': [
            f"{metrics_combined.annual_return:.2%}",
            f"{metrics_combined.annual_volatility:.2%}",
            f"{metrics_combined.sharpe_ratio:.2f}",
            f"{metrics_combined.max_drawdown:.2%}",
            f"{metrics_combined.calmar_ratio:.2f}",
            f"{metrics_combined.win_rate:.2%}",
            f"{metrics_combined.profit_factor:.2f}"
        ]
    }, index=['Annual Return', 'Annual Vol', 'Sharpe', 'Max Drawdown', 
              'Calmar Ratio', 'Win Rate', 'Profit Factor'])

    st.dataframe(metrics_table, use_container_width=True)

    # ===== CORRELATION MATRIX =====
    st.subheader("Strategy Correlation Matrix")

    returns_df = pd.DataFrame({
        'Momentum': result_momentum.daily_returns,
        'Mean Reversion': result_mr.daily_returns,
        'EWMA Crossover': result_ewma.daily_returns,
        'Combined': result_combined.daily_returns
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

    st.plotly_chart(fig_corr, use_container_width=True)

    # ===== PORTFOLIO WEIGHTS =====
    st.subheader("Combined Portfolio Composition")

    weights_info = pd.DataFrame({
        'Strategy': ['Momentum', 'Mean Reversion', 'EWMA Crossover'],
        'Weight': [
            f"{momentum_weight:.1%}",
            f"{mr_weight:.1%}",
            f"{ewma_weight:.1%}"
        ]
    })

    st.dataframe(weights_info, use_container_width=True)

    st.info(
        f"""
        **Combined Portfolio:**
        - Allocation: {momentum_weight:.1%} Momentum + {mr_weight:.1%} Mean Reversion + {ewma_weight:.1%} EWMA
        - Return: {metrics_combined.annual_return:.2%} annually
        - Volatility: {metrics_combined.annual_volatility:.2%} annually
        - Sharpe: {metrics_combined.sharpe_ratio:.2f}
        - Max Drawdown: {metrics_combined.max_drawdown:.2%}
        """
    )

    # ===== SIGNAL HEATMAPS =====
    st.subheader("Strategy Signals - Unfiltered Z-Scores (Latest 50 Hours)")

    # Calculate global min/max across all unfiltered signals
    all_signals = pd.concat([
        momentum_unfiltered.iloc[-50:],
        mr_unfiltered.iloc[-50:],
        ewma_unfiltered.iloc[-50:]
    ])
    vmin = all_signals.min().min()
    vmax = all_signals.max().max()

    tabs = st.tabs(["Momentum", "Mean Reversion", "EWMA Crossover"])

    with tabs[0]:
        recent_momentum = momentum_unfiltered.iloc[-50:].T
        fig_hm = px.imshow(
            recent_momentum,
            color_continuous_scale='RdBu_r',
            zmin=vmin,
            zmax=vmax,
            text_auto='.2f',
            labels=dict(x="Date", y="Symbol", color="Z-Score")
        )
        fig_hm.update_layout(height=400)
        st.plotly_chart(fig_hm, use_container_width=True)

    with tabs[1]:
        recent_mr = mr_unfiltered.iloc[-50:].T
        fig_hm = px.imshow(
            recent_mr,
            color_continuous_scale='RdBu_r',
            zmin=vmin,
            zmax=vmax,
            text_auto='.2f',
            labels=dict(x="Date", y="Symbol", color="Z-Score")
        )
        fig_hm.update_layout(height=400)
        st.plotly_chart(fig_hm, use_container_width=True)

    with tabs[2]:
        recent_ewma = ewma_unfiltered.iloc[-50:].T
        fig_hm = px.imshow(
            recent_ewma,
            color_continuous_scale='RdBu_r',
            zmin=vmin,
            zmax=vmax,
            text_auto='.2f',
            labels=dict(x="Date", y="Symbol", color="Z-Score")
        )
        fig_hm.update_layout(height=400)
        st.plotly_chart(fig_hm, use_container_width=True)

    # ===== HOURLY RETURNS DISTRIBUTION =====
    st.subheader("Hourly Returns Distribution")

    fig_dist = go.Figure()

    fig_dist.add_trace(go.Histogram(
        x=result_momentum.daily_returns,
        name='Momentum',
        opacity=0.7,
        nbinsx=50
    ))

    fig_dist.add_trace(go.Histogram(
        x=result_mr.daily_returns,
        name='Mean Reversion',
        opacity=0.7,
        nbinsx=50
    ))

    fig_dist.add_trace(go.Histogram(
        x=result_ewma.daily_returns,
        name='EWMA Crossover',
        opacity=0.7,
        nbinsx=50
    ))

    fig_dist.add_trace(go.Histogram(
        x=result_combined.daily_returns,
        name='Combined',
        opacity=0.7,
        nbinsx=50
    ))

    fig_dist.update_layout(
        barmode='overlay',
        xaxis_title='Hourly Returns',
        yaxis_title='Frequency',
        height=400
    )

    st.plotly_chart(fig_dist, use_container_width=True)

    # ===== DRAWDOWN CHART =====
    st.subheader("Drawdown Analysis")
    
    fig_dd = go.Figure()
    
    # Compute drawdowns
    dd_momentum = metrics_calc.rolling_drawdown(result_momentum.daily_returns)
    dd_mr = metrics_calc.rolling_drawdown(result_mr.daily_returns)
    dd_ewma = metrics_calc.rolling_drawdown(result_ewma.daily_returns)
    dd_combined = metrics_calc.rolling_drawdown(result_combined.daily_returns)
    
    fig_dd.add_trace(go.Scatter(
        x=dd_momentum.index,
        y=dd_momentum.values,
        mode='lines',
        name='Momentum',
        line=dict(width=1, color='#0066CC'),
        fill='tozeroy'
    ))
    
    fig_dd.add_trace(go.Scatter(
        x=dd_mr.index,
        y=dd_mr.values,
        mode='lines',
        name='Mean Reversion',
        line=dict(width=1, color='#FF8C00'),
        fill='tozeroy'
    ))
    
    fig_dd.add_trace(go.Scatter(
        x=dd_ewma.index,
        y=dd_ewma.values,
        mode='lines',
        name='EWMA Crossover',
        line=dict(width=1, color='#00AA00'),
        fill='tozeroy'
    ))
    
    fig_dd.add_trace(go.Scatter(
        x=dd_combined.index,
        y=dd_combined.values,
        mode='lines',
        name='Combined',
        line=dict(width=2, color='#DD0000', dash='dash')
    ))
    
    fig_dd.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)

else:
    st.info("Adjust parameters and press Run Backtest to start.")
