"""
Cryptocurrency Research Platform - Main Entry Point

Example usage of the pipeline architecture:
1. DataPipeline: Fetch and preprocess data
2. Predictor (x3): Compute UNFILTERED signals for each strategy (raw → z-score)
3. Portfolio: Combine unfiltered signals → filter → re-zscore → weights
4. TargetEngineer: Compute 1-day returns
5. Backtester: Run backtest
6. MetricsCalculator: Compute performance metrics
"""

from datetime import datetime, timedelta
from src.pipeline import (
    DataPipeline,
    MomentumStrategy,
    MeanReversionStrategy,
    EWMACrossoverStrategy,
    Predictor,
    Portfolio,
    FeatureEngineer,
    TargetEngineer,
    MetricsCalculator,
    Backtester
)


def main():
    """Run the full pipeline."""
    
    # ==========================================================================
    # 1. DATA EXTRACTION & PREPROCESSING
    # ==========================================================================
    print("=" * 60)
    print("Step 1: Data Extraction & Preprocessing")
    print("=" * 60)
    
    data_pipeline = DataPipeline()
    
    # Define symbols and date range
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", 
               "SOL/USDT", "DOGE/USDT"]
    
    # Default: Nov 1, 2024 to Dec 31, 2024 (2 months of hourly data)
    start_date = datetime(2024, 11, 1)
    end_date = datetime(2024, 12, 31)
    
    # Fetch data
    raw_data = data_pipeline.fetch(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    print(f"Fetched data for {len(raw_data)} symbols")
    
    # Preprocess
    cleaned_data = data_pipeline.preprocess()
    print(f"Preprocessed {len(cleaned_data)} symbols")
    
    # Get price matrix
    price_matrix = data_pipeline.get_price_matrix()
    print(f"Price matrix shape: {price_matrix.shape}")
    print(f"Date range: {price_matrix.index[0]} to {price_matrix.index[-1]}")
    
    # ==========================================================================
    # 2. FEATURE ENGINEERING (3 strategies) - Hourly parameters
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Feature Engineering")
    print("=" * 60)
    
    # Hourly lookbacks: 24h = 1 day, 168h = 1 week, etc.
    feature_engineer = FeatureEngineer(
        momentum_lookback=168,       # 1 week (168 hours)
        mean_reversion_lookback=72,  # 3 days (72 hours)
        ewma_fast=24,                # 1 day
        ewma_slow=168,               # 1 week
        ewma_std=72                  # 3 days
    )
    
    features = feature_engineer.compute_features(price_matrix)
    for name, df in features.items():
        print(f"  - {name}: {df.shape}")
    
    # ==========================================================================
    # 3. TARGET ENGINEERING (1-day returns)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Target Engineering")
    print("=" * 60)
    
    target_engineer = TargetEngineer(forward_period=1)
    targets = target_engineer.compute_targets(price_matrix)
    print(f"Targets shape: {targets.shape}")
    
    # ==========================================================================
    # 4. CREATE PREDICTORS (1 predictor = 1 signal strategy)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Create Predictors")
    print("=" * 60)
    
    # Create 3 predictors, one for each strategy (hourly lookbacks)
    momentum_predictor = Predictor(
        strategy=MomentumStrategy(lookback=168),  # 1 week
        top_q=0.8,
        bottom_q=0.2,
        long_short=True,
        discrete=False
    )
    
    mean_reversion_predictor = Predictor(
        strategy=MeanReversionStrategy(lookback=72),  # 3 days
        top_q=0.8,
        bottom_q=0.2,
        long_short=True,
        discrete=False
    )
    
    ewma_predictor = Predictor(
        strategy=EWMACrossoverStrategy(fast_window=24, slow_window=168, std_window=72),
        top_q=0.8,
        bottom_q=0.2,
        long_short=True,
        discrete=False
    )
    
    # Compute UNFILTERED signals (raw → z-score)
    momentum_unfiltered = momentum_predictor.compute_unfiltered_signal(price_matrix)
    mean_reversion_unfiltered = mean_reversion_predictor.compute_unfiltered_signal(price_matrix)
    ewma_unfiltered = ewma_predictor.compute_unfiltered_signal(price_matrix)
    
    print(f"  - Momentum UNFILTERED signals: {momentum_unfiltered.shape}")
    print(f"  - Mean Reversion UNFILTERED signals: {mean_reversion_unfiltered.shape}")
    print(f"  - EWMA UNFILTERED signals: {ewma_unfiltered.shape}")
    
    # ==========================================================================
    # 5. PORTFOLIO CONSTRUCTION (combine UNFILTERED signals → process)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Step 5: Portfolio Construction")
    print("=" * 60)
    
    # Create portfolio with custom weights and filtering parameters
    portfolio = Portfolio(
        predictor_weights={
            "momentum": 0.4,
            "mean_reversion": 0.3,
            "ewma_crossover": 0.3
        },
        top_q=0.8,
        bottom_q=0.2,
        long_short=True,
        discrete=False,
        enable_vol_target=False  # Turned off for now
    )
    
    # Add UNFILTERED signals from each predictor (z-scored but not filtered)
    portfolio.add_predictor_unfiltered_signal("momentum", momentum_unfiltered)
    portfolio.add_predictor_unfiltered_signal("mean_reversion", mean_reversion_unfiltered)
    portfolio.add_predictor_unfiltered_signal("ewma_crossover", ewma_unfiltered)
    
    # Process: combine unfiltered → filter → re-zscore → weights
    portfolio_weights = portfolio.combine_and_process()
    
    print(f"Combined unfiltered signal shape: {portfolio.get_combined_unfiltered().shape}")
    print(f"Portfolio weights shape: {portfolio_weights.shape}")
    print(f"Predictor contributions: {portfolio.get_predictor_contribution()}")
    
    # ==========================================================================
    # 6. BACKTESTING
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Step 6: Backtesting")
    print("=" * 60)
    
    backtester = Backtester(
        rebalance_frequency=24,  # Daily rebalancing (24 hours)
        transaction_cost_bps=10.0,
        slippage_bps=5.0
    )
    
    result = backtester.run(portfolio_weights, price_matrix)
    
    print(f"Daily returns: {len(result.daily_returns)} days")
    print(f"Final cumulative return: {result.cumulative_returns.iloc[-1]:.2%}")
    print(f"Average turnover: {result.turnover.mean():.4f}")
    
    # ==========================================================================
    # 7. METRICS
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Step 7: Performance Metrics")
    print("=" * 60)
    
    metrics = result.metrics
    print(f"  Annual Return:     {metrics.annual_return:>10.2%}")
    print(f"  Annual Volatility: {metrics.annual_volatility:>10.2%}")
    print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:>10.2f}")
    print(f"  Max Drawdown:      {metrics.max_drawdown:>10.2%}")
    print(f"  Calmar Ratio:      {metrics.calmar_ratio:>10.2f}")
    print(f"  Win Rate:          {metrics.win_rate:>10.2%}")
    print(f"  Profit Factor:     {metrics.profit_factor:>10.2f}")
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    main()
