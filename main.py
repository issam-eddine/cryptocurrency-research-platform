"""
Cryptocurrency Research Platform - Main Entry Point

Example usage of the pipeline architecture:
1. DataPipeline: Fetch and preprocess data
2. FeatureEngineer: Compute features for 3 strategies
3. Predictor (x3): Rank and filter signals
4. Portfolio: Combine predictors with weights
5. TargetEngineer: Compute 1-day returns
6. Backtester: Run backtest
7. MetricsCalculator: Compute performance metrics
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
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years of data
    
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
    # 2. FEATURE ENGINEERING (3 strategies)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Feature Engineering")
    print("=" * 60)
    
    feature_engineer = FeatureEngineer(
        momentum_lookback=21,
        mean_reversion_lookback=21,
        ewma_fast=12,
        ewma_slow=26,
        ewma_std=20
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
    # 4. PREDICTOR CONSTRUCTION (1 predictor = 1 signal strategy)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Predictor Construction (Ranking + Filtering)")
    print("=" * 60)
    
    # Create 3 predictors, one for each strategy
    momentum_predictor = Predictor(
        strategy=MomentumStrategy(lookback=21),
        top_q=0.8,
        bottom_q=0.2,
        long_short=True
    )
    
    mean_reversion_predictor = Predictor(
        strategy=MeanReversionStrategy(lookback=21),
        top_q=0.8,
        bottom_q=0.2,
        long_short=True
    )
    
    ewma_predictor = Predictor(
        strategy=EWMACrossoverStrategy(fast_window=12, slow_window=26, std_window=20),
        top_q=0.8,
        bottom_q=0.2,
        long_short=True
    )
    
    # Generate predictions (signals)
    momentum_signals = momentum_predictor.predict(price_matrix)
    mean_reversion_signals = mean_reversion_predictor.predict(price_matrix)
    ewma_signals = ewma_predictor.predict(price_matrix)
    
    print(f"  - Momentum signals: {momentum_signals.shape}")
    print(f"  - Mean Reversion signals: {mean_reversion_signals.shape}")
    print(f"  - EWMA signals: {ewma_signals.shape}")
    
    # ==========================================================================
    # 5. PORTFOLIO CONSTRUCTION (combine 3 predictors)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Step 5: Portfolio Construction")
    print("=" * 60)
    
    # Create portfolio with custom weights
    portfolio = Portfolio(predictor_weights={
        "momentum": 0.4,
        "mean_reversion": 0.3,
        "ewma_crossover": 0.3
    })
    
    # Add predictor signals
    portfolio.add_predictor_signal("momentum", momentum_signals)
    portfolio.add_predictor_signal("mean_reversion", mean_reversion_signals)
    portfolio.add_predictor_signal("ewma_crossover", ewma_signals)
    
    # Combine signals and generate weights
    combined_signal = portfolio.combine_signals()
    portfolio_weights = portfolio.generate_weights(long_short=True)
    
    print(f"Combined signal shape: {combined_signal.shape}")
    print(f"Portfolio weights shape: {portfolio_weights.shape}")
    print(f"Predictor contributions: {portfolio.get_predictor_contribution()}")
    
    # ==========================================================================
    # 6. BACKTESTING
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Step 6: Backtesting")
    print("=" * 60)
    
    backtester = Backtester(
        rebalance_frequency=21,  # Monthly rebalancing
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
