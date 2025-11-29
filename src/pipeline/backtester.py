"""
Backtester - Orchestrates the full backtesting pipeline.

TIMING CONVENTION:
------------------
- weights[t] = portfolio weights determined at close of day t
- returns[t] = (price[t] - price[t-1]) / price[t-1] (return from t-1 to t)
- portfolio_return[t] = weights[t-1] * returns[t]

This means:
1. At close of day t-1, we determine our weights
2. We hold these weights through day t
3. We realize returns from close t-1 to close t

The backtest correctly shifts weights by 1 day before multiplying with returns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .metrics import MetricsCalculator, PerformanceMetrics
from .target_engineer import TargetEngineer


@dataclass
class BacktestResult:
    """Container for backtest results."""
    daily_returns: pd.Series
    cumulative_returns: pd.Series
    weights: pd.DataFrame
    metrics: PerformanceMetrics
    turnover: pd.Series
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "daily_returns": self.daily_returns,
            "cumulative_returns": self.cumulative_returns,
            "weights": self.weights,
            "metrics": self.metrics.to_dict(),
            "turnover": self.turnover
        }


class Backtester:
    """
    Backtester that runs the full backtest simulation.
    
    Handles:
    - Portfolio weight application with rebalancing
    - Transaction cost modeling
    - Return calculation
    - Performance metrics computation
    """
    
    def __init__(self, 
                 rebalance_frequency: int = 24,
                 transaction_cost_bps: float = 10.0,
                 slippage_bps: float = 5.0):
        """
        Initialize the backtester.
        
        Args:
            rebalance_frequency: Periods between rebalancing (default 24 = daily for hourly data)
            transaction_cost_bps: Transaction cost in basis points
            slippage_bps: Slippage estimate in basis points
        """
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost_bps = transaction_cost_bps / 10000  # Convert to decimal
        self.slippage_bps = slippage_bps / 10000
        
        self.metrics_calculator = MetricsCalculator()
        self.target_engineer = TargetEngineer(forward_period=1)
        
        self._result: Optional[BacktestResult] = None
    
    def run(self, weights: pd.DataFrame, 
            price_matrix: pd.DataFrame) -> BacktestResult:
        """
        Run the backtest.
        
        Args:
            weights: Target portfolio weights (dates x symbols)
            price_matrix: Price matrix (dates x symbols)
            
        Returns:
            BacktestResult with returns, metrics, etc.
        """
        # Align weights and prices
        common_dates = weights.index.intersection(price_matrix.index)
        common_symbols = weights.columns.intersection(price_matrix.columns)
        
        weights = weights.loc[common_dates, common_symbols]
        prices = price_matrix.loc[common_dates, common_symbols]
        
        # Apply rebalancing schedule
        rebalanced_weights = self._apply_rebalance_schedule(weights)
        
        # Compute returns using TargetEngineer
        asset_returns = self.target_engineer.compute_returns(prices).fillna(0)
        
        # Shift weights (trade at close, realize returns next day)
        shifted_weights = rebalanced_weights.shift(1).fillna(0)
        
        # Compute turnover and transaction costs
        turnover = self._compute_turnover(shifted_weights)
        transaction_costs = turnover * (self.transaction_cost_bps + self.slippage_bps)
        
        # Compute portfolio returns
        portfolio_returns = (shifted_weights * asset_returns).sum(axis=1)
        net_returns = portfolio_returns - transaction_costs
        
        # Compute cumulative returns
        cumulative = (1 + net_returns).cumprod()
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_all(net_returns)
        
        self._result = BacktestResult(
            daily_returns=net_returns,
            cumulative_returns=cumulative,
            weights=rebalanced_weights,
            metrics=metrics,
            turnover=turnover
        )
        
        return self._result
    
    def _apply_rebalance_schedule(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rebalancing schedule to target weights.
        
        Args:
            weights: Target weights DataFrame
            
        Returns:
            Weights with rebalancing schedule applied
        """
        rebalanced = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
        
        for i, date in enumerate(weights.index):
            if i % self.rebalance_frequency == 0:
                rebalanced.loc[date] = weights.loc[date]
        
        # Forward fill between rebalance dates
        rebalanced = rebalanced.replace(0, np.nan).ffill().fillna(0)
        
        return rebalanced
    
    def _compute_turnover(self, weights: pd.DataFrame) -> pd.Series:
        """
        Compute portfolio turnover.
        
        Args:
            weights: Portfolio weights DataFrame
            
        Returns:
            Turnover series (sum of absolute weight changes / 2)
        """
        weight_changes = weights.diff().abs()
        turnover = weight_changes.sum(axis=1) / 2
        return turnover.fillna(0)
    
    def get_result(self) -> Optional[BacktestResult]:
        """Return the most recent backtest result."""
        return self._result
    
    def run_with_benchmark(self, weights: pd.DataFrame,
                           price_matrix: pd.DataFrame,
                           benchmark_symbol: str = "BTC/USDT") -> Tuple[BacktestResult, BacktestResult]:
        """
        Run backtest with benchmark comparison.
        
        Args:
            weights: Target portfolio weights
            price_matrix: Price matrix
            benchmark_symbol: Symbol to use as benchmark
            
        Returns:
            Tuple of (strategy_result, benchmark_result)
        """
        # Strategy backtest
        strategy_result = self.run(weights, price_matrix)
        
        # Benchmark backtest (100% in benchmark)
        if benchmark_symbol in price_matrix.columns:
            benchmark_weights = pd.DataFrame(0.0, 
                                            index=weights.index, 
                                            columns=weights.columns)
            benchmark_weights[benchmark_symbol] = 1.0
            
            benchmark_result = self.run(benchmark_weights, price_matrix)
        else:
            # Equal weight as fallback
            benchmark_weights = pd.DataFrame(
                1.0 / len(weights.columns),
                index=weights.index,
                columns=weights.columns
            )
            benchmark_result = self.run(benchmark_weights, price_matrix)
        
        return strategy_result, benchmark_result
    
    def compute_rolling_metrics(self, returns: pd.Series, 
                                 window: int = 168) -> pd.DataFrame:
        """
        Compute rolling performance metrics.
        
        Args:
            returns: Hourly returns series
            window: Rolling window size (default 168 = 1 week of hourly data)
            
        Returns:
            DataFrame with rolling metrics
        """
        rolling_sharpe = self.metrics_calculator.rolling_sharpe(returns, window)
        rolling_dd = self.metrics_calculator.rolling_drawdown(returns)
        rolling_vol = returns.rolling(window).std() * np.sqrt(8760)
        
        return pd.DataFrame({
            "rolling_sharpe": rolling_sharpe,
            "rolling_drawdown": rolling_dd,
            "rolling_volatility": rolling_vol
        })

