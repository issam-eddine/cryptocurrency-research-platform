"""
Metrics Calculator - Computes performance metrics for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_periods: int
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "total_periods": self.total_periods
        }


class MetricsCalculator:
    """
    Calculator for portfolio performance metrics.
    
    Computes:
    - Annualized return and volatility
    - Sharpe ratio
    - Maximum drawdown
    - Calmar ratio
    - Win rate and profit factor
    """
    
    def __init__(self, periods_per_year: int = 8760, risk_free_rate: float = 0.0):
        """
        Initialize the metrics calculator.
        
        Args:
            periods_per_year: Trading periods per year (8760 = 365*24 for hourly crypto)
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

    def annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return using geometric mean."""
        if returns.empty:
            return 0.0
        
        # Compound all returns using geometric mean
        total_return = (1.0 + returns).prod() - 1.0
        
        # Calculate number of years
        n_periods = len(returns)
        n_years = n_periods / self.periods_per_year

        # Annualize
        return (1.0 + total_return) ** (1.0 / n_years) - 1.0

    def annualized_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if returns.empty:
            return 0.0
        return returns.std() * np.sqrt(self.periods_per_year)

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        ann_ret = self.annualized_return(returns)
        ann_vol = self.annualized_volatility(returns)
        
        if ann_vol == 0:
            return 0.0
        
        return (ann_ret - self.risk_free_rate) / ann_vol

    def max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if returns.empty:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        
        return drawdown.min()

    def calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        ann_ret = self.annualized_return(returns)
        mdd = abs(self.max_drawdown(returns))
        
        if mdd == 0:
            return 0.0
        
        return ann_ret / mdd

    def win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate (% of positive returns)."""
        if returns.empty:
            return 0.0
        
        positive = (returns > 0).sum()
        total = len(returns)
        
        return positive / total if total > 0 else 0.0

    def profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if returns.empty:
            return 0.0
        
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return float('inf') if gains > 0 else 0.0
        
        return gains / losses


    def compute_all(self, returns: pd.Series) -> PerformanceMetrics:
        """
        Compute all performance metrics.
        
        Args:
            returns: Series of period returns
            
        Returns:
            PerformanceMetrics dataclass with all metrics
        """
        returns = returns.dropna()
        
        if returns.empty:
            return PerformanceMetrics(
                annual_return=0.0,
                annual_volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                total_periods=0
            )
        
        # Compute average win/loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0
        
        return PerformanceMetrics(
            annual_return=self.annualized_return(returns),
            annual_volatility=self.annualized_volatility(returns),
            sharpe_ratio=self.sharpe_ratio(returns),
            max_drawdown=self.max_drawdown(returns),
            calmar_ratio=self.calmar_ratio(returns),
            win_rate=self.win_rate(returns),
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=self.profit_factor(returns),
            total_periods=len(returns)
        )
    
    def compute_dict(self, returns: pd.Series) -> Dict[str, float]:
        """
        Compute all metrics and return as dictionary.
        
        Args:
            returns: Series of period returns
            
        Returns:
            Dictionary of metrics
        """
        return self.compute_all(returns).to_dict()
    
    def cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """Compute cumulative returns."""
        return (1 + returns).cumprod()
    
    def rolling_sharpe(self, returns: pd.Series, window: int = 168) -> pd.Series:
        """
        Compute rolling Sharpe ratio.
        
        Args:
            returns: Series of period returns
            window: Rolling window size (default 168 = 1 week of hourly data)
            
        Returns:
            Rolling Sharpe ratio series
        """
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        
        # Annualize
        ann_factor = np.sqrt(self.periods_per_year)
        
        return (rolling_mean * self.periods_per_year - self.risk_free_rate) / (rolling_std * ann_factor)
    
    def rolling_drawdown(self, returns: pd.Series) -> pd.Series:
        """Compute rolling drawdown series."""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        return (cumulative - peak) / peak

