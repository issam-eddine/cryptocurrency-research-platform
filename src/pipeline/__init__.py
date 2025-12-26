"""
Pipeline module for cryptocurrency research platform.

Clean class-based pipeline architecture:
- DataPipeline: Data extraction and preprocessing
- SignalStrategy: Base class for signal strategies (Momentum, MeanReversion, EWMACrossover)
- Predictor: Ranking and filtering of signals
- Portfolio: Combining multiple predictors with weights
- FeatureEngineer: Feature computation for all strategies
- TargetEngineer: Target variable computation (1-day returns)
- MetricsCalculator: Performance metrics
- Backtester: Full backtest orchestration
"""

from .data_pipeline import DataPipeline
from .signal_strategy import (
    SignalStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    EWMACrossoverStrategy
)
from .predictor import Predictor
from .portfolio import Portfolio
from .feature_engineer import FeatureEngineer
from .target_engineer import TargetEngineer
from .metrics import MetricsCalculator, PerformanceMetrics
from .backtester import Backtester, BacktestResult
from .signal_utils import zscore_cross_sectional, filter_by_quantile, rezscore_active


__all__ = [
    # Data
    "DataPipeline",
    
    # Strategies
    "SignalStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "EWMACrossoverStrategy",
    
    # Predictor & Portfolio
    "Predictor",
    "Portfolio",
    
    # Engineering
    "FeatureEngineer",
    "TargetEngineer",
    
    # Metrics & Backtest
    "MetricsCalculator",
    "PerformanceMetrics",
    "Backtester",
    "BacktestResult",
    
    # Signal Utilities
    "zscore_cross_sectional",
    "filter_by_quantile",
    "rezscore_active",
]

