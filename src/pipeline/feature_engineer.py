"""
Feature Engineering - Computes features for all 3 strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from .signal_strategy import (
    SignalStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    EWMACrossoverStrategy
)


class FeatureEngineer:
    """
    Feature engineer that computes features for all signal strategies.
    
    Computes features for:
    1. Momentum strategy
    2. Mean reversion strategy
    3. EWMA crossover strategy
    """
    
    def __init__(self, 
                 momentum_lookback: int = 21,
                 mean_reversion_lookback: int = 21,
                 ewma_fast: int = 12,
                 ewma_slow: int = 26,
                 ewma_std: int = 20):
        """
        Initialize the feature engineer with strategy parameters.
        
        Args:
            momentum_lookback: Lookback period for momentum
            mean_reversion_lookback: Lookback period for mean reversion z-score
            ewma_fast: Fast EWMA span
            ewma_slow: Slow EWMA span
            ewma_std: Rolling std window for EWMA normalization
        """
        self.strategies: Dict[str, SignalStrategy] = {
            "momentum": MomentumStrategy(lookback=momentum_lookback),
            "mean_reversion": MeanReversionStrategy(lookback=mean_reversion_lookback),
            "ewma_crossover": EWMACrossoverStrategy(
                fast_window=ewma_fast,
                slow_window=ewma_slow,
                std_window=ewma_std
            )
        }
        
        self._features: Dict[str, pd.DataFrame] = {}
    
    def compute_features(self, price_matrix: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Compute features for all strategies.
        
        Args:
            price_matrix: DataFrame with dates as index, symbols as columns
            
        Returns:
            Dict mapping strategy names to feature DataFrames
        """
        for name, strategy in self.strategies.items():
            self._features[name] = strategy.compute_universe(price_matrix)
        
        return self._features
    
    def compute_single_feature(self, name: str, 
                               price_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for a single strategy.
        
        Args:
            name: Strategy name ("momentum", "mean_reversion", or "ewma_crossover")
            price_matrix: DataFrame with dates as index, symbols as columns
            
        Returns:
            Feature DataFrame for the specified strategy
        """
        if name not in self.strategies:
            raise ValueError(f"Unknown strategy: {name}. "
                           f"Available: {list(self.strategies.keys())}")
        
        feature = self.strategies[name].compute_universe(price_matrix)
        self._features[name] = feature
        return feature
    
    def get_features(self) -> Dict[str, pd.DataFrame]:
        """Return all computed features."""
        return self._features.copy()
    
    def get_feature(self, name: str) -> Optional[pd.DataFrame]:
        """Return features for a specific strategy."""
        return self._features.get(name)
    
    def get_strategy(self, name: str) -> Optional[SignalStrategy]:
        """Return a specific strategy instance."""
        return self.strategies.get(name)
    
    def get_strategy_names(self) -> List[str]:
        """Return list of available strategy names."""
        return list(self.strategies.keys())
    
    def add_strategy(self, name: str, strategy: SignalStrategy):
        """
        Add a custom strategy.
        
        Args:
            name: Name for the strategy
            strategy: SignalStrategy instance
        """
        self.strategies[name] = strategy
    
    def get_combined_features(self, price_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features and combine into a single multi-level DataFrame.
        
        Args:
            price_matrix: DataFrame with dates as index, symbols as columns
            
        Returns:
            Multi-level DataFrame with (strategy, symbol) columns
        """
        features = self.compute_features(price_matrix)
        
        combined = pd.concat(
            features.values(),
            axis=1,
            keys=features.keys()
        )
        
        return combined
