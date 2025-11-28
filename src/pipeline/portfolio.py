"""
Portfolio - Combines multiple predictors with configurable weights.
1 portfolio = combine the 3 predictors.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


class Portfolio:
    """
    Portfolio that combines multiple predictors with configurable weights.
    
    The portfolio:
    1. Collects signals from multiple predictors
    2. Combines them using weighted average
    3. Generates final portfolio weights
    """
    
    def __init__(self, predictor_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the portfolio.
        
        Args:
            predictor_weights: Dict mapping predictor names to weights.
                              If None, equal weights are used.
        """
        self.predictor_weights = predictor_weights or {}
        self._predictor_signals: Dict[str, pd.DataFrame] = {}
        self._combined_signal: Optional[pd.DataFrame] = None
        self._portfolio_weights: Optional[pd.DataFrame] = None
    
    def add_predictor_signal(self, name: str, signals: pd.DataFrame):
        """
        Add signals from a predictor.
        
        Args:
            name: Predictor name
            signals: Signals DataFrame (output from Predictor.predict())
        """
        self._predictor_signals[name] = signals
    
    def combine_signals(self) -> pd.DataFrame:
        """
        Combine signals from all predictors using weighted average.
        
        If predictor_weights is empty/None, equal weights are used.
        
        Returns:
            Combined signal DataFrame
        """
        if not self._predictor_signals:
            raise ValueError("No predictor signals added. Call add_predictor_signal first.")
        
        # Determine weights
        names = list(self._predictor_signals.keys())
        
        if not self.predictor_weights:
            # Equal weights
            weights = {name: 1.0 / len(names) for name in names}
        else:
            # Normalize provided weights
            total = sum(self.predictor_weights.get(name, 0) for name in names)
            if total == 0:
                weights = {name: 1.0 / len(names) for name in names}
            else:
                weights = {name: self.predictor_weights.get(name, 0) / total for name in names}
        
        # Combine signals
        combined = None
        for name, signals in self._predictor_signals.items():
            weighted = signals * weights[name]
            if combined is None:
                combined = weighted
            else:
                combined = combined.add(weighted, fill_value=0)
        
        # Apply cross-sectional z-score normalization
        cs_mean = combined.mean(axis=1)
        cs_std = combined.std(axis=1)
        combined = combined.sub(cs_mean, axis=0).div(cs_std.replace(0, np.nan), axis=0).fillna(0)
        
        self._combined_signal = combined
        return combined
    
    def generate_weights(self, long_short: bool = True) -> pd.DataFrame:
        """
        Generate portfolio weights from combined signals.
        
        Weights are the cross-sectional z-scores directly.
        
        Args:
            long_short: If True, include short positions (negative z-scores)
            
        Returns:
            Portfolio weights DataFrame
        """
        if self._combined_signal is None:
            raise ValueError("No combined signal. Call combine_signals first.")
        
        # Weights are the z-scores directly
        weights = self._combined_signal.copy()
        
        # If long only, zero out negative weights
        if not long_short:
            weights = weights.clip(lower=0)
        
        self._portfolio_weights = weights
        return weights
    
    def get_weights(self) -> Optional[pd.DataFrame]:
        """Return the most recent portfolio weights."""
        return self._portfolio_weights
    
    def get_combined_signal(self) -> Optional[pd.DataFrame]:
        """Return the most recent combined signal."""
        return self._combined_signal
    
    def set_weights(self, weights: Dict[str, float]):
        """
        Update predictor weights.
        
        Args:
            weights: Dict mapping predictor names to weights
        """
        self.predictor_weights = weights
    
    def get_predictor_contribution(self) -> Dict[str, float]:
        """
        Get the normalized contribution of each predictor.
        
        Returns:
            Dict mapping predictor names to their weight contribution
        """
        names = list(self._predictor_signals.keys())
        
        if not self.predictor_weights:
            return {name: 1.0 / len(names) for name in names}
        
        total = sum(self.predictor_weights.get(name, 0) for name in names)
        if total == 0:
            return {name: 1.0 / len(names) for name in names}
        
        return {name: self.predictor_weights.get(name, 0) / total for name in names}
