"""
Predictor - Wraps a SignalStrategy with z-scoring and filtering.
Predictor construction = cross-sectional z-score + quantile filtering.
"""

import pandas as pd
import numpy as np
from typing import Optional

from .signal_strategy import SignalStrategy


class Predictor:
    """
    Predictor that wraps a SignalStrategy and applies cross-sectional z-scoring + quantile filtering.
    
    The predictor:
    1. Computes raw signals using the strategy
    2. Z-scores signals cross-sectionally (preserves magnitude information)
    3. Filters based on quantile thresholds (top_q for longs, bottom_q for shorts)
    """
    
    def __init__(self, strategy: SignalStrategy, 
                 top_q: float = 0.8, 
                 bottom_q: float = 0.2,
                 long_short: bool = True):
        """
        Initialize the predictor.
        
        Args:
            strategy: The signal strategy to use
            top_q: Quantile threshold for long positions (e.g., 0.8 = top 20%)
            bottom_q: Quantile threshold for short positions (e.g., 0.2 = bottom 20%)
            long_short: If True, generate both long and short signals; if False, long only
        """
        self.strategy = strategy
        self.top_q = top_q
        self.bottom_q = bottom_q
        self.long_short = long_short
        
        self._raw_signals: Optional[pd.DataFrame] = None
        self._zscored_signals: Optional[pd.DataFrame] = None
        self._filtered_signals: Optional[pd.DataFrame] = None
    
    @property
    def name(self) -> str:
        """Return the predictor name (from strategy)."""
        return self.strategy.name
    
    def compute_signals(self, price_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute raw signals for the universe.
        
        Args:
            price_matrix: DataFrame with dates as index, symbols as columns
            
        Returns:
            Raw signal DataFrame
        """
        self._raw_signals = self.strategy.compute_universe(price_matrix)
        return self._raw_signals
    
    def zscore(self, signals: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute cross-sectional z-scores (preserves signal magnitude).
        
        Args:
            signals: Optional signals DataFrame (uses cached if None)
            
        Returns:
            Z-scored signals DataFrame
        """
        if signals is None:
            signals = self._raw_signals
        
        if signals is None:
            raise ValueError("No signals to z-score. Call compute_signals first.")
        
        # Cross-sectional z-score: (x - mean) / std for each row
        cs_mean = signals.mean(axis=1)
        cs_std = signals.std(axis=1)
        self._zscored_signals = signals.sub(cs_mean, axis=0).div(cs_std, axis=0)
        return self._zscored_signals

    def filter(self, zscored_signals: Optional[pd.DataFrame] = None, discrete: bool = False) -> pd.DataFrame:
        """
        Filter z-scored signals based on quantile thresholds.
        
        Returns a DataFrame with:
        - If discrete=False (default): Z-score values for assets in top/bottom quantiles
        - If discrete=True: +1/-1 for assets in top/bottom quantiles
        - 0.0 for assets in the middle
        
        Args:
            zscored_signals: Optional z-scored signals (uses cached if None)
            discrete: If True, use +1/0/-1; if False, preserve z-score magnitudes
            
        Returns:
            Filtered signals DataFrame
        """
        if zscored_signals is None:
            zscored_signals = self._zscored_signals
        
        if zscored_signals is None:
            raise ValueError("No z-scored signals. Call zscore first.")
        
        # Compute quantile thresholds for each row
        long_threshold = zscored_signals.quantile(self.top_q, axis=1)
        short_threshold = zscored_signals.quantile(self.bottom_q, axis=1)
        
        filtered = pd.DataFrame(0.0, index=zscored_signals.index, columns=zscored_signals.columns)
        
        # Long positions: above top quantile threshold
        long_mask = zscored_signals.ge(long_threshold, axis=0)
        if discrete:
            filtered = filtered.where(~long_mask, 1.0)
        else:
            filtered = filtered.where(~long_mask, zscored_signals)
        
        # Short positions: below bottom quantile threshold (if enabled)
        if self.long_short:
            short_mask = zscored_signals.le(short_threshold, axis=0)
            if discrete:
                filtered = filtered.where(~short_mask, -1.0)
            else:
                filtered = filtered.where(~short_mask, zscored_signals)
        
        self._filtered_signals = filtered
        return filtered

    def predict(self, price_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Full prediction pipeline: compute signals -> z-score -> filter.
        
        Args:
            price_matrix: DataFrame with dates as index, symbols as columns
            
        Returns:
            Filtered signals ready for portfolio construction
        """
        self.compute_signals(price_matrix)
        self.zscore()
        return self.filter()
    
    def get_zscored_signals(self) -> Optional[pd.DataFrame]:
        """Return the most recent z-scored signals."""
        return self._zscored_signals
    
    def get_raw_signals(self) -> Optional[pd.DataFrame]:
        """Return the most recent raw signals."""
        return self._raw_signals
