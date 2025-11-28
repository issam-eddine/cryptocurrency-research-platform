"""
Predictor - Wraps a SignalStrategy with z-scoring and filtering.

Workflow:
1. compute_raw_signal(): Just the transformation (momentum, mean_rev, etc)
2. For multi-strategy: Portfolio combines raw signals
3. For single strategy: process_signal() does z-score → filter → re-zscore
"""

import pandas as pd
import numpy as np
from typing import Optional

from .signal_strategy import SignalStrategy


class Predictor:
    """
    Predictor that wraps a SignalStrategy and applies cross-sectional 
    z-scoring + quantile filtering + re-z-scoring.
    
    The predictor:
    1. Computes raw signals using the strategy
    2. Z-scores signals cross-sectionally (full universe)
    3. Filters based on quantile thresholds (top_q for longs, bottom_q for shorts)
    4. Re-z-scores on active (non-zero) names only
    """
    
    def __init__(self, strategy: SignalStrategy, 
                 top_q: float = 0.8, 
                 bottom_q: float = 0.2,
                 long_short: bool = True,
                 discrete: bool = False):
        """
        Initialize the predictor.
        
        Args:
            strategy: The signal strategy to use
            top_q: Quantile threshold for long positions (e.g., 0.8 = top 20%)
            bottom_q: Quantile threshold for short positions (e.g., 0.2 = bottom 20%)
            long_short: If True, generate both long and short signals; if False, long only
            discrete: If True, use +1/0/-1; if False, preserve z-score magnitudes
        """
        self.strategy = strategy
        self.top_q = top_q
        self.bottom_q = bottom_q
        self.long_short = long_short
        self.discrete = discrete
        
        self._raw_signal: Optional[pd.DataFrame] = None
        self._processed_signal: Optional[pd.DataFrame] = None
    
    @property
    def name(self) -> str:
        """Return the predictor name (from strategy)."""
        return self.strategy.name
    
    def compute_raw_signal(self, price_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute raw signal (transformation only, no z-scoring).
        
        Args:
            price_matrix: DataFrame with dates as index, symbols as columns
            
        Returns:
            Raw signal DataFrame
        """
        self._raw_signal = self.strategy.compute_universe(price_matrix)
        return self._raw_signal
    
    def process_signal(self, raw_signal: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Process: z-score → filter → re-zscore.
        
        Steps:
        1. Z-score on full universe
        2. Filter by quantiles
        3. Re-zscore on active (non-zero) names only
        
        Args:
            raw_signal: Optional raw signal DataFrame (uses cached if None)
            
        Returns:
            Processed signal DataFrame
        """
        if raw_signal is None:
            raw_signal = self._raw_signal
        
        if raw_signal is None:
            raise ValueError("No raw signal. Call compute_raw_signal first.")
        
        # Step 1: Z-score on full universe
        zscored = self._zscore(raw_signal)
        
        # Step 2: Filter by quantiles
        filtered = self._filter(zscored)
        
        # Step 3: Re-zscore on active names only
        re_zscored = self._rezscore_active(filtered)
        
        self._processed_signal = re_zscored
        return re_zscored
    
    def _zscore(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectional z-score (full universe).
        
        Args:
            signals: Signal DataFrame
            
        Returns:
            Z-scored signals DataFrame
        """
        cs_mean = signals.mean(axis=1, skipna=True)
        cs_std = signals.std(axis=1, skipna=True)
        cs_std = cs_std.replace(0, np.nan)  # Avoid division by zero
        
        zscored = signals.sub(cs_mean, axis=0).div(cs_std, axis=0)
        return zscored.fillna(0)
    
    def _filter(self, zscored: pd.DataFrame) -> pd.DataFrame:
        """
        Filter by top/bottom quantiles.
        
        Args:
            zscored: Z-scored signals DataFrame
            
        Returns:
            Filtered signals DataFrame
        """
        long_threshold = zscored.quantile(self.top_q, axis=1)
        short_threshold = zscored.quantile(self.bottom_q, axis=1)
        
        filtered = pd.DataFrame(0.0, index=zscored.index, columns=zscored.columns)
        
        # Long: above top quantile
        long_mask = zscored.ge(long_threshold, axis=0)
        filtered = filtered.where(~long_mask, 1.0 if self.discrete else zscored)
        
        # Short: below bottom quantile (if enabled)
        if self.long_short:
            short_mask = zscored.le(short_threshold, axis=0)
            filtered = filtered.where(~short_mask, -1.0 if self.discrete else zscored)
        
        return filtered
    
    def _rezscore_active(self, filtered: pd.DataFrame) -> pd.DataFrame:
        """
        Re-zscore on active (non-zero) names only.
        
        Args:
            filtered: Filtered signals DataFrame
            
        Returns:
            Re-z-scored signals DataFrame
        """
        re_zscored = filtered.copy()
        
        for date in filtered.index:
            row = filtered.loc[date]
            active_mask = row != 0
            
            if active_mask.sum() > 1:  # Need at least 2 for std
                active_values = row[active_mask]
                mean_active = active_values.mean()
                std_active = active_values.std()
                
                if std_active > 0:
                    re_zscored.loc[date, active_mask] = \
                        (active_values - mean_active) / std_active
        
        return re_zscored
    
    def predict(self, price_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Full prediction pipeline: raw signal → z-score → filter → re-zscore.
        
        Args:
            price_matrix: DataFrame with dates as index, symbols as columns
            
        Returns:
            Processed signals ready for portfolio construction
        """
        self.compute_raw_signal(price_matrix)
        return self.process_signal()
    
    def get_raw_signal(self) -> Optional[pd.DataFrame]:
        """Return the most recent raw signal."""
        return self._raw_signal
    
    def get_processed_signal(self) -> Optional[pd.DataFrame]:
        """Return the most recent processed signal."""
        return self._processed_signal
