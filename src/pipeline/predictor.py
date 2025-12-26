"""
Predictor - Wraps a SignalStrategy with z-scoring and filtering.

Workflow:
1. compute_raw_signal(): Just the transformation (momentum, mean_rev, etc)
2. compute_unfiltered_signal(): raw signal → cross-sectional z-score
3. For multi-strategy: Portfolio combines UNFILTERED signals (z-scored, not filtered)
4. For single strategy: process_signal() does z-score → filter → re-zscore
"""

import pandas as pd
from typing import Optional

from .signal_strategy import SignalStrategy
from .signal_utils import zscore_cross_sectional, filter_by_quantile, rezscore_active


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
        self._unfiltered_signal: Optional[pd.DataFrame] = None
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
    
    def compute_unfiltered_signal(self, price_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute unfiltered signal (raw signal → cross-sectional z-score).
        
        This is the signal that should be combined across predictors in a portfolio,
        as it puts different strategies on a comparable scale before combining.
        
        Args:
            price_matrix: DataFrame with dates as index, symbols as columns
            
        Returns:
            Unfiltered signal DataFrame (z-scored but not filtered)
        """
        # Compute raw signal first
        raw_signal = self.compute_raw_signal(price_matrix)
        
        # Apply cross-sectional z-score
        self._unfiltered_signal = zscore_cross_sectional(raw_signal)
        return self._unfiltered_signal
    
    def get_unfiltered_signal(self) -> Optional[pd.DataFrame]:
        """Return the most recent unfiltered signal (z-scored but not filtered)."""
        return self._unfiltered_signal
    
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
        zscored = zscore_cross_sectional(raw_signal)
        
        # Step 2: Filter by quantiles
        filtered = filter_by_quantile(
            zscored, 
            top_q=self.top_q, 
            bottom_q=self.bottom_q,
            long_short=self.long_short, 
            discrete=self.discrete
        )
        
        # Step 3: Re-zscore on active names only (vectorized)
        re_zscored = rezscore_active(filtered)
        
        self._processed_signal = re_zscored
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
