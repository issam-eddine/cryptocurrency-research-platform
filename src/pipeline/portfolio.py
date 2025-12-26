"""
Portfolio - Combines multiple predictors with configurable weights.

Workflow (correct implementation):
1. Collect UNFILTERED signals from each predictor (raw signal → z-score)
2. Combine: unfiltered_combined = sum(allocation * unfiltered_signal)
3. Filter by quantiles
4. Re-zscore on active names
5. weights = final signal (vol targeting optional, disabled by default)

Note: Combining at the unfiltered signal level (z-scored) ensures signals from
different strategies are on a comparable scale before combining.
"""

import pandas as pd
from typing import Dict, Optional

from .signal_utils import filter_by_quantile, rezscore_active


class Portfolio:
    """
    Portfolio that combines multiple predictors (strategies).
    
    The portfolio:
    1. Collects UNFILTERED signals from each predictor (z-scored but not filtered)
    2. Combines them using weighted average
    3. Filters by quantiles
    4. Re-z-scores on active names
    5. Returns final weights (= final signal for now)
    
    Note: Combining at the unfiltered signal level ensures signals are on a
    comparable scale before combining, since raw signals from different
    strategies have different distributions.
    """
    
    def __init__(self, 
                 predictor_weights: Optional[Dict[str, float]] = None,
                 top_q: float = 0.8,
                 bottom_q: float = 0.2,
                 long_short: bool = True,
                 discrete: bool = False,
                 enable_vol_target: bool = False,
                 vol_target: float = 0.20):
        """
        Initialize the portfolio.
        
        Args:
            predictor_weights: Dict mapping predictor names to allocation weights.
                              If None, equal weights are used.
            top_q: Quantile for long positions (0.8 = top 20%)
            bottom_q: Quantile for short positions (0.2 = bottom 20%)
            long_short: Enable short positions
            discrete: Use +1/0/-1 instead of continuous values
            enable_vol_target: Enable volatility targeting (off by default)
            vol_target: Target annualized volatility (20%)
        """
        self.predictor_weights = predictor_weights or {}
        self.top_q = top_q
        self.bottom_q = bottom_q
        self.long_short = long_short
        self.discrete = discrete
        self.enable_vol_target = enable_vol_target
        self.vol_target = vol_target
        
        self._unfiltered_signals: Dict[str, pd.DataFrame] = {}
        self._combined_unfiltered: Optional[pd.DataFrame] = None
        self._combined_signal: Optional[pd.DataFrame] = None
        self._weights: Optional[pd.DataFrame] = None
    
    def add_predictor_unfiltered_signal(self, name: str, unfiltered_signal: pd.DataFrame):
        """
        Add UNFILTERED signal from a predictor (z-scored but not filtered).
        
        Args:
            name: Predictor name
            unfiltered_signal: Unfiltered signal DataFrame (from Predictor.compute_unfiltered_signal())
        """
        self._unfiltered_signals[name] = unfiltered_signal
    
    def combine_and_process(self) -> pd.DataFrame:
        """
        Combine unfiltered signals and process to final weights.
        
        Steps:
        1. Weighted combination of unfiltered signals (already z-scored)
        2. Filter by quantiles
        3. Re-zscore on active names
        4. Apply volatility targeting (if enabled)
        
        Returns:
            Final weights DataFrame
        """
        if not self._unfiltered_signals:
            raise ValueError("No unfiltered signals added. Call add_predictor_unfiltered_signal first.")
        
        # Step 1: Combine unfiltered signals with weights
        self._combined_unfiltered = self._combine_unfiltered_signals()
        
        # Step 2: Filter by quantiles (using shared utility)
        filtered = filter_by_quantile(
            self._combined_unfiltered,
            top_q=self.top_q,
            bottom_q=self.bottom_q,
            long_short=self.long_short,
            discrete=self.discrete
        )
        
        # Step 3: Re-zscore on active names (using shared vectorized utility)
        self._combined_signal = rezscore_active(filtered)
        
        # Step 4: Apply volatility targeting (if enabled)
        self._weights = self._apply_vol_target(self._combined_signal)
        
        return self._weights
    
    def _combine_unfiltered_signals(self) -> pd.DataFrame:
        """
        Weighted combination of unfiltered signals (already z-scored).
        
        Returns:
            Combined unfiltered signal DataFrame
        """
        names = list(self._unfiltered_signals.keys())
        
        # Determine weights
        if not self.predictor_weights:
            weights = {name: 1.0 / len(names) for name in names}
        else:
            total = sum(self.predictor_weights.get(name, 0) for name in names)
            if total == 0:
                weights = {name: 1.0 / len(names) for name in names}
            else:
                weights = {name: self.predictor_weights.get(name, 0) / total 
                          for name in names}
        
        # Combine
        combined = None
        for name, unfiltered_signal in self._unfiltered_signals.items():
            weighted = unfiltered_signal * weights[name]
            if combined is None:
                combined = weighted
            else:
                combined = combined.add(weighted, fill_value=0)
        
        return combined
    
    def _apply_vol_target(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply volatility targeting (turned off by default).
        
        When enabled:
        - Compute realized portfolio volatility
        - Scale weights to achieve target volatility
        
        Args:
            signals: Signal DataFrame
            
        Returns:
            Scaled weights DataFrame
        """
        if not self.enable_vol_target:
            return signals  # weights = signals
        
        # TODO: Implement volatility targeting
        # port_returns = (signals.shift(1) * returns).sum(axis=1)
        # port_vol = port_returns.std() * np.sqrt(252)
        # scale = self.vol_target / port_vol if port_vol > 0 else 1.0
        # return signals * scale
        
        return signals
    
    def get_weights(self) -> Optional[pd.DataFrame]:
        """Return the most recent portfolio weights."""
        return self._weights
    
    def get_combined_unfiltered(self) -> Optional[pd.DataFrame]:
        """Return combined unfiltered signal (before filtering)."""
        return self._combined_unfiltered
    
    def get_combined_signal(self) -> Optional[pd.DataFrame]:
        """Return combined processed signal."""
        return self._combined_signal
    
    def get_predictor_contribution(self) -> Dict[str, float]:
        """
        Get the normalized contribution of each predictor.
        
        Returns:
            Dict mapping predictor names to their weight contribution
        """
        names = list(self._unfiltered_signals.keys())
        
        if not names:
            return {}
        
        if not self.predictor_weights:
            return {name: 1.0 / len(names) for name in names}
        
        total = sum(self.predictor_weights.get(name, 0) for name in names)
        if total == 0:
            return {name: 1.0 / len(names) for name in names}
        
        return {name: self.predictor_weights.get(name, 0) / total for name in names}
