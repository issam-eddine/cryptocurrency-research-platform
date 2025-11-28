"""
Portfolio - Combines multiple predictors with configurable weights.

Workflow (correct implementation):
1. Collect RAW signals from each predictor (before any processing)
2. Combine: raw_combined = sum(allocation * raw_signal)
3. Z-score the combined raw signal
4. Filter by quantiles
5. Re-zscore on active names
6. weights = final signal (vol targeting optional, disabled by default)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class Portfolio:
    """
    Portfolio that combines multiple predictors (strategies).
    
    The portfolio:
    1. Collects RAW signals from each predictor
    2. Combines them using weighted average
    3. Z-scores the combined signal
    4. Filters by quantiles
    5. Re-z-scores on active names
    6. Returns final weights (= final signal for now)
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
        
        self._raw_signals: Dict[str, pd.DataFrame] = {}
        self._combined_raw: Optional[pd.DataFrame] = None
        self._combined_signal: Optional[pd.DataFrame] = None
        self._weights: Optional[pd.DataFrame] = None
    
    def add_predictor_raw_signal(self, name: str, raw_signal: pd.DataFrame):
        """
        Add RAW signal from a predictor (before z-scoring/filtering).
        
        Args:
            name: Predictor name
            raw_signal: Raw signal DataFrame (from Predictor.compute_raw_signal())
        """
        self._raw_signals[name] = raw_signal
    
    def combine_and_process(self) -> pd.DataFrame:
        """
        Combine raw signals and process to final weights.
        
        Steps:
        1. Weighted combination of raw signals
        2. Z-score on full universe
        3. Filter by quantiles
        4. Re-zscore on active names
        5. Apply volatility targeting (if enabled)
        
        Returns:
            Final weights DataFrame
        """
        if not self._raw_signals:
            raise ValueError("No raw signals added. Call add_predictor_raw_signal first.")
        
        # Step 1: Combine raw signals with weights
        self._combined_raw = self._combine_raw_signals()
        
        # Step 2: Z-score on full universe
        zscored = self._zscore(self._combined_raw)
        
        # Step 3: Filter by quantiles
        filtered = self._filter(zscored)
        
        # Step 4: Re-zscore on active names
        self._combined_signal = self._rezscore_active(filtered)
        
        # Step 5: Apply volatility targeting (if enabled)
        self._weights = self._apply_vol_target(self._combined_signal)
        
        return self._weights
    
    def _combine_raw_signals(self) -> pd.DataFrame:
        """
        Weighted combination of raw signals.
        
        Returns:
            Combined raw signal DataFrame
        """
        names = list(self._raw_signals.keys())
        
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
        for name, raw_signal in self._raw_signals.items():
            weighted = raw_signal * weights[name]
            if combined is None:
                combined = weighted
            else:
                combined = combined.add(weighted, fill_value=0)
        
        return combined
    
    def _zscore(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectional z-score.
        
        Args:
            signals: Signal DataFrame
            
        Returns:
            Z-scored signals DataFrame
        """
        cs_mean = signals.mean(axis=1, skipna=True)
        cs_std = signals.std(axis=1, skipna=True)
        cs_std = cs_std.replace(0, np.nan)
        
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
        
        # Long positions
        long_mask = zscored.ge(long_threshold, axis=0)
        filtered = filtered.where(~long_mask, 1.0 if self.discrete else zscored)
        
        # Short positions
        if self.long_short:
            short_mask = zscored.le(short_threshold, axis=0)
            filtered = filtered.where(~short_mask, -1.0 if self.discrete else zscored)
        
        return filtered
    
    def _rezscore_active(self, filtered: pd.DataFrame) -> pd.DataFrame:
        """
        Re-zscore on active (non-zero) names.
        
        Args:
            filtered: Filtered signals DataFrame
            
        Returns:
            Re-z-scored signals DataFrame
        """
        re_zscored = filtered.copy()
        
        for date in filtered.index:
            row = filtered.loc[date]
            active_mask = row != 0
            
            if active_mask.sum() > 1:
                active_values = row[active_mask]
                mean_active = active_values.mean()
                std_active = active_values.std()
                
                if std_active > 0:
                    re_zscored.loc[date, active_mask] = \
                        (active_values - mean_active) / std_active
        
        return re_zscored
    
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
    
    def get_combined_raw(self) -> Optional[pd.DataFrame]:
        """Return combined raw signal (before z-scoring)."""
        return self._combined_raw
    
    def get_combined_signal(self) -> Optional[pd.DataFrame]:
        """Return combined processed signal."""
        return self._combined_signal
    
    def get_predictor_contribution(self) -> Dict[str, float]:
        """
        Get the normalized contribution of each predictor.
        
        Returns:
            Dict mapping predictor names to their weight contribution
        """
        names = list(self._raw_signals.keys())
        
        if not names:
            return {}
        
        if not self.predictor_weights:
            return {name: 1.0 / len(names) for name in names}
        
        total = sum(self.predictor_weights.get(name, 0) for name in names)
        if total == 0:
            return {name: 1.0 / len(names) for name in names}
        
        return {name: self.predictor_weights.get(name, 0) / total for name in names}
