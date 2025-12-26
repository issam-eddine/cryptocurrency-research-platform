"""
Signal Utilities - Shared functions for signal processing.

Provides vectorized implementations of:
- Cross-sectional z-scoring
- Quantile filtering
- Re-z-scoring on active positions
"""

import pandas as pd
import numpy as np


def zscore_cross_sectional(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score (across assets for each time period).
    
    Args:
        signals: Signal DataFrame (dates x symbols)
        
    Returns:
        Z-scored signals DataFrame
    """
    cs_mean = signals.mean(axis=1, skipna=True)
    cs_std = signals.std(axis=1, skipna=True)
    cs_std = cs_std.replace(0, np.nan)  # Avoid division by zero
    
    zscored = signals.sub(cs_mean, axis=0).div(cs_std, axis=0)
    return zscored.fillna(0)


def filter_by_quantile(zscored: pd.DataFrame, 
                       top_q: float = 0.8, 
                       bottom_q: float = 0.2,
                       long_short: bool = True,
                       discrete: bool = False) -> pd.DataFrame:
    """
    Filter signals by top/bottom quantiles.
    
    Args:
        zscored: Z-scored signals DataFrame
        top_q: Quantile threshold for long positions (e.g., 0.8 = top 20%)
        bottom_q: Quantile threshold for short positions (e.g., 0.2 = bottom 20%)
        long_short: If True, include short positions
        discrete: If True, use +1/0/-1 instead of continuous values
        
    Returns:
        Filtered signals DataFrame
    """
    long_threshold = zscored.quantile(top_q, axis=1)
    short_threshold = zscored.quantile(bottom_q, axis=1)
    
    filtered = pd.DataFrame(0.0, index=zscored.index, columns=zscored.columns)
    
    # Long positions: above top quantile
    long_mask = zscored.ge(long_threshold, axis=0)
    filtered = filtered.where(~long_mask, 1.0 if discrete else zscored)
    
    # Short positions: below bottom quantile
    if long_short:
        short_mask = zscored.le(short_threshold, axis=0)
        filtered = filtered.where(~short_mask, -1.0 if discrete else zscored)
    
    return filtered


def rezscore_active(filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Re-z-score on active (non-zero) positions only.
    
    Vectorized implementation using masked operations.
    
    Args:
        filtered: Filtered signals DataFrame
        
    Returns:
        Re-z-scored signals DataFrame
    """
    # Create mask for active (non-zero) positions
    active_mask = filtered != 0
    
    # Replace zeros with NaN for calculation (so they're excluded from mean/std)
    masked = filtered.where(active_mask)
    
    # Compute mean and std only on active values (row-wise)
    active_mean = masked.mean(axis=1, skipna=True)
    active_std = masked.std(axis=1, skipna=True)
    
    # Avoid division by zero
    active_std = active_std.replace(0, np.nan)
    
    # Compute z-scores
    re_zscored = masked.sub(active_mean, axis=0).div(active_std, axis=0)
    
    # Restore zeros for inactive positions
    re_zscored = re_zscored.fillna(0)
    
    # Keep original zeros where positions were inactive
    re_zscored = re_zscored.where(active_mask, 0.0)
    
    return re_zscored

