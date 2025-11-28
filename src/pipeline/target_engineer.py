"""
Target Engineering - Computes target variables (1-day forward returns).

TIMING CONVENTION:
------------------
- returns[t] = (price[t] - price[t-1]) / price[t-1]
  This is the return from close of day t-1 to close of day t.
  
- In backtesting, we use: portfolio_return[t] = weights[t-1] * returns[t]
  This means: weights determined at close of day t-1 are held during day t,
  and we realize the returns that occur from t-1 close to t close.
  
- Forward returns (targets): target[t] = return from t to t+forward_period
  Used for training/evaluation, shifted so we're predicting future returns.
"""

import pandas as pd
import numpy as np
from typing import Optional


class TargetEngineer:
    """
    Target engineer that computes target variables for the prediction task.
    
    Primary target: 1-day forward returns.
    
    Timing Convention:
    ------------------
    - compute_returns(): returns[t] = return from t-1 to t (standard pct_change)
    - compute_targets(): target[t] = return from t to t+forward_period
    
    For backtesting with Backtester class:
    - weights[t-1] * returns[t] = portfolio return at time t
    - This matches the convention: trade at close t-1, realize return at close t
    """
    
    def __init__(self, forward_period: int = 1):
        """
        Initialize the target engineer.
        
        Args:
            forward_period: Number of periods ahead for forward returns (default 1)
        """
        self.forward_period = forward_period
        self._targets: Optional[pd.DataFrame] = None
        self._returns: Optional[pd.DataFrame] = None
    
    def compute_targets(self, price_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute forward returns as targets.
        
        Timing:
        - target[t] = return from t to t+forward_period
        - Calculated as: returns.shift(-forward_period)
        
        Args:
            price_matrix: DataFrame with dates as index, symbols as columns
            
        Returns:
            Forward returns DataFrame (shifted so target[t] = return from t to t+1)
        """
        # Compute returns: returns[t] = (price[t] - price[t-1]) / price[t-1]
        returns = price_matrix.pct_change()
        
        # Shift backwards to get forward returns
        # target[t] = return from t to t+forward_period
        self._targets = returns.shift(-self.forward_period)
        self._returns = returns
        
        return self._targets
    
    def compute_returns(self, price_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute simple returns (not forward-looking).
        
        Timing:
        - returns[t] = (price[t] - price[t-1]) / price[t-1]
        - This is the return realized from close of t-1 to close of t
        
        For backtesting:
        - portfolio_return[t] = weights[t-1] * returns[t]
        - weights are determined at close of t-1
        - returns[t] is realized during day t
        
        Args:
            price_matrix: DataFrame with dates as index, symbols as columns
            
        Returns:
            Returns DataFrame
        """
        self._returns = price_matrix.pct_change()
        return self._returns
    
    def get_targets(self) -> Optional[pd.DataFrame]:
        """Return the most recent computed targets."""
        return self._targets
    
    def get_returns(self) -> Optional[pd.DataFrame]:
        """Return the most recent computed returns."""
        return self._returns
    
    def compute_binary_targets(self, price_matrix: pd.DataFrame,
                               threshold: float = 0.0) -> pd.DataFrame:
        """
        Compute binary classification targets.
        
        Args:
            price_matrix: DataFrame with dates as index, symbols as columns
            threshold: Return threshold for positive class (default 0.0)
            
        Returns:
            Binary targets (1 if return > threshold, 0 otherwise)
        """
        targets = self.compute_targets(price_matrix)
        return (targets > threshold).astype(int)
    
    def compute_tercile_targets(self, price_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute tercile classification targets (cross-sectional).
        
        Args:
            price_matrix: DataFrame with dates as index, symbols as columns
            
        Returns:
            Tercile targets (0=bottom, 1=middle, 2=top) for each date
        """
        targets = self.compute_targets(price_matrix)
        
        terciles = pd.DataFrame(index=targets.index, columns=targets.columns)
        
        for date in targets.index:
            row = targets.loc[date].dropna()
            if len(row) > 0:
                terciles.loc[date, row.index] = pd.qcut(
                    row, q=3, labels=[0, 1, 2], duplicates='drop'
                )
        
        return terciles.astype(float)
    
    def align_features_targets(self, features: pd.DataFrame, 
                               targets: pd.DataFrame) -> tuple:
        """
        Align features and targets, dropping NaN rows.
        
        Args:
            features: Feature DataFrame
            targets: Target DataFrame
            
        Returns:
            Tuple of (aligned_features, aligned_targets)
        """
        # Find common index
        common_idx = features.index.intersection(targets.index)
        
        aligned_features = features.loc[common_idx]
        aligned_targets = targets.loc[common_idx]
        
        # Drop rows with any NaN
        valid_mask = ~(aligned_features.isna().any(axis=1) | 
                       aligned_targets.isna().any(axis=1))
        
        return aligned_features[valid_mask], aligned_targets[valid_mask]
