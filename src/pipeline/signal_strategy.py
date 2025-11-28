"""
Signal Strategies - Each strategy produces a signal for a predictor.
1 predictor = 1 signal strategy.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any


class SignalStrategy(ABC):
    """
    Abstract base class for signal strategies.
    Each concrete strategy computes a signal from price data.
    """
    
    def __init__(self, name: str, **params):
        self.name = name
        self.params = params
    
    @abstractmethod
    def compute(self, close: pd.Series) -> pd.Series:
        """
        Compute the signal for a single asset.
        
        Args:
            close: Close price series for one asset
            
        Returns:
            Signal series (higher = more bullish)
        """
        pass
    
    def compute_universe(self, price_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute signals for all assets in the universe.
        
        Args:
            price_matrix: DataFrame with dates as index, symbols as columns
            
        Returns:
            DataFrame of signals with same shape as input
        """
        signals = {}
        for symbol in price_matrix.columns:
            signals[symbol] = self.compute(price_matrix[symbol])
        
        return pd.DataFrame(signals, index=price_matrix.index)
    
    def get_params(self) -> Dict[str, Any]:
        """Return strategy parameters."""
        return self.params.copy()


class MomentumStrategy(SignalStrategy):
    """
    Past-return momentum strategy.
    
    Signal = return over lookback period (shifted by 1 to avoid lookahead).
    Positive momentum suggests continued upward movement.
    """
    
    def __init__(self, lookback: int = 21):
        super().__init__(name="momentum", lookback=lookback)
        self.lookback = lookback
    
    def compute(self, close: pd.Series) -> pd.Series:
        """Compute momentum signal."""
        return close.pct_change(periods=self.lookback).shift(1)


class MeanReversionStrategy(SignalStrategy):
    """
    Z-score mean reversion strategy.
    
    Signal = negative z-score of returns (shifted by 1).
    Negative z-score (oversold) produces positive signal for mean reversion.
    """
    
    def __init__(self, lookback: int = 21):
        super().__init__(name="mean_reversion", lookback=lookback)
        self.lookback = lookback
    
    def compute(self, close: pd.Series) -> pd.Series:
        """Compute mean reversion signal."""
        ret = close.pct_change().fillna(0)
        mu = ret.rolling(self.lookback).mean()
        sigma = ret.rolling(self.lookback).std().replace(0, np.nan)
        z = (ret - mu) / sigma
        return -z.shift(1)  # Negative for mean-reversion


class EWMACrossoverStrategy(SignalStrategy):
    """
    Exponential Weighted Moving Average Crossover strategy.
    
    Signal = (fast_ewma - slow_ewma) / rolling_std
    
    Positive signals indicate uptrend (fast EWMA above slow EWMA).
    Negative signals indicate downtrend.
    Magnitude is normalized by volatility.
    """
    
    def __init__(self, fast_window: int = 12, slow_window: int = 26, std_window: int = 20):
        super().__init__(
            name="ewma_crossover",
            fast_window=fast_window,
            slow_window=slow_window,
            std_window=std_window
        )
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.std_window = std_window
    
    def compute(self, close: pd.Series) -> pd.Series:
        """Compute EWMA crossover signal."""
        fast_ewma = close.ewm(span=self.fast_window).mean()
        slow_ewma = close.ewm(span=self.slow_window).mean()
        rolling_std = close.rolling(window=self.std_window).std().replace(0, np.nan)
        
        signal = (fast_ewma - slow_ewma) / rolling_std
        return signal.shift(1).fillna(0)
