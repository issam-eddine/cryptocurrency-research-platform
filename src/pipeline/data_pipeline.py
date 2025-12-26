"""
Data Pipeline - Handles data extraction and preprocessing.
Combines functionality from data_fetch.py and preprocessing.py into a single class.
"""

import ccxt
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class DataPipeline:
    """
    Unified data pipeline for fetching and preprocessing cryptocurrency data.
    
    Handles:
    - OHLCV data fetching from Binance via ccxt
    - Caching to parquet files
    - Data cleaning and normalization
    - Building aligned price matrices across multiple symbols
    """
    
    def __init__(self, cache_dir: str = "data/raw", exchange_id: str = "binanceus",
                 max_retries: int = 5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        
        self.exchange = ccxt.binanceus({
            'enableRateLimit': True,
            'timeout': 4000,
        })
        
        self._raw_data: Dict[str, pd.DataFrame] = {}
        self._price_matrix: Optional[pd.DataFrame] = None
    
    def fetch(self, symbols: List[str], timeframe: str = "1h",
              start_date: Optional[datetime] = None,
              end_date: Optional[datetime] = None,
              force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.
        
        Args:
            symbols: List of trading pairs (e.g., ["BTC/USDT", "ETH/USDT"])
            timeframe: Candle timeframe (default "1d")
            start_date: Optional start date filter
            end_date: Optional end date filter
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            Dictionary mapping symbols to their OHLCV DataFrames
        """
        for symbol in symbols:
            df = self._fetch_single(symbol, timeframe, force_refresh)
            
            if not df.empty and start_date and end_date:
                df = df.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
            
            self._raw_data[symbol] = df
        
        return self._raw_data
    
    def _fetch_single(self, symbol: str, timeframe: str = "1h",
                      force_refresh: bool = False) -> pd.DataFrame:
        """Fetch OHLCV for a single symbol with caching."""
        if not force_refresh:
            cached = self._load_from_cache(symbol, timeframe)
            if cached is not None:
                print(f"Loaded {symbol} from cache")
                return cached
        
        print(f"Fetching {symbol} from exchange")
        df = self._fetch_from_exchange(symbol, timeframe)
        
        if not df.empty:
            self._save_to_cache(symbol, df, timeframe)
        
        return df
    
    def _fetch_from_exchange(self, symbol: str, timeframe: str = "1h",
                             limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV directly from the exchange with retry logic."""
        all_rows = []
        
        # Start from 10 years ago
        since_param = int((datetime.now().timestamp() - (10 * 365 * 24 * 60 * 60)) * 1000)
        consecutive_failures = 0
        
        while True:
            try:
                chunk = self.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=since_param, limit=limit
                )
                consecutive_failures = 0  # Reset on success
            except ccxt.BaseError as e:
                consecutive_failures += 1
                if consecutive_failures >= self.max_retries:
                    print(f"Max retries ({self.max_retries}) exceeded for {symbol}: {e}")
                    break
                # Exponential backoff: 2, 4, 8, 16, 32 seconds
                wait_time = 2 ** consecutive_failures
                print(f"Error: {e}, retrying in {wait_time}s (attempt {consecutive_failures}/{self.max_retries})...")
                time.sleep(wait_time)
                continue
            
            if not chunk or len(chunk) < limit:
                all_rows.extend(chunk if chunk else [])
                break
            
            all_rows.extend(chunk)
            since_param = chunk[-1][0] + 1
        
        if not all_rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.set_index('datetime').drop(columns=['ts'])
        return df.sort_index()
    
    def _save_to_cache(self, symbol: str, df: pd.DataFrame, timeframe: str):
        """Save OHLCV to parquet cache."""
        path = self.cache_dir / f"{symbol.replace('/', '-')}__{timeframe}.parquet"
        df.to_parquet(path)
    
    def _load_from_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load from parquet cache if exists."""
        path = self.cache_dir / f"{symbol.replace('/', '-')}__{timeframe}.parquet"
        return pd.read_parquet(path) if path.exists() else None
    
    def preprocess(self) -> Dict[str, pd.DataFrame]:
        """
        Clean and normalize all fetched data.
        
        Returns:
            Dictionary of cleaned DataFrames
        """
        cleaned = {}
        for symbol, df in self._raw_data.items():
            cleaned[symbol] = self._clean_dataframe(df)
        
        self._raw_data = cleaned
        return cleaned
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize a single price DataFrame."""
        if df.empty:
            return df
        
        df = df.copy()
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    def get_price_matrix(self, freq: str = "1h") -> pd.DataFrame:
        """
        Build aligned close price matrix across all symbols.
        
        Args:
            freq: Resampling frequency (default "1h" for hourly)
            
        Returns:
            DataFrame with dates as index, symbols as columns, close prices as values
        """
        closes = {}
        
        for symbol, df in self._raw_data.items():
            if df.empty or "close" not in df.columns:
                continue
            
            resampled = df.resample(freq).last().ffill()
            closes[symbol] = resampled["close"]
        
        if not closes:
            return pd.DataFrame()
        
        self._price_matrix = pd.concat(closes.values(), axis=1, keys=closes.keys())
        self._price_matrix = self._price_matrix.dropna(axis=1, how="all")
        
        return self._price_matrix
    
    def get_top_symbols(self, n: int = 8, quote: str = "USDT") -> List[str]:
        """Get top-n trading symbols by volume."""
        try:
            markets = self.exchange.fetch_markets()
            filtered = [m for m in markets if m.get("quote") == quote and m.get("active")]
            filtered.sort(
                key=lambda m: float(m.get("info", {}).get("quoteVolume", 0) or 0),
                reverse=True
            )
            return [m["symbol"] for m in filtered][:n]
        except ccxt.BaseError as e:
            print(f"Failed to fetch markets: {e}, using default symbols")
            return ["BTC/USDT", "ETH/USDT", "XRP/USDT", "BNB/USDT",
                    "SOL/USDT", "TRX/USDT", "DOGE/USDT", "ADA/USDT"][:n]
