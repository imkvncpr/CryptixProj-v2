
"""
CryptoCoin Predictor Base Class

Provides YFinance-specific functionality for cryptocurrency prediction.
All specific crypto predictors (Bitcoin, Ethereum, etc.) inherit from this.
"""

import pandas as pd
import yfinance as yf
from typing import Optional, Dict
from loguru import logger
from src.models.base.crypto_predictor import CryptoPredictor


class CryptoCoinPredictor(CryptoPredictor):
    """
    Base class for YFinance-based cryptocurrency predictors.
    
    Inherits from CryptoPredictor and adds:
    - YFinance data downloading
    - Crypto-specific data cleaning
    - Market info retrieval
    - Common crypto parameters
    """
    
    def __init__(
        self,
        ticker: str,
        name: str,
        look_back: int = 60,
        units: list = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        model_version: str = "v1.0"
    ):
        """
        Initialize cryptocurrency predictor.
        
        Args:
            ticker: YFinance ticker symbol (e.g., 'BTC-USD')
            name: Cryptocurrency name (e.g., 'Bitcoin')
            look_back: Number of days to look back
            units: LSTM units per layer
            dropout: Dropout rate
            learning_rate: Learning rate for Adam optimizer
            model_version: Model version string
        """
        super().__init__(
            ticker=ticker,
            look_back=look_back,
            units=units,
            dropout=dropout,
            learning_rate=learning_rate,
            model_version=model_version
        )
        
        self.name = name
        logger.info(f"Initialized {name}Predictor for {ticker}")
    
    def download_data(
        self,
        period: str = "2y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Download cryptocurrency price data from Yahoo Finance.
        
        Args:
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data or None if download fails
        """
        try:
            logger.info(
                f"Downloading {self.name} data: "
                f"period={period if not start else f'{start} to {end}'}, "
                f"interval={interval}"
            )
            
            if start and end:
                data = yf.download(
                    self.ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False
                )
            else:
                data = yf.download(
                    self.ticker,
                    period=period,
                    interval=interval,
                    progress=False
                )
            
            if data.empty:
                logger.warning(f"No data found for {self.name} with ticker {self.ticker}")
                return None
            
            data = self._clean_data(data)
            
            if data is None:
                return None
            
            logger.success(f"Downloaded {len(data)} data points for {self.name}")
            logger.debug(f"Date range: {data.index[0]} to {data.index[-1]}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading {self.name} data: {e}")
            return None
    
    def _clean_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Clean and validate downloaded data.
        
        Args:
            data: Raw data from YFinance
            
        Returns:
            Cleaned DataFrame or None if validation fails
        """
        try:
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Convert column names to lowercase
            data.columns = [col.lower() for col in data.columns]
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                logger.error(
                    f"Missing required columns: {missing_cols}. "
                    f"Available columns: {data.columns.tolist()}"
                )
                return None
            
            # Remove rows with NaN values
            original_length = len(data)
            data = data.dropna()
            
            if len(data) < original_length:
                logger.warning(
                    f"Removed {original_length - len(data)} rows with NaN values"
                )
            
            # Validate data types
            for col in required_cols:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    logger.error(f"Column {col} has non-numeric data type")
                    return None
            
            # Validate positive prices
            if (data['close'] <= 0).any():
                logger.error("Close prices contain non-positive values")
                return None
            
            logger.debug(f"Data validation passed: {len(data)} rows, {len(data.columns)} columns")
            
            return data
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return None
    
    def get_crypto_info(self) -> Dict:
        """
        Get current cryptocurrency information from Yahoo Finance.
        
        Returns:
            Dictionary with crypto info (price, market cap, volume, etc.)
        """
        try:
            ticker_obj = yf.Ticker(self.ticker)
            info = ticker_obj.info
            
            crypto_info = {
                'name': info.get('longName', self.name),
                'symbol': info.get('symbol', self.ticker.split('-')[0]),
                'current_price': info.get('regularMarketPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'market_cap': info.get('marketCap', 0),
                'volume_24h': info.get('volume24Hr', info.get('volume', 0)),
                'circulating_supply': info.get('circulatingSupply', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'year_high': info.get('fiftyTwoWeekHigh', 0),
                'year_low': info.get('fiftyTwoWeekLow', 0),
            }
            
            logger.info(f"{self.name} Price: ${crypto_info['current_price']:,.2f}")
            return crypto_info
            
        except Exception as e:
            logger.error(f"Error fetching {self.name} info: {e}")
            return {
                'name': self.name,
                'symbol': self.ticker.split('-')[0],
                'current_price': 0
            }
    
    def get_price_summary(self, data: pd.DataFrame) -> Dict:
        """
        Get summary statistics for price data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            close_prices = data['close']
            
            summary = {
                'count': len(close_prices),
                'mean': close_prices.mean(),
                'std': close_prices.std(),
                'min': close_prices.min(),
                'max': close_prices.max(),
                'median': close_prices.median(),
                'current': close_prices.iloc[-1],
                'change': close_prices.iloc[-1] - close_prices.iloc[0],
                'change_pct': ((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]) * 100,
                'volatility': close_prices.pct_change().std() * 100
            }
            
            logger.debug(f"Price summary: ${summary['mean']:,.2f} avg, {summary['volatility']:.2f}% volatility")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating price summary: {e}")
            return {}
    
    def __repr__(self) -> str:
        return f"{self.name}Predictor(ticker='{self.ticker}', version='{self.model_version}')"


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("CRYPTOCOIN PREDICTOR BASE CLASS - TEST")
    print("=" * 70)
    
    predictor = CryptoCoinPredictor(
        ticker="BTC-USD",
        name="Bitcoin"
    )
    
    print(f"\n1. Created predictor: {predictor}")
    
    print("\n2. Downloading data...")
    data = predictor.download_data(period="3mo")
    
    if data is not None:
        print(f"   ✓ Downloaded {len(data)} days of data")
        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        print("\n3. Price summary:")
        summary = predictor.get_price_summary(data)
        print(f"   Current price: ${summary['current']:,.2f}")
        print(f"   Average price: ${summary['mean']:,.2f}")
        print(f"   Price range: ${summary['min']:,.2f} - ${summary['max']:,.2f}")
        print(f"   Volatility: {summary['volatility']:.2f}%")
        
        print("\n4. Current market info:")
        info = predictor.get_crypto_info()
        print(f"   Market cap: ${info['market_cap']:,.0f}")
        print(f"   24h volume: {info['volume_24h']:,.0f}")
        
        print("\n" + "=" * 70)
        print("BASE CLASS TEST COMPLETE!")
        print("=" * 70)
    else:
        print("   ✗ Failed to download data")