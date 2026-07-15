import requests
from typing import List
from datetime import datetime
from time import sleep
import logging

from src.data_feed.models import OHLCV
from src.data_feed.base import DataFeed
from src.data_feed.exceptions import APIError, ValidationError

logger = logging.getLogger(__name__)


class CoinGeckoFeed(DataFeed):
    """
    CoinGecko API implementation
    
    Fetches OHLCV data from CoinGecko's free API
    
    Features:
    - No API key required
    - Free tier (10-50 requests/minute)
    - Supports 100+ cryptocurrencies
    - Historical data up to 365 days
    
    Example:
        >>> feed = CoinGeckoFeed()
        >>> data = feed.get_ohlcv('bitcoin', days=30)
        >>> for candle in data:
        ...     print(candle)
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        """Initialize CoinGecko feed"""
        self.session = requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_count = 0
        logger.debug(f"CoinGeckoFeed initialized with timeout={timeout}s, max_retries={max_retries}")
    
    def get_ohlcv(self, symbol: str, days: int = 30) -> List[OHLCV]:
        """Fetch OHLCV data from CoinGecko"""
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        if not isinstance(days, int) or days < 1 or days > 365:
            raise ValueError(f"Days must be 1-365, got {days}")
        
        logger.info(f"Fetching {symbol} data for {days} days")
        
        try:
            data = self._fetch_from_api(symbol, days)
            
            ohlcv_list = self._parse_response(data, symbol)
            
            self.validate_data(ohlcv_list)
            
            logger.info(f"✅ Successfully fetched {len(ohlcv_list)} candles for {symbol}")
            return ohlcv_list
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise APIError(f"Failed to fetch {symbol} data: {e}")
        
        except ValueError as e:
            logger.error(f"Data parsing failed: {e}")
            raise ValidationError(f"Invalid data for {symbol}: {e}")
    
    def _fetch_from_api(self, symbol: str, days: int) -> list:
        """Fetch raw data from CoinGecko API with retry logic"""
        url = f"{self.BASE_URL}/coins/{symbol}/ohlc"
        
        params = {
            'vs_currency': 'usd',
            'days': days
        }
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"API call attempt {attempt + 1}/{self.max_retries}")
                
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                
                data = response.json()
                
                self.retry_count = 0
                
                return data
            
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time}s before retry...")
                    sleep(wait_time)
                else:
                    raise APIError(f"Timeout after {self.max_retries} attempts")
            
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time}s before retry...")
                    sleep(wait_time)
                else:
                    raise APIError(f"Connection failed after {self.max_retries} attempts")
            
            except requests.exceptions.HTTPError as e:
                if response.status_code == 404:
                    raise APIError(f"Symbol '{symbol}' not found on CoinGecko")
                elif response.status_code == 429:
                    raise APIError(f"Rate limited by CoinGecko. Wait before retrying.")
                else:
                    raise APIError(f"HTTP {response.status_code}: {e}")
    
    def _parse_response(self, data: list, symbol: str) -> List[OHLCV]:
        """Parse CoinGecko API response into OHLCV objects"""
        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data)}")
        
        if len(data) == 0:
            raise ValueError(f"No data returned for {symbol}")
        
        ohlcv_list = []
        
        for i, candle in enumerate(data):
            try:
                if len(candle) < 5:
                    logger.warning(f"Candle {i} incomplete: {candle}")
                    continue
                
                timestamp_ms = candle[0]
                open_price = candle[1]
                high = candle[2]
                low = candle[3]
                close = candle[4]
                
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                
                ohlcv = OHLCV(
                    timestamp=timestamp,
                    open=float(open_price),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=0.0
                )
                
                ohlcv_list.append(ohlcv)
            
            except (TypeError, ValueError, IndexError) as e:
                logger.warning(f"Failed to parse candle {i}: {e}")
                continue
        
        if not ohlcv_list:
            raise ValueError(f"Could not parse any valid candles from {symbol} data")
        
        logger.debug(f"Parsed {len(ohlcv_list)} candles from API response")
        return ohlcv_list


# ============================================================================
# SUPPORTED CRYPTOCURRENCIES (CoinGecko IDs)
# ============================================================================

COINGECKO_SYMBOLS = {
    'bitcoin': 'BTC',
    'ethereum': 'ETH',
    'binancecoin': 'BNB',
    'cardano': 'ADA',
    'solana': 'SOL',
    'ripple': 'XRP',
    'polkadot': 'DOT',
    'dogecoin': 'DOGE',
    'litecoin': 'LTC',
    'bitcoincash': 'BCH',
}