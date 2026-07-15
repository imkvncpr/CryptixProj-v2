import logging
from typing import List, Optional, Dict
from datetime import datetime

from ..data_feed.models import OHLCV
from ..data_feed.base import DataFeed
from ..data_feed.coingecko import CoinGeckoFeed
from ..data_feed.cache import CacheManager, cache_ohlcv
from src.data_feed.exceptions import APIError, ValidationError, DataFeedError
from src.data_feed import validators

logger = logging.getLogger(__name__)

class DataFeedManager:
    def __init__(self,
                 feed_type: str = 'coingecko',
                 cache_dir: str = '.cache',
                 cache_ttl_hours: int = 24,
                use_cache: bool = True):
        
        if feed_type == 'coingecko':
            self.feed = CoinGeckoFeed()
            logger.info("Initialized CoinGeckoFeed")
        else:
            raise ValueError(f"Unknown feed type: {feed_type}")
        
        if use_cache:
            self.cache = CacheManager(cache_dir=cache_dir, ttl_hours=cache_ttl_hours)
            logger.info(f"Cache enabled: {cache_dir} (TTL: {cache_ttl_hours}h)")
        else:
            self.cache = None
            logger.info("Cache disabled")
            
        logger.info("DataFeedManager initialized successfully")
        
    def get_data(
        self,
        symbol: str,
        days: int = 30,
        use_cache: Optional[bool] = None,
        validate: bool = True
    )-> List[OHLCV]:
        
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        if not isinstance(days, int) or days < 1 or days > 365:
            raise ValueError(f"Days must be 1-365, got {days}")
        
        logger.info(f"Getting data for {symbol} ({days} days)")
        
        use_cache_flag = use_cache if use_cache is not None else self.use_cache
        
        if use_cache_flag and self.cache is not None:
            cached_data = self.cache.get(symbol)
            if cached_data is not None:
                logger.info(f"✅ Cache hit: {symbol} ({len(cached_data)} candles)")
                return cached_data
            
        try:
            logger.debug(f"Cache miss for {symbol}, fetching from API")
            data = self.feed.get_ohlcv(symbol, days)
            logger.debug(f"Fetched {len(data)} candles from API")
            
        except APIError as e:
            logger.error(f"API error for {symbol}: {e}")
            
        if validate:
            try:
                validators.validate_ohlcv_list(data)
                logger.debug(f"Validation passed for {symbol}")
            except ValidationError as e:
                logger.warning(f"Failed to cache {symbol}: {e}")
                
                
        logger.info(f"✅ Got {len(data)} candles for {symbol}")
        return data
    
    def get_multiple(
        self, 
        symbols: List[str],
        days: int = 30
    )->Dict[str, Optional[List[OHLCV]]]:
        logger.info(f"Fetching {len(symbols)} symbols")
        results = {}
        
        for symbol in symbols:
            try:
                data = self.get_data(symbol, days)
                results[symbol] = data
            except (APIError, ValidationError, ValueError) as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                results[symbol] = None
                
        success_count = sum(1 for v in results.values() if v is not None)
        logger.info(f"Fetched {success_count}/{len(symbols)} symbols successfully")
        
        return results
    
    def check_quality(self, symbol: str, days: int = 30)-> Dict:
        try:
            logger.info(f"Checking quality for {symbol}")
            
            data = self.get_data(symbol, days, validate=False)
            
            if not data:
                logger.error(f"No data for quality check: {symbol}")
                return{
                    'is_valid': False,
                    'error': f'No data available for {symbol}',
                    'symbol': symbol
                }
                
            report = validators.check_data_quality(data)
            
            report['symbol'] = symbol
            report['days'] = days
            report['fetched_at'] = datetime.utctimetuple().isoformat
            
            logger.info(
                f"Quality report for {symbol}: "
                f"score={report['quality_score']:.0f}, "
                f"issues={len(report.get('issues', []))}, "
                f"candles={report.get('total_candles', 0)}"
            )
            
            return report
        
        except Exception as e:
            logger.error(f"Quality check failed for {symbol}: {e}")
            return {
                'is_valid': False,
                'error': str(e),
                'symbol': symbol
            }
            
    def clear_cache(self, symbol: Optional[str] = None)-> bool:
        if self.cache is None:
            logger.warning("Cache not enabled")
            return False
        
        try:
            success = self.cache.clear(symbol)
            
            if symbol:
                logger.info(f"Cleared cache for {symbol}")
            else:
                logger.info("Cleared all cache")
                
            return success
        
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
        
    def get_cache_stats(self)-> Dict:
        if self.cache is None:
            return {'error': 'Cache not enabled'}
        
        try:
            stats = self.cache.get_cache_stats()
            logger.debug(f"Cache stats: {stats['total_files']} files, {stats['total_size_mb']:.2f} MB")
            return stats
        
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'error': str(e)}
        
    def health_check(self)-> Dict:
        status = {
            'feed': 'unknown',
            'cache': 'unknown',
            'overall': 'unknown'
        }
        
        try:
            logger.debug("Health check: testing feed...")
            data = self.feed.get_ohlcv('bitcoin', days=1)
            status['feed'] = 'healthy' if data and len(data) > 0 else 'error'
        
        except Exception as e:
            logger.warning(f"Feed health check failed: {e}")
            status['feed'] = 'error'
            
        status['cache'] = 'enabled' if self.cache else 'disabled'
        
        status['overall'] = 'healthy' if status ['feed'] == 'healthy' else 'error'
        
        logger.info(f"Health check: {status}")
        return status
    
    def get_supported_symbols(self)-> List[str]:
        if self.feed_type == 'coingecko':
            from ..data_feed.coingecko import  COINGECKO_SYMBOLS
            return list(COINGECKO_SYMBOLS.keys())
        
        return []
    
    def __repr__(self) -> str:
        cache_status = "enabled" if self.use_cache else "disabled"
        return f"DataFeedManager(feed={self.feed_type}, cache={cache_status})"
            
    