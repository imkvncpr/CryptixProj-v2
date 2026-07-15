import json
import logging
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
from functools import wraps
import time

from ..data_feed.models import OHLCV
from ..data_feed.exceptions import CacheError, ValidationError

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_dir: str = ".cache", ttl_hours: int = 24):
        try:
            self.cache_dir = Path(cache_dir)
            self.ttl_hours = ttl_hours
            self.ttl_seconds = ttl_hours * 3600
        except Exception as e:
            logger.error(f"Failed to create cache directory: {e}")
            raise CacheError(f"Cannot create cache directory: {e}")
        
        logger.info(f"CacheManager initialized: dir={self.cache_dir}, ttl={ttl_hours}h")
        
    def get(self, symbol: str)-> Optional[List[OHLCV]]:
        
        cache_file = self.cache_dir / f"{symbol}.json"
        
        if not cache_file.exists():
            logger.debug(f"Cache miss: {symbol} (file not found)")
            return None
        
        if self.is_expired(symbol):
            logger.debug(f"Cache expired: {symbol}")
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete expired cache: {e}")
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                
            candles = cache_data.get('candles', [])
            
            ohlcv_list = []
            for candle in candles:
                try:
                    ohlcv = ohlcv(
                        timestamp=datetime.fromisoformat(candle['timestamp']),
                        open=float(candle['open']),
                        high=float(candle['high']),
                        low=float(candle['low']),
                        close=float(candle['close']),
                        volume=float(candle['volume'])
                    )
                    ohlcv_list.append(ohlcv)
                except (TypeError, ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse candle from cache: {e}")
                    continue
                
            if not ohlcv_list:
                logger.warning(f"No valid candles in cache: {symbol}")
                return None
            
            logger.info(f"Cache hit: {symbol} ({len(ohlcv_list)} candles)")
            return ohlcv_list
        
        except json.JSONDecodeError:
            logger.error(f"Corrupted cache file: {cache_file}")
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete corrupted cache: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Cache read error: {e}")
            raise CacheError(f"Failed to read cache: {e}")
        
    def set(self, symbol: str, data: List[OHLCV])-> bool:
        if not data:
            logger.warning(f"Cannot cache empty data for {symbol}")
            raise CacheError("Cannot cache empty data")
        
        try:
            from ..data_feed.validators import validate_ohlcv_list
            validate_ohlcv_list(data)
        except ValidationError as e:
            logger.error(f"Invalid data for cache: {e}")
            raise CacheError(f"Cannot cache invalid data: {e}")
        
        cache_data = {
            'symbol': symbol,
            'cached_at': datetime.utcnow().isoformat(),
            'count': len(data),
            'candles': [ohlcv.to_dict() for ohlcv in data]
        }
        
        cache_file = self.cache_dir / f"{symbol}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            logger.info(f"Cached {symbol}: {len(data)} candles")
            return True
        
        except Exception as e:
            logger.error(f"Cache write failed: {e}")
            raise CacheError(f"Failed to write cache: {e}")
        
    def is_expired(self, symbol: str)-> bool:
        cache_file = self.cache_dir / f"{symbol}.json"
        
        if not cache_file.exists():
            return True
        
        try:
            file_mtime = cache_file.stat().st_mtime
            current_time = time.time()
            age_seconds = current_time - file_mtime
            
            if age_seconds > self.ttl_seconds:
                logger.debug(
                    f"Cache expired for {symbol}: "
                    f"age={age_seconds:.0f}s, ttl={self.ttl_seconds}s"
                )
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to check expiration: {e}")
            return True
        
    def delete(self, symbol: str)-> bool:
        cache_file = self.cache_dir / f"{symbol}.json"
        
        if not cache_file.exists():
            logger.debug(f"Cache not found: {symbol}")
            return False
        
        try:
            cache_file.unlink()
            logger.info(f"Deleted cache: {symbol}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete cache: {e}")
            return False
        
    def clear(self, symbol: Optional[str] = None)-> bool:
        if symbol:
            return self.delete(symbol)
        
        try:
            count = 0
            for cache_file in self.cache_dir.glob('*.json'):
                try:
                    cache_file.unlink()
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {cache_file}: {e}")
        
            logger.info(f"Cleared {count} cache files")
            return True
    
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
        
    def get_cache_stats(self)-> Dict:
        stats = {
            'total_files': 0,
            'total_size_bytes': 0,
            'total_size_mb': 0.0,
            'symbols': {}
        }
        
        try:
            current_time = time.time()
            
            for cache_file in self.cache_dir.glob('*.json'):
                symbol = cache_file.stem
                size_bytes = cache_file.stat().st_size
                file_mtime = cache_file.stat().st_mtime
                age_hours = (current_time - file_mtime) / 3600
                
                stats['symbols'][symbol] = {
                    'size_bytes': size_bytes,
                    'age_hours': age_hours,
                    'size_mb': size_bytes / (1024 * 1024)
                }
                
                stats['total_files'] += 1
                stats['total_size_bytes'] += size_bytes
                
            stats['total_size_mb'] = stats['total_size_bytes'] / (1024 * 1024)
            
            logger.debug(f"Cache stats: {stats['total_files']} files, {stats['total_size_mb']:.2f} MB")
            return stats
        
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return stats
        
# ============================================================================
# CACHE DECORATOR
# ============================================================================

def cache_ohlcv(ttl_hours: int = 24):
    
    def decorator(func):
        cache = CacheManager(ttl_hours=ttl_hours)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            symbol = args[0] if args else None
            
            if not symbol:
                logger.warning("Cannot cache without symbol argument")
                return func(*args, **kwargs)
            
            cache_data = cache.get(symbol)
            if cache_data is not None:
                logger.info(f"Cache hit for{symbol}")
                return cache_data
            
            logger.info(f"Cache miss for {symbol}, calling function")
            result = func(*args, **kwargs)
            
            try:
                cache.set(symbol, result)
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")
                
            return result
        
        return wrapper
    
    return decorator


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_cache_path(symbol: str, cache_dir: str = "cache")-> Path:
    return Path(cache_dir) / f"{symbol}.json"

def format_cache_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} TB"