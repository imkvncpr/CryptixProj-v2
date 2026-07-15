import csv
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import random

from .models import Candle

logger = logging.getLogger(__name__)

class HistoricalDataHandler:
    def __init__(self, data_dir: str = './data'):
        self.data_dir = Path(data_dir)
        self.cache: Dict[str, List[Candle]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"HistoricalDataHandler initialized with data_dir: {self.data_dir}")
        
    def load_csv(self, symbol: str, file_path: str) -> List[Candle]:
        cache_key = f"{symbol}_{file_path}"  
        
        if cache_key in self.cache:
            self.logger.debug(f"Loading {symbol} from cache")
            return self.cache[cache_key]
        
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            self.logger.error(f"CSV file not found: {file_path}")
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        self.logger.info(f"Loading {symbol} from CSV: {file_path}")
        candles: List[Candle] = []
        
        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or 'timestamp' not in reader.fieldnames:
                    raise ValueError("CSV must have 'timestamp' column")
                
                for row in reader:
                    try:
                        timestamp_str = row.get('timestamp', '').strip()
                        open_str = row.get('open', '').strip()
                        high_str = row.get('high', '').strip()
                        low_str = row.get('low', '').strip()
                        close_str = row.get('close', '').strip()
                        volume_str = row.get('volume', '').strip()
                        
                        if not all([timestamp_str, open_str, high_str, low_str, close_str, volume_str]):
                            continue
                        
                        try:
                            if 'T' in timestamp_str:
                                timestamp = datetime.fromisoformat(timestamp_str)
                            else:
                                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            continue
                        
                        try:
                            open_price = float(open_str)
                            high = float(high_str)
                            low = float(low_str)
                            close = float(close_str)
                            volume = float(volume_str)
                        except ValueError:
                            continue
                        
                        try:
                            candle = Candle(timestamp = timestamp, open = open_price, high = high, low = low, close = close, volume = volume, symbol = symbol)
                            candles.append(candle)
                        except ValueError:
                            continue
                    except Exception:
                        continue
        
        except Exception as e:
            self.logger.error(f"Failed to read CSV file {file_path}: {e}")
            raise
        
        if len(candles) == 0:
            raise ValueError(f"No valid candles found in {file_path}")
        
        self.validate_candles(candles)
        self.cache[cache_key] = candles
        return candles
    
    def validate_candles(self, candles: List[Candle]) -> bool:
        if not candles:
            raise ValueError("Candle list is empty")
        
        for i in range(len(candles) - 1):
            if candles[i].timestamp >= candles[i + 1].timestamp:
                raise ValueError("Timestamps out of order")
        
        return True
    
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Candle]:
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")
        
        all_candles = self.get_all_candles(symbol)
        
        if not all_candles:
            return []
        
        filtered = [c for c in all_candles if start_date <= c.timestamp <= end_date]
        return filtered
    
    def get_all_candles(self, symbol: str) -> List[Candle]:
        for cache_key, candles in self.cache.items():
            if symbol in cache_key:
                return candles
        
        return []
    
    def get_candle_count(self, symbol: str) -> int:
        return len(self.get_all_candles(symbol))
    
    def get_date_range(self, symbol: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        candles = self.get_all_candles(symbol)
        if not candles:
            return None, None
        
        return candles[0].timestamp, candles[-1].timestamp
    
    def list_available_symbols(self) -> List[str]:
        symbols = set()
        
        for cache_key in self.cache.keys():
            symbol = cache_key.split('_')[0]
            symbols.add(symbol)
        
        return sorted(list(symbols))
    
    def clear_cache(self, symbol: str = None) -> None:
        if symbol is None:
            self.cache.clear()
            self.logger.info("Cleared entire cache")
        else:
            keys_to_delete = [k for k in self.cache.keys() if symbol in k]
            for key in keys_to_delete:
                del self.cache[key]
            self.logger.info(f"Cleared cache for {symbol}")
    
    def get_cache_stats(self) -> Dict[str, any]:
        total_candles = sum(len(c) for c in self.cache.values())
        total_symbols = len(self.list_available_symbols())
        memory_kb = (total_candles * 200) / 1024
        
        return {
            'symbols': total_symbols,
            'total_candles': total_candles,
            'cache_size_kb': round(memory_kb, 2),
            'cached_symbols': self.list_available_symbols()
        }
    
    def __repr__(self) -> str:
        stats = self.get_cache_stats()
        return (
            f"HistoricalDataHandler | "
            f"{stats['symbols']} symbols | "
            f"{stats['total_candles']} candles"
        )


def create_sample_csv(file_path: str, symbol: str = 'BTC/USD',
                      num_days: int = 365, start_price: float = 28500.0) -> None:
    """Create sample CSV data for testing"""
    
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating sample CSV: {file_path}")
    
    candles = []
    current_price = start_price
    current_date = datetime(2023, 1, 1)
    
    for day in range(num_days):
        daily_change = random.gauss(0, 500)
        volatility = abs(random.gauss(0, 300))
        
        open_price = current_price
        close_price = current_price + daily_change
        high_price = max(open_price, close_price) + volatility
        low_price = min(open_price, close_price) - volatility
        
        volume = random.uniform(100, 300)
        
        candles.append({
            'timestamp': current_date.strftime('%Y-%m-%d %H:%M:%S'),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 1)
        })
        
        current_price = close_price
        current_date += timedelta(days=1)
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        writer.writeheader()
        writer.writerows(candles)
    
    logger.info(f"Created {num_days} days of sample data")