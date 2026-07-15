"""
Data models for the data feed system

Contains OHLCV candlestick data model
"""

from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class OHLCV:
    """
    Open, High, Low, Close, Volume candlestick data
    
    Represents one time period of market data for a cryptocurrency.
    """
    
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def __post_init__(self):
        """Convert types and validate after initialization"""
        if isinstance(self.timestamp, (int, float)):
            self.timestamp = datetime.fromtimestamp(self.timestamp)
        elif isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        
        self.open = float(self.open)
        self.high = float(self.high)
        self.low = float(self.low)
        self.close = float(self.close)
        self.volume = float(self.volume)
        
        self.validate()
    
    def validate(self) -> None:
        """Validate OHLCV data integrity"""
        if self.high < self.low:
            raise ValueError(
                f"High ({self.high}) must be >= Low ({self.low})"
            )
        
        if self.open > self.high or self.open < self.low:
            raise ValueError(
                f"Open ({self.open}) must be between Low ({self.low}) "
                f"and High ({self.high})"
            )
        
        if self.close > self.high or self.close < self.low:
            raise ValueError(
                f"Close ({self.close}) must be between Low ({self.low}) "
                f"and High ({self.high})"
            )
        
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValueError(
                f"All prices must be positive. "
                f"Got: O={self.open}, H={self.high}, L={self.low}, C={self.close}"
            )
        
        if self.volume < 0:
            raise ValueError(f"Volume ({self.volume}) cannot be negative")
        
        if self.timestamp is None:
            raise ValueError("Timestamp cannot be None")
    
    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': float(self.open),
            'high': float(self.high),
            'low': float(self.low),
            'close': float(self.close),
            'volume': float(self.volume)
        }
    
    def to_list(self) -> list:
        """Convert to list format [timestamp_ms, o, h, l, c, v]"""
        timestamp_ms = int(self.timestamp.timestamp() * 1000)
        return [timestamp_ms, self.open, self.high, self.low, self.close, self.volume]
    
    @property
    def hl_range(self) -> float:
        """High-Low range (volatility measure)"""
        return self.high - self.low
    
    @property
    def typical_price(self) -> float:
        """Typical price (average of high, low, close)"""
        return (self.high + self.low + self.close) / 3
    
    @property
    def hl2(self) -> float:
        """HL2 - Simple average of high and low"""
        return (self.high + self.low) / 2
    
    @property
    def hlc3(self) -> float:
        """HLC3 - Average of high, low, close"""
        return self.typical_price
    
    @property
    def body(self) -> float:
        """Candle body size"""
        return abs(self.close - self.open)
    
    @property
    def upper_wick(self) -> float:
        """Upper wick size"""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        """Lower wick size"""
        return min(self.open, self.close) - self.low
    
    def __str__(self) -> str:
        """User-friendly string representation"""
        timestamp_str = self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        return (
            f"{timestamp_str} | "
            f"O:{self.open:.2f} H:{self.high:.2f} "
            f"L:{self.low:.2f} C:{self.close:.2f} "
            f"V:{self.volume:.0f}"
        )
    
    def __repr__(self) -> str:
        """Developer-friendly representation"""
        return (
            f"OHLCV("
            f"timestamp={self.timestamp.isoformat()}, "
            f"open={self.open}, high={self.high}, "
            f"low={self.low}, close={self.close}, "
            f"volume={self.volume})"
        )
               
    
