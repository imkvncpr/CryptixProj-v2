"""
Base Indicator Class

All technical indicators inherit from this abstract base class.
Provides common interface and utilities.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union


class BaseIndicator(ABC):
    """
    Abstract base class for all technical indicators.
    
    All indicators must implement:
    - calculate() method: Compute indicator values
    - interpret() method: Generate trading signals
    """
    
    def __init__(self, name: str):
        """
        Initialize indicator.
        
        Args:
            name: Indicator name (e.g., "RSI", "MACD")
        """
        self.name = name
        self.values = None
        self._last_calculated = None
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate indicator values from OHLCV data.
        
        Args:
            data: DataFrame with columns: open, high, low, close, volume
            
        Returns:
            Series or DataFrame with indicator values
        """
        pass
    
    @abstractmethod
    def interpret(self, current_value: Union[float, Dict]) -> Dict[str, Any]:
        """
        Interpret current indicator value and generate signal.
        
        Args:
            current_value: Most recent indicator value(s)
            
        Returns:
            Dictionary with:
            - signal: BUY/SELL/HOLD
            - strength: STRONG/WEAK
            - confidence: 0-100
            - reasoning: Explanation
        """
        pass
    
    def get_latest(self) -> Union[float, Dict]:
        """Get the most recent indicator value"""
        if self.values is None:
            raise ValueError(f"{self.name} not calculated yet. Call calculate() first.")
        
        if isinstance(self.values, pd.Series):
            return self.values.iloc[-1]
        elif isinstance(self.values, pd.DataFrame):
            return self.values.iloc[-1].to_dict()
        else:
            return self.values
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data has required columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if valid, raises ValueError if not
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(data) == 0:
            raise ValueError("Data is empty")
        
        return True
    
    def __repr__(self) -> str:
        return f"{self.name}Indicator()"


# Helper functions
def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return series.rolling(window=period).mean()


def calculate_std(series: pd.Series, period: int) -> pd.Series:
    """Calculate Standard Deviation"""
    return series.rolling(window=period).std()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("BASE INDICATOR CLASS")
    print("=" * 70)
    print("\nBaseIndicator is an abstract base class.")
    print("All indicators must inherit from it and implement:")
    print("  - calculate() method")
    print("  - interpret() method")
    print("\nHelper functions available:")
    print("  - calculate_ema(series, period)")
    print("  - calculate_sma(series, period)")
    print("  - calculate_std(series, period)")
    print("\n" + "=" * 70)
    print("✅ Base class ready for use!")
    print("=" * 70)