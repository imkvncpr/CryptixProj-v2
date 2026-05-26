import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, Any

from src.indicators.base_indicator import BaseIndicator


class MovingAverage(BaseIndicator):
    def __init__(self, period: int = 20, ma_type: str = 'SMA'):
        super().__init__(name=f"{ma_type}{period}")
        
        self.period = period
        self.ma_type = ma_type.upper()
        
        if self.ma_type not in ['SMA', 'EMA', 'WMA']:
            raise ValueError(f"Invalid MA type: {ma_type}. Use SMA, EMA or WMA")
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        close = data['close']
        
        if self.ma_type == 'SMA':
            self.values = close.rolling(window=self.period).mean()
            
        elif self.ma_type == 'EMA':
            self.values = close.ewm(span=self.period, adjust=False).mean()
            
        elif self.ma_type == 'WMA':
            weights = np.arange(1, self.period + 1)
            self.values = close.rolling(window=self.period).apply(
                lambda x: np.dot(x, weights) / weights.sum(),
                raw=True
            )
            
        return self.values
    
    def interpret(self, current_value: float, current_price: float = None) -> Dict[str, Any]:
        if current_price is None:
            raise ValueError("current_price is required")
        
        distance_pct = ((current_price - current_value) / current_value) * 100
        
        if current_price > current_value:
            if distance_pct > 5:
                signal, strength, confidence = "BUY", "STRONG", 85
                reasoning = f"Price {distance_pct:.1f}% above {self.ma_type}({self.period}) - Strong uptrend"
            elif distance_pct > 2:
                signal, strength, confidence = "BUY", "MODERATE", 70
                reasoning = f"Price {distance_pct:.1f}% above {self.ma_type}({self.period}) - Uptrend"
            else:
                signal, strength, confidence = "BUY", "WEAK", 55
                reasoning = f"Price slightly above {self.ma_type}({self.period}) - Mild uptrend"
            
        elif current_price < current_value:
            if abs(distance_pct) > 5:
                signal, strength, confidence = "SELL", "STRONG", 85
                reasoning = f"Price {abs(distance_pct):.1f}% below {self.ma_type}({self.period}) - Strong downtrend"
            elif abs(distance_pct) > 2:
                signal, strength, confidence = "SELL", "MODERATE", 70
                reasoning = f"Price {abs(distance_pct):.1f}% below {self.ma_type}({self.period}) - Downtrend"
            else:
                signal, strength, confidence = "SELL", "WEAK", 55
                reasoning = f"Price slightly below {self.ma_type}({self.period}) - Mild downtrend"
                
        else:
            signal, strength, confidence = "HOLD", "NEUTRAL", 50
            reasoning = f"Price at {self.ma_type}({self.period}) - Neutral"
                
        return {
            'indicator': self.name,
            'value': round(current_value, 2),
            'current_price': round(current_price, 2),
            'distance_pct': round(distance_pct, 2),
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'reasoning': reasoning
        }


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MOVING AVERAGES")
    print("=" * 60)
    
    from src.models.cryptos.bitcoin_predictor import BitcoinPredictor
    
    # Download Bitcoin data
    print("\n1. Downloading Bitcoin data...")
    btc = BitcoinPredictor()
    data = btc.download_data(period="3mo")
    current_price = data['close'].iloc[-1]
    
    print(f"   Current Bitcoin Price: ${current_price:,.2f}")
    print(f"   Data points: {len(data)}")
    
    # Test SMA
    print("\n2. Testing SMA (Simple Moving Average)...")
    sma20 = MovingAverage(period=20, ma_type='SMA')
    sma_values = sma20.calculate(data)
    sma_current = sma20.get_latest()
    
    print(f"   SMA(20): ${sma_current:,.2f}")
    result = sma20.interpret(sma_current, current_price)
    print(f"   Signal: {result['signal']} ({result['strength']})")
    print(f"   Confidence: {result['confidence']}%")
    print(f"   Reasoning: {result['reasoning']}")
    
    # Test EMA
    print("\n3. Testing EMA (Exponential Moving Average)...")
    ema20 = MovingAverage(period=20, ma_type='EMA')
    ema_values = ema20.calculate(data)
    ema_current = ema20.get_latest()
    
    print(f"   EMA(20): ${ema_current:,.2f}")
    result = ema20.interpret(ema_current, current_price)
    print(f"   Signal: {result['signal']} ({result['strength']})")
    print(f"   Confidence: {result['confidence']}%")
    print(f"   Reasoning: {result['reasoning']}")
    
    # Test WMA
    print("\n4. Testing WMA (Weighted Moving Average)...")
    wma20 = MovingAverage(period=20, ma_type='WMA')
    wma_values = wma20.calculate(data)
    wma_current = wma20.get_latest()
    
    print(f"   WMA(20): ${wma_current:,.2f}")
    result = wma20.interpret(wma_current, current_price)
    print(f"   Signal: {result['signal']} ({result['strength']})")
    print(f"   Confidence: {result['confidence']}%")
    print(f"   Reasoning: {result['reasoning']}")
    
    # Compare all three
    print("\n5. Comparison:")
    print(f"   SMA(20): ${sma_current:,.2f}")
    print(f"   EMA(20): ${ema_current:,.2f}")
    print(f"   WMA(20): ${wma_current:,.2f}")
    print(f"   Current:  ${current_price:,.2f}")
    print(f"\n   Note: EMA typically closer to current price (more reactive)")
    
    # Test different periods
    print("\n6. Testing different periods...")
    for period in [10, 20, 50]:
        ma = MovingAverage(period=period, ma_type='SMA')
        ma.calculate(data)
        ma_val = ma.get_latest()
        print(f"   SMA({period}): ${ma_val:,.2f}")
    
    print("\n" + "=" * 60)
    print("✅ MOVING AVERAGES TEST COMPLETE!")
    print("=" * 60)