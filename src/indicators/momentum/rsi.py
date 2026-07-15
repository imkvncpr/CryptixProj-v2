import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from typing import Dict, Any

from src.indicators.base_indicator import BaseIndicator

class RSI(BaseIndicator):
    def __init__(self, period: int = 14):
        super().__init__(name = 'RSI')
        self.period = period
        
    def calculate(self, data: pd.DataFrame)-> pd.Series:
        self.validate_data(data)
        close = data['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window = self.period).mean()
        avg_loss = loss.rolling(window = self.period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        self.values = rsi
        return rsi
    
    def interpret(self, current_value: float)-> Dict[str, Any]:
        if current_value > 70:
            signal = "SELL"
            strength = "STRONG" if current_value > 80 else "MODERATE"
            confidence = 85 if current_value > 80 else 70
            reasoning = f"RSI at {current_value:.1f} - Overbought Condition"
        elif current_value < 30:
            signal = "BUY"
            strength = "STRONG" if current_value < 20 else "MODERATE"
            confidence = 85 if current_value < 20 else 70
            reasoning = f"RSI at {current_value:.1f} - Oversold Condition" 
        else:
            signal = "HOLD"
            strength = "NEUTRAL"
            confidence = 50
            reasoning = f"RSI at {current_value:.1f} - Neutral Zone"
            
        return{
            'indicator': 'RSI',
            'value': round(current_value, 2),
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'reasoning' : reasoning,
            }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_rsi(closes: list, period: int = 14) -> float:

    if len(closes) < period + 1:
        return None
    
    # Use only last period + 1 candles
    relevant_closes = closes[-(period + 1):]
    
    # Calculate changes
    changes = []
    for i in range(1, len(relevant_closes)):
        change = relevant_closes[i] - relevant_closes[i - 1]
        changes.append(change)
    
    # Separate gains and losses
    gains = []
    losses = []
    for change in changes:
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    # Calculate averages
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    # Calculate RS and RSI
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    return rsi


def is_overbought(rsi: float, threshold: float = 70.0) -> bool:
    return rsi > threshold


def is_oversold(rsi: float, threshold: float = 30.0) -> bool:
    return rsi < threshold


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING RSI INDICATOR")
    print("=" * 60)

    from src.models.cryptos.bitcoin_predictor import BitcoinPredictor

    # Download Bitcoin data
    print("\n1. Downloading Bitcoin data...")
    btc = BitcoinPredictor()
    data = btc.download_data(period="3mo")

    print(f"   Data points: {len(data)}")
    print(f"   Current price: ${data['close'].iloc[-1]:,.2f}")

    # Create and calculate RSI
    print("\n2. Calculating RSI(14)...")
    rsi = RSI(period=14)
    rsi_values = rsi.calculate(data)
    current_rsi = rsi.get_latest()

    print(f"   Current RSI: {current_rsi:.2f}")

    # Interpret signal
    print("\n3. Interpreting RSI signal...")
    result = rsi.interpret(current_rsi)

    print(f"   Signal: {result['signal']}")
    print(f"   Strength: {result['strength']}")
    print(f"   Confidence: {result['confidence']}%")
    print(f"   Reasoning: {result['reasoning']}")

    # Show recent RSI values
    print("\n4. Recent RSI values:")
    for i in range(-5, 0):
        print(f"   {i} days ago: {rsi_values.iloc[i]:.2f}")

    print("\n" + "=" * 60)
    print("✅ RSI TEST COMPLETE!")
    print("=" * 60)