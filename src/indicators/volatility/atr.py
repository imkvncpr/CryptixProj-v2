"""
ATR (Average True Range) Indicator
Measures volatility for risk management
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from typing import Dict, Any

from src.indicators.base_indicator import BaseIndicator


class ATR(BaseIndicator):
    """
    Average True Range (ATR)
    
    Measures market volatility. Used for risk management
    and stop-loss placement.
    """
    
    def __init__(self, period: int = 14):
        """Initialize ATR indicator"""
        super().__init__(name='ATR')
        self.period = period
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ATR values"""
        self.validate_data(data)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()      # ✅ FIXED: Use .abs()
        tr3 = (low - prev_close).abs()       # ✅ FIXED: Use .abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = true_range.rolling(window=self.period).mean()
        
        self.values = atr
        return atr
    
    def interpret(self, current_value: float, current_price: float = None) -> Dict[str, Any]:
        """Interpret ATR and assess volatility"""
        if current_price is None: 
            raise ValueError("current_price is required for ATR interpretation")
        
        atr_pct = (current_value / current_price) * 100
        
        if atr_pct > 5:
            volatility_level = "HIGH"
            description = "High volatility - large price swings"
        elif atr_pct > 2:
            volatility_level = "MEDIUM"
            description = "Medium volatility - normal movements"  # ✅ FIXED
        else:
            volatility_level = "LOW"
            description = "Low volatility - stable price"
            
        stop_distance = current_value * 2
        
        return {
            'indicator': 'ATR',
            'value': round(current_value, 2),
            'current_price': round(current_price, 2),
            'atr_pct': round(atr_pct, 2),
            'volatility': volatility_level,
            'description': description,
            'signal': 'NEUTRAL',
            'stop_loss_distance': round(stop_distance, 2),
            'reasoning': f"ATR at {atr_pct:.1f}% of price - {description}"
        }


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING ATR INDICATOR")
    print("=" * 60)
    
    from src.models.cryptos.bitcoin_predictor import BitcoinPredictor
    
    # Download Bitcoin data
    print("\n1. Downloading Bitcoin data...")
    btc = BitcoinPredictor()
    data = btc.download_data(period="3mo")
    current_price = data['close'].iloc[-1]
    
    print(f"   Current Bitcoin Price: ${current_price:,.2f}")
    print(f"   Data points: {len(data)}")
    
    # Calculate ATR
    print("\n2. Calculating ATR(14)...")
    atr = ATR(period=14)
    atr_values = atr.calculate(data)
    current_atr = atr.get_latest()
    
    print(f"   Current ATR: ${current_atr:,.2f}")
    
    # Interpret ATR
    print("\n3. Interpreting volatility...")
    result = atr.interpret(current_atr, current_price)
    
    print(f"   ATR Value: ${result['value']:,.2f}")
    print(f"   ATR Percentage: {result['atr_pct']:.2f}%")
    print(f"   Volatility Level: {result['volatility']}")
    print(f"   Description: {result['description']}")
    print(f"   Signal: {result['signal']}")
    print(f"   Stop-Loss Distance: ${result['stop_loss_distance']:,.2f}")
    print(f"   Reasoning: {result['reasoning']}")
    
    # Show recent ATR values
    print("\n4. Recent ATR values:")
    for i in range(-5, 0):
        print(f"   {i} days ago: ${atr_values.iloc[i]:,.2f}")
    
    # Risk management example
    print("\n5. Risk Management Example:")
    print(f"   If buying at: ${current_price:,.2f}")
    print(f"   Stop-loss at: ${current_price - result['stop_loss_distance']:,.2f}")
    print(f"   Risk per share: ${result['stop_loss_distance']:,.2f}")
    
    # Test different periods
    print("\n6. Testing different periods...")
    for period in [7, 14, 21]:
        test_atr = ATR(period=period)
        test_atr.calculate(data)
        test_val = test_atr.get_latest()
        print(f"   ATR({period}): ${test_val:,.2f}")
    
    print("\n" + "=" * 60)
    print("✅ ATR TEST COMPLETE!")
    print("=" * 60)