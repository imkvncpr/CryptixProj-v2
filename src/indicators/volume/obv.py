
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from typing import Dict, Any

from src.indicators.base_indicator import BaseIndicator


class OBV(BaseIndicator):
    def __init__(self):
        super().__init__(name='OBV')  # ✅ FIXED: Uppercase 'V'
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate OBV values"""
        self.validate_data(data)  # ✅ FIXED: Added validation
        
        close = data['close']
        volume = data['volume']
        
        prev_close = close.shift(1)
        
        signed_volume = volume.copy()
        
        signed_volume[close > prev_close] = volume
        signed_volume[close < prev_close] = -volume
        signed_volume[close == prev_close] = 0
        
        obv = signed_volume.cumsum()
        
        self.values = obv
        return obv
    
    def interpret(self, current_value: float, current_price: float = None, 
                  prev_value: float = None, prev_price: float = None) -> Dict[str, Any]:
        """Interpret OBV by comparing trends"""
        
        if current_price is None:
            raise ValueError("current_price is required")
        
        if prev_value is None or prev_price is None:
            return {
                'indicator': 'OBV',
                'value': round(current_value, 2),
                'current_price': round(current_price, 2),
                'signal': 'HOLD',
                'strength': 'NEUTRAL',
                'confidence': 50,
                'reasoning': 'Insufficient data - need previous values for trend analysis'
            }
            
        # Determine price trend
        if current_price > prev_price:
            price_trend = "UP"
        elif current_price < prev_price:
            price_trend = "DOWN"
        else:
            price_trend = "FLAT"
            
        # Determine OBV trend
        if current_value > prev_value:
            obv_trend = "UP"
        elif current_value < prev_value:
            obv_trend = "DOWN"
        else:
            obv_trend = "FLAT"
            
        # Interpret combinations
        if price_trend == "UP" and obv_trend == "UP":
            signal, strength, confidence = "BUY", "STRONG", 90
            reasoning = "Confirmed uptrend - volume confirms price rise"
            
        elif price_trend == "DOWN" and obv_trend == "DOWN":
            signal, strength, confidence = "SELL", "STRONG", 90  # ✅ FIXED: Added 90
            reasoning = "Confirmed downtrend - volume confirms price fall"  # ✅ FIXED: Correct reasoning
            
        elif price_trend == "DOWN" and obv_trend == "UP":  # ✅ FIXED: Added missing case!
            signal, strength, confidence = "BUY", "MODERATE", 70
            reasoning = "Bullish divergence - accumulation despite falling price"
            
        elif price_trend == "UP" and obv_trend == "DOWN":
            signal, strength, confidence = "SELL", "MODERATE", 70
            reasoning = "Bearish divergence - distribution despite rising price"
            
        else:
            signal, strength, confidence = "HOLD", "NEUTRAL", 50
            reasoning = "No clear trend - price and volume both flat"
            
        return {
            'indicator': 'OBV',
            'value': round(current_value, 2),
            'current_price': round(current_price, 2),
            'prev_value': round(prev_value, 2),
            'prev_price': round(prev_price, 2),
            'price_trend': price_trend,
            'obv_trend': obv_trend,
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'reasoning': reasoning
        }


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING OBV INDICATOR")
    print("=" * 60)
    
    from src.models.cryptos.bitcoin_predictor import BitcoinPredictor
    
    print("\n1. Downloading Bitcoin data...")
    btc = BitcoinPredictor()
    data = btc.download_data(period="3mo")
    
    current_price = data['close'].iloc[-1]
    prev_price = data['close'].iloc[-2]
    
    print(f"   Current Bitcoin Price: ${current_price:,.2f}")
    print(f"   Previous Price: ${prev_price:,.2f}")
    print(f"   Data points: {len(data)}")
    
    print("\n2. Calculating OBV...")
    obv = OBV()
    obv_values = obv.calculate(data)
    
    current_obv = obv.get_latest()
    prev_obv = obv_values.iloc[-2]
    
    print(f"   Current OBV: {current_obv:,.0f}")
    print(f"   Previous OBV: {prev_obv:,.0f}")
    
    print("\n3. Interpreting OBV...")
    result = obv.interpret(current_obv, current_price, prev_obv, prev_price)
    
    print(f"   Current OBV: {result['value']:,.0f}")
    print(f"   Price Trend: {result['price_trend']}")
    print(f"   OBV Trend: {result['obv_trend']}")
    print(f"   Signal: {result['signal']} ({result['strength']})")
    print(f"   Confidence: {result['confidence']}%")
    print(f"   Reasoning: {result['reasoning']}")
    
    print("\n4. Recent OBV values:")
    for i in range(-5, 0):
        price = data['close'].iloc[i]
        obv_val = obv_values.iloc[i]
        print(f"   {i} days ago: OBV={obv_val:,.0f}, Price=${price:,.2f}")
    
    print("\n5. Price vs OBV Trend (last 10 days):")
    print(f"   {'Day':<10} {'Price':<15} {'OBV':<20} {'Price Δ':<12} {'OBV Δ'}")
    for i in range(-10, 0):
        price = data['close'].iloc[i]
        obv_val = obv_values.iloc[i]
        
        if i > -10:
            price_change = price - data['close'].iloc[i-1]
            obv_change = obv_val - obv_values.iloc[i-1]
            price_dir = "↑" if price_change > 0 else ("↓" if price_change < 0 else "→")
            obv_dir = "↑" if obv_change > 0 else ("↓" if obv_change < 0 else "→")
        else:
            price_dir = "-"
            obv_dir = "-"
        
        print(f"   {i:<10} ${price:<14,.2f} {obv_val:<19,.0f} {price_dir:<12} {obv_dir}")
    
    print("\n" + "=" * 60)
    print("✅ OBV TEST COMPLETE!")
    print("=" * 60)
                
                
                
                
        
                    