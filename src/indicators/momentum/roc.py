import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from typing import Dict, Any

from src.indicators.base_indicator import BaseIndicator

class ROC(BaseIndicator):
    def __init__(self, period: int = 10):
        super().__init__(name = 'ROC')
        self.period = period
        
    def calculate(self, data: pd.DataFrame)-> pd.Series:
        self.validate_data(data)
        close = data['close']
        
        prev_close = close.shift(self.period)
        
        roc = ((close - prev_close) / prev_close) * 100
        
        self.values = roc 
        return roc
    
    def interpret(self, current_value: float, current_price: float = None)->Dict[str, Any]:
        if current_value > 5:
            signal, strength, confidence = "BUY", "STRONG", 85
            reasoning = f"Strong upward momentum({current_value: .1f}%)"
        
        elif current_value > 0:
            signal, strength, confidence = "BUY", "WEAK", 60
            reasoning = f"Weak upward momentum({current_value: .1f}%)"
            
        elif current_value > -5:
            signal, strength, confidence = "SELL", "STRONG", 85
            reasoning = f"Strong downward momentum({current_value: .1f})%"
            
        elif current_value < 0:
            signal, strength, confidence = "SELL", "WEAK", 60
            reasoning = f"Weak downward momentum({current_value: .1f}%)" 
            
        else:
            signal, strength, confidence = "HOLD", "NEUTRAL", 50
            reasoning = "No momentum - Flat price"
            
        result = {
            'indicator': 'ROC',
            'value': round(current_value, 2),
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'reasoning': reasoning
        }
        
        if current_price is not None:
            result['current_price'] = round(current_price, 2)
        
        return result
    
    # ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING ROC INDICATOR")
    print("=" * 60)
    
    from src.models.cryptos.bitcoin_predictor import BitcoinPredictor
    
    print("\n1. Downloading Bitcoin data...")
    btc = BitcoinPredictor()
    data = btc.download_data(period="3mo")
    current_price = data['close'].iloc[-1]
    
    print(f"   Current Bitcoin Price: ${current_price:,.2f}")
    print(f"   Data points: {len(data)}")
    
    print("\n2. Calculating ROC(10)...")
    roc = ROC(period=10)
    roc_values = roc.calculate(data)
    current_roc = roc.get_latest()
    
    print(f"   Current ROC: {current_roc:.2f}%")
    
    print("\n3. Interpreting ROC...")
    result = roc.interpret(current_roc, current_price)
    
    print(f"   ROC Value: {result['value']:.2f}%")
    print(f"   Signal: {result['signal']} ({result['strength']})")
    print(f"   Confidence: {result['confidence']}%")
    print(f"   Reasoning: {result['reasoning']}")
    
    print("\n4. Recent ROC values:")
    for i in range(-5, 0):
        roc_val = roc_values.iloc[i]
        price = data['close'].iloc[i]
        direction = "↑" if roc_val > 0 else ("↓" if roc_val < 0 else "→")
        print(f"   {i} days ago: ROC={roc_val:+.2f}%, Price=${price:,.2f} {direction}")
    
    print("\n5. Testing different periods...")
    for period in [5, 10, 20]:
        test_roc = ROC(period=period)
        test_roc.calculate(data)
        test_val = test_roc.get_latest()
        print(f"   ROC({period}): {test_val:+.2f}%")
    
    print("\n6. Period comparison:")
    print(f"   {'Period':<10} {'ROC %':<12} {'Signal'}")
    for period in [5, 10, 20]:
        test_roc = ROC(period=period)
        test_roc.calculate(data)
        test_val = test_roc.get_latest()
        test_result = test_roc.interpret(test_val)
        print(f"   {period:<10} {test_val:+.2f}%{' ':<8} {test_result['signal']} ({test_result['strength']})")
    
    print("\n" + "=" * 60)
    print("✅ ROC TEST COMPLETE!")
    print("=" * 60)
    