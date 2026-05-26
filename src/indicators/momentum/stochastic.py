from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class Stochastic:
    
    def __init__(self, k_period: int = 14, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        self.name = "STOCHASTIC"
    
    
    def analyze(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            if isinstance(data.get('high'), pd.Series):
                high = data['high']
                low = data['low']
                close = data['close']
            else:
                high = pd.Series(data['high'])
                low = pd.Series(data['low'])
                close = pd.Series(data['close'])
            
            minimum_required = self.k_period + self.d_period
            if len(close) < minimum_required:
                return None
            
            lowest_low = low.rolling(window=self.k_period).min()
            highest_high = high.rolling(window=self.k_period).max()
            
            numerator = close - lowest_low
            denominator = highest_high - lowest_low
            denominator = denominator.replace(0, 0.0001)
            
            k_percent = 100 * (numerator / denominator)
            d_percent = k_percent.rolling(window=self.d_period).mean()
            
            if len(k_percent) < 2 or len(d_percent) < 2:
                return None
            
            current_k = k_percent.iloc[-1]
            current_d = d_percent.iloc[-1]
            previous_k = k_percent.iloc[-2]
            previous_d = d_percent.iloc[-2]
            
            bullish_crossover = (previous_k < previous_d) and (current_k > current_d)
            bearish_crossover = (previous_k > previous_d) and (current_k < current_d)
            
            if current_k < 20 and current_d < 20:
                if bullish_crossover:
                    signal_type = "BUY"
                    strength = "STRONG"
                    confidence = min(95, 75 + (20 - current_k))
                    reasoning = f"Bullish crossover in oversold zone (K={current_k:.1f}, D={current_d:.1f})"
                else:
                    signal_type = "BUY"
                    strength = "MODERATE"
                    confidence = min(85, 65 + (20 - current_k))
                    reasoning = f"Oversold (K={current_k:.1f}, D={current_d:.1f})"
            
            elif current_k > 80 and current_d > 80:
                if bearish_crossover:
                    signal_type = "SELL"
                    strength = "STRONG"
                    confidence = min(95, 75 + (current_k - 80))
                    reasoning = f"Bearish crossover in overbought zone (K={current_k:.1f}, D={current_d:.1f})"
                else:
                    signal_type = "SELL"
                    strength = "MODERATE"
                    confidence = min(85, 65 + (current_k - 80))
                    reasoning = f"Overbought (K={current_k:.1f}, D={current_d:.1f})"
            
            else:
                signal_type = "HOLD"
                strength = "NEUTRAL"
                confidence = 50
                reasoning = f"Neutral zone (K={current_k:.1f}, D={current_d:.1f})"
            
            return {
                'indicator': self.name,
                'signal': signal_type,
                'strength': strength,
                'confidence': int(confidence),
                'reasoning': reasoning,
                'values': {
                    'k_percent': float(current_k),
                    'd_percent': float(current_d)
                }
            }
            
        except Exception as e:
            print(f"Error in Stochastic analysis: {e}")
            return None


if __name__ == "__main__":
    print("Stochastic Oscillator Indicator Test")
    print("=" * 50)
    
    import numpy as np
    np.random.seed(42)
    
    base_prices = np.linspace(100, 90, 50)
    noise = np.random.uniform(-2, 2, 50)
    close_prices = base_prices + noise
    high_prices = close_prices + np.random.uniform(0.5, 2, 50)
    low_prices = close_prices - np.random.uniform(0.5, 2, 50)
    
    sample_data = {
        'high': list(high_prices),
        'low': list(low_prices),
        'close': list(close_prices)
    }
    
    indicator = Stochastic()
    result = indicator.analyze(sample_data)
    
    if result:
        print(f"Signal: {result['signal']}")
        print(f"Strength: {result['strength']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Reasoning: {result['reasoning']}")
        print(f"%K: {result['values']['k_percent']:.2f}")
        print(f"%D: {result['values']['d_percent']:.2f}")
        print("\n✅ Stochastic indicator working!")
    else:
        print("❌ No signal generated")
            
                      
            
                    
                    
                            
                    
            