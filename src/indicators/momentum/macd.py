from typing import Dict, Any, Optional
import pandas as pd

class MACD:
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.name = "MACD"

    def analyze(self, data: Dict[str, Any])-> Optional[Dict[str, Any]]:
        try:
            if isinstance(data.get('close'), pd.Series):
                close  = data['close']
            else:
                close = pd.Series(data['close'])
                
            min_required = self.slow_period + self.signal_period
            if len(close) < min_required:
                return None
            
            ema_fast = close.ewm(span = self.fast_period, adjust = False).mean()
            ema_slow = close.ewm(span = self.slow_period, adjust = False).mean()
            macd_line = ema_fast - ema_slow
            
            signal_line = macd_line.ewm(span = self.signal_period, adjust = False).mean()
            
            histogram = macd_line - signal_line

            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_histogram =  histogram.iloc[-1]
            previous_histogram = histogram.iloc[-2]
            
            if previous_histogram < 0 and current_histogram < 0:
                signal_type = "SELL"
                strength = self._calculate_strength(abs(current_histogram))
                confidence = min(95, 60 + abs(current_histogram) * 10)
                reasoning = f"MACD crossed above signal line (histogram: {current_histogram:.4f})"
                
            elif previous_histogram > 0 and current_histogram < 0:
                signal_type = 'SELL'
                strength = self._calculate_strength(abs(current_histogram))
                confidence = min(95, 60 + abs(current_histogram) * 10)
                reasoning = f"MACD crossed below signal line (histogram: {current_histogram:.4f})"
                
            else:
                signal_type = "HOLD"
                strength = "NEUTRAL"
                confidence = 50
                
                if current_histogram > 0:
                    reasoning = f"MACD above signal (bullish trend, histogram: {current_histogram:.4f})"
                else:
                    reasoning = f"MACD below signal (bearish trend, histogram: {current_histogram:.4f})"
                    
            return {
                'indicator': self.name,
                'signal': signal_type,
                'strength': strength,
                'confidence': int(confidence),
                'reasoning': reasoning,
                'values': {
                    'macd': float(current_macd),
                    'signal': float(current_signal),
                    'histogram': float(current_histogram)
                }
            }
            
        except Exception as e:
            print(f"Error in MACD analysis: {e}")
            return None
    
    
    def _calculate_strength(self, histogram_value: float) -> str:
        """
        Calculate signal strength based on histogram magnitude
        
        Args:
            histogram_value: Absolute value of histogram
            
        Returns:
            Strength string
        """
        if histogram_value > 2.0:
            return "STRONG"
        elif histogram_value > 0.5:
            return "MODERATE"
        else:
            return "WEAK"


if __name__ == "__main__":
    # Test the indicator
    print("MACD Indicator Test")
    print("=" * 50)
    
    # Create sample data
    import numpy as np
    sample_data = {
        'close': list(np.random.uniform(95, 105, 100))
    }
    
    indicator = MACD()
    result = indicator.analyze(sample_data)
    
    if result:
        print(f"Signal: {result['signal']}")
        print(f"Strength: {result['strength']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Reasoning: {result['reasoning']}")
        print(f"MACD: {result['values']['macd']:.4f}")
        print(f"Signal Line: {result['values']['signal']:.4f}")
        print(f"Histogram: {result['values']['histogram']:.4f}")
        print("\n✅ MACD indicator working!")
    else:
        print(" No signal generated")                
                