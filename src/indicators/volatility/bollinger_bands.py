from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class BollingerBands:

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        self.name = "BOLLINGER"

    def analyze(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:

        try:

            if isinstance(data.get('close'), pd.Series):
                close = data['close']
            else:
                close = pd.Series(data['close'])

            if len(close) < self.period:
                return None

            middle_band = close.rolling(window=self.period).mean()
            std = close.rolling(window=self.period).std()

            upper_band = middle_band + (self.std_dev * std)
            lower_band = middle_band - (self.std_dev * std)

            current_price = close.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_middle = middle_band.iloc[-1]
            current_lower = lower_band.iloc[-1]

            band_width = (
                (current_upper - current_lower)
                / current_middle
            ) * 100

            position = (
                (current_price - current_lower)
                / (current_upper - current_lower)
            ) * 100

            if position <= 20:

                signal_type = "BUY"
                strength = "STRONG" if position <= 10 else "MODERATE"

                confidence = min(95, 70 + (20 - position))

                reasoning = (
                    f"Price near lower band "
                    f"({position:.1f}% position, oversold)"
                )

            elif position >= 80:

                signal_type = "SELL"
                strength = "STRONG" if position >= 90 else "MODERATE"

                confidence = min(95, 70 + (position - 80))

                reasoning = (
                    f"Price near upper band "
                    f"({position:.1f}% position, overbought)"
                )

            else:

                signal_type = "HOLD"
                strength = "NEUTRAL"
                confidence = 50

                reasoning = (
                    f"Price in middle zone "
                    f"({position:.1f}% position)"
                )

            return {
                'indicator': self.name,
                'signal': signal_type,
                'strength': strength,
                'confidence': int(confidence),
                'reasoning': reasoning,
                'values': {
                    'price': float(current_price),
                    'upper_band': float(current_upper),
                    'middle_band': float(current_middle),
                    'lower_band': float(current_lower),
                    'position': float(position),
                    'band_width': float(band_width)
                }
            }

        except Exception as e:
            print(f"Error in Bollinger Bands analysis: {e}")
            return None


if __name__ == "__main__":

    print("Bollinger Bands Indicator Test")
    print("=" * 50)

    np.random.seed(42)

    sample_data = {
        'close': list(np.random.uniform(95, 105, 100))
    }

    indicator = BollingerBands()

    result = indicator.analyze(sample_data)

    if result:

        print(f"Signal: {result['signal']}")
        print(f"Strength: {result['strength']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Price: ${result['values']['price']:.2f}")
        print(f"Upper Band: ${result['values']['upper_band']:.2f}")
        print(f"Middle Band: ${result['values']['middle_band']:.2f}")
        print(f"Lower Band: ${result['values']['lower_band']:.2f}")
        print(f"Position: {result['values']['position']:.1f}%")

        print("\n✅ Bollinger Bands indicator working!")

    else:
        print("❌ No signal generated")
            
            
     