import numpy as np
from ...indicators.volatility.bollinger_bands import BollingerBands
from ...signals import create_signal
from ...strategies.base import BaseStrategy

class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("Bollinger Bands")
        self.period = period
        self.std_dev = std_dev
        self.bb = BollingerBands()
        
    def calculate_signal(self, prices: np.ndarray, **kwargs):
        if len(prices) < self.period + 1:
            return create_signal(
                indicator="Bollinger Bands",
                signal="HOLD",
                strength="NEUTRAL",
                confidence=0,
                reasoning="Insufficient data",
                value=prices[-1]
            )
            
        try:
            upper, middle, lower = self.bb.calculate(prices, self.period, self.std_dev)
        except:
            return create_signal(
                indicator="Bollinger Bands",
                signal="HOLD",
                strength="NEUTRAL",
                confidence=0,
                reasoning="BB calculation failed",
                value=prices[-1]
            )
            
        price_now = prices[-1]
        upper_now = prices[-1]
        lower_now = prices[-1]
        band_width = upper_now - lower_now
        
        if price_now < lower_now:
            distance = (
        (lower_now - price_now) / band_width * 100
        if band_width > 0
        else 0)
            strength_val = min(distance + 50, 100)
            
            if strength_val > 75:
                strength = "STRONG"
                confidence = min(int(strength_val), 95)
            elif strength_val > 50:
                strength = "MODERATE"
                confidence = min(int(strength_val * 0.8), 85)
            else:
                strength = "WEAK"
                confidence = min(int(strength_val * 0.6), 70)
                
            return create_signal(
                indicator="Bollinger Bands",
                signal="BUY",
                strength=strength,
                confidence=confidence,
                reasoning="Price below lower band",
                value=price_now
            )
            
        elif price_now > upper_now:
            distance = (
        (price_now - upper_now) / band_width * 100
        if band_width > 0
        else 0)
            strength_val = min(distance + 50, 100)
            
            if strength_val > 75:
                strength = "STRONG"
                confidence = min(int(strength_val), 95)
            elif strength_val > 50:
                strength = "MODERATE"
                confidence = min(int(strength_val * 0.8), 85)
            else:
                strength = "WEAK"
                confidence = min(int(strength_val * 0.6), 70)
            
            return create_signal(
                indicator="Bollinger Bands",
                signal="SELL",
                strength=strength,
                confidence=confidence,
                reasoning="Price above upper band",
                value=price_now
            )
        
      
        else:
            return create_signal(
                indicator="Bollinger Bands",
                signal="HOLD",
                strength="WEAK",
                confidence=50,
                reasoning="Price within bands",
                value=price_now
            )
            
                