import numpy as np
from ...indicators.momentum.rsi import RSI
from ...signals import create_signal
from ...strategies.base import BaseStrategy


class RSIExtremeStrategy(BaseStrategy):
    def __init__(self, period: int = 14, overbought: float = 70.0, oversold: float = 30.0):
        super().__init__("RSI Extremes")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.rsi = RSI()
        
    def calculate_signal(self, prices: np.ndarray, **kwargs):
        if len(prices) < self.period + 1:
            return create_signal(
                indicator="RSI Extremes",
                signal="HOLD",
                strength="NEUTRAL",
                confidence=0,
                reasoning="Insufficient data for RSI",
                value=prices[-1]
            )
        
        try:
            rsi_values = self.rsi.calculate(prices, self.period)
            rsi_now = rsi_values[-1]
        except:
            return create_signal(
                indicator="RSI Extremes",
                signal="HOLD",
                strength="NEUTRAL",
                confidence=0,
                reasoning="RSI calculation failed",
                value=prices[-1]
            )
        
        if rsi_now > self.overbought:
            strength_val = rsi_now - self.overbought
            
            if strength_val > 15:
                strength = "STRONG"
                confidence = min(int(strength_val * 3), 95)
            elif strength_val > 5:
                strength = "MODERATE"
                confidence = min(int(strength_val * 4), 85)
            else:
                strength = "WEAK"
                confidence = min(int(strength_val * 5), 70)
            
            return create_signal(
                indicator="RSI Extremes",
                signal="SELL",
                strength=strength,
                confidence=confidence,
                reasoning=f"RSI overbought at {rsi_now:.1f}",
                value=prices[-1]
            )
        
        elif rsi_now < self.oversold:
            strength_val = self.oversold - rsi_now
            
            if strength_val > 15:
                strength = "STRONG"
                confidence = min(int(strength_val * 3), 95)
            elif strength_val > 5:
                strength = "MODERATE"
                confidence = min(int(strength_val * 4), 85)
            else:
                strength = "WEAK"
                confidence = min(int(strength_val * 5), 70)
            
            return create_signal(
                indicator="RSI Extremes",
                signal="BUY",
                strength=strength,
                confidence=confidence,
                reasoning=f"RSI oversold at {rsi_now:.1f}",
                value=prices[-1]
            )
        
        else:
            return create_signal(
                indicator="RSI Extremes",
                signal="HOLD",
                strength="WEAK",
                confidence=50,
                reasoning=f"RSI neutral at {rsi_now:.1f}",
                value=prices[-1]
            )
