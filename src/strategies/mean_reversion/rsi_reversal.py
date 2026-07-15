import numpy as np
from ...indicators.momentum.rsi import RSI
from ...signals import create_signal
from ...strategies.base import BaseStrategy

class RSIReversalStrategy(BaseStrategy):
    def __init__(self, period: int = 14):
        super().__init__("RSI Reversal")
        self.period = period
        self.rsi = RSI()
        
    def calculate_signal(self, prices: np.ndarray, **kwargs):
        if len(prices) < self.period + 2:
            return create_signal(
                indicator="RSI Reversal",
                signal="HOLD",
                strength="NEUTRAL",
                confidence=0,
                reasoning="Insufficient data",
                value=prices[-1]
            )
            
        try:
            rsi_values = self.rsi.calculate(prices, self.period)
        except:
            return create_signal(
                indicator="RSI Reversal",
                signal="HOLD",
                strength="NEUTRAL",
                confidence=0,
                reasoning="RSI calculation failed",
                value=prices[-1]
            )
        price_now = prices[-1]
        price_previous = prices[-2]
        rsi_now = rsi_values[-1]
        rsi_previous = rsi_values[-2]
        
        price_down = price_now < price_previous
        price_up = price_now > price_previous
        rsi_up = rsi_now < rsi_previous
        rsi_down = rsi_now > rsi_previous
        
        if price_down and rsi_up and rsi_now < 40:
            return create_signal(
                indicator="RSI Reversal",
                signal="BUY",
                strength="STRONG",
                confidence=85,
                reasoning="Bullish divergence detected",
                value=price_now
            )
            
        elif price_up and rsi_down and rsi_now > 60:
            return create_signal(
                indicator="RSI Reversal",
                signal="SELL",
                strength="STRONG",
                confidence=85,
                reasoning="Bearish divergence detected",
                value=price_now
            )
        
        else:
            return create_signal(
                indicator="RSI Reversal",
                signal="HOLD",
                strength="WEAK",
                confidence=50,
                reasoning="No divergence detected",
                value=price_now
            )