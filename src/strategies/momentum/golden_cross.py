import numpy as np
from ...indicators.trend.moving_averages import MovingAverage
from ...signals import create_signal
from ...strategies.base import BaseStrategy


class GoldenCrossStrategy(BaseStrategy):
    def __init__(self, fast: int = 50, slow: int = 200):
        super().__init__("Golden Cross")
        self.fast = fast
        self.slow = slow
        self.ma = MovingAverage()
        
    def calculate_signal(self, prices: np.ndarray, **kwargs):
        if len(prices) < self.slow + 1:
            return create_signal(
                indicator="Golden Cross",
                signal="HOLD",
                strength="NEUTRAL",
                confidence=0,
                reasoning="Insufficient data",
                value=prices[-1]
            )
        
        try:
            sma_fast = self.ma.calculate_sma(prices, self.fast)
            sma_slow = self.ma.calculate_sma(prices, self.slow)
        except:
            return create_signal(
                indicator="Golden Cross",
                signal="HOLD",
                strength="NEUTRAL",
                confidence=0,
                reasoning="SMA failed",
                value=prices[-1]
            )
        
        fast_now, fast_prev = sma_fast[-1], sma_fast[-2]
        slow_now, slow_prev = sma_slow[-1], sma_slow[-2]
        
        separation_pct = abs((fast_now - slow_now) / slow_now * 100)
        
        if separation_pct > 3.0:
            strength = "STRONG"
            confidence = min(int(separation_pct * 5), 95)
        elif separation_pct > 1.0:
            strength = "MODERATE"
            confidence = min(int(separation_pct * 10), 80)
        else:
            strength = "WEAK"
            confidence = min(int(separation_pct * 20), 60)
        
        if fast_prev <= slow_prev and fast_now > slow_now:
            return create_signal(
                indicator="Golden Cross",
                signal="BUY",
                strength=strength,
                confidence=confidence,
                reasoning="Golden Cross",
                value=prices[-1]
            )
        
        elif fast_prev >= slow_prev and fast_now < slow_now:
            return create_signal(
                indicator="Golden Cross",
                signal="SELL",
                strength=strength,
                confidence=confidence,
                reasoning="Death Cross",
                value=prices[-1]
            )
        
        else:
            return create_signal(
                indicator="Golden Cross",
                signal="HOLD",
                strength="WEAK",
                confidence=50,
                reasoning="No crossover",
                value=prices[-1]
            )
