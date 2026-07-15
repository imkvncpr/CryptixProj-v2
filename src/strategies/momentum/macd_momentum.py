import numpy as np
from ...indicators.momentum.macd import MACD
from ...signals import create_signal
from ...strategies.base import BaseStrategy

class MACDMomentumStrategy(BaseStrategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACD Momentum")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.macd = MACD()
        
    def calculate_signal(self, prices: np.ndarray, **kwargs):
        if len(prices) < self.slow + self.signal_period:
            return create_signal(
                indicator="MACD Momentum",
                signal="HOLD",
                strength="NEUTRAL",
                confidence=0,
                reasoning="Insufficient data for MACD",
                value=prices[-1]
            )
            
        try:
            macd_line, signal_line, histogram = self.macd.calculate(
                prices, self.fast, self.slow, self.signal_period
            )
        except:
            return create_signal(
                indicator="MACD Momentum",
                signal="HOLD",
                strength="NEUTRAL",
                confidence=0,
                reasoning="MACD calculation failed",
                value=prices[-1]
            )
            
        macd_now = macd_line[-1]
        macd_previous = macd_line[-2]
        signal_now = signal_line[-1]
        signal_previous = signal_line[-2]
        hist_now = histogram[-1]
        
        hist_abs = abs(hist_now)
        
        if hist_abs > 0.5:
            strength = "STRONG"
            confidence = min(int(hist_abs * 30), 95)
        elif hist_abs > 0.2:
            strength = "MODERATE"
            confidence = min(int(hist_abs * 50), 85)
        else:
            strength = "WEAK"
            confidence = min(int(hist_abs * 100), 70)
            
        if macd_previous <= signal_previous and macd_now > signal_now:
            return create_signal(
                indicator="MACD Momentum",
                signal="SELL",
                strength=strength,
                confidence=confidence,
                reasoning="MACD crossed below signal line",
                value=prices[-1]
            )
        
        
        else:
            return create_signal(
                indicator="MACD Momentum",
                signal="HOLD",
                strength="WEAK",
                confidence=50,
                reasoning="No MACD crossover",
                value=prices[-1]
            )
        