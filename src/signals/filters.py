import numpy as np
from typing import Optional

class SignalFilter:
    def __init__(self):
        pass
    
    @staticmethod
    def filter_by_strength(signal, min_strength: float = 50.0)-> bool:
        
        strength_map = {
            "STRONG": 80,
            "MODERATE": 60,
            "WEAK": 40,
            "NEUTRAL": 0
        }
        
        strength_val = strength_map.get(str(signal.strength), 0)
        return strength_val >= min_strength
    
    @staticmethod
    def filter_by_confidence(signal, min_confidence: int = 60)-> bool:
        confidence_str = str(signal.confidence).replace("%", "")
        try:
            confidence_val = int(float(confidence_str))
        except:
            confidence_val = 0
        
        return confidence_val >= min_confidence
    
    @staticmethod
    def filter_by_trend(signal, prices: np.ndarray, trend_period: int = 200)-> bool:
        if len(prices) < trend_period:
            return True
        
        try:
            sma_slow = np.mean(prices[-trend_period:])
            price_now = prices[-1]
            
            if signal.signal_type == 'BUY':
                return price_now > sma_slow
            
            elif signal.signal_type == 'SELL':
                return price_now < sma_slow 
            
            else:
                return True
            
        except:
            return True
        
    @staticmethod
    def apply_filters(
        signal,
        prices: np.ndarray,
        min_strength: float = 50.0,
        min_confidence: int = 60,
        use_trend: bool = False,
        trend_period: int = 200
    ) -> bool:
        
        if not SignalFilter.filter_by_strength(signal, min_strength):
            return False
        
        if not SignalFilter.filter_by_confidence(signal, min_confidence):
            return False
        
        if use_trend:
            if not SignalFilter.filter_by_trend(signal, prices, trend_period):
                return False
        
        return True
