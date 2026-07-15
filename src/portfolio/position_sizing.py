import numpy as np
from typing import Dict, List, Optional


class PositionSizer:
    def __init__(self):
        pass

    @staticmethod
    def kelly_criterion(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        account_size: float,
        max_fraction: float = 0.25
    )-> float:
        
        if avg_win <= 0:
            return 0
        
        kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        kelly_f = max(0, min(kelly_f, max_fraction))
        
        Position_size = account_size * kelly_f
        return Position_size
    
    @staticmethod
    def fixed_fraction(
        account_size: float,
        fraction: float = 0.02
    ) -> float:
        fraction = max(0, min(fraction, 1.0))
        
        position_size = account_size * fraction
        return position_size
    
    @staticmethod
    def volatility_adjusted(
       account_size: float,
       atr_value: float,
       risk_pct: float = 1.0 
    )-> float:
        if atr_value <= 0:
            return 0
        
        risk_amount = account_size * (risk_pct / 100)
        
        position_size = risk_amount / atr_value
        
        max_position = account_size * 0.5
        position_size = min(position_size, max_position)
        
        return position_size
    
    @staticmethod
    def correlation_adjusted(
        position_sizes: Dict[str, float],
        correlations: Dict[tuple, float]
    )-> Dict[str, float]:
        adjusted_sizes = position_sizes.copy()
        symbols = list(position_sizes.keys())
        
        for symbol in symbols:
            correlations_with_others = []
            
            for (s1, s2), corr in correlations.items():
                if s1 == symbol:
                    correlations_with_others.append(corr)
                elif s2 == symbol:
                    correlations_with_others.append(corr)
                    
            if correlations_with_others:
                avg_correlation = np.mean(correlations_with_others)
                
                if avg_correlation > 0.7:
                    reduction_factor = 1 - avg_correlation
                    adjusted_sizes[symbol] = position_sizes[symbol] * reduction_factor
        
        return adjusted_sizes