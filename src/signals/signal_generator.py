import logging
from typing import List, Dict, Optional, Any


from src.signals.signal_types import Signal, SignalType, Strength, Confidence, create_signal
from ..indicators.momentum.rsi import RSI   
from ..indicators.momentum.macd import MACD
from ..indicators.volatility.bollinger_bands import BollingerBands
from ..indicators.momentum.stochastic import Stochastic
from ..indicators.trend.moving_averages import MovingAverage
from ..indicators.volatility.atr import ATR
from ..indicators.volume.obv import OBV
from ..indicators.momentum.roc import ROC

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generates unified trading signals from multiple indicators
    """
    
    def __init__(self):
        """Initialize all technical indicators"""
        self.indicators = {
            'RSI': RSI(),
            'MACD': MACD(),
            'BOLLINGER': BollingerBands(),
            'STOCHASTIC': Stochastic(),
            'MA': MovingAverage(),
            'ATR': ATR(),
            'OBV': OBV(),
            'ROC': ROC()
        }
        
        logger.info(f"SignalGenerator initialized with {len(self.indicators)} indicators")
    
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals from all indicators"""
        logger.info(f"Generating signals from {len(self.indicators)} indicators")
        
        # Collect signals from all indicators
        individual_signals = []
        
        for indicator_name, indicator in self.indicators.items():
            signal = self._run_indicator(indicator_name, indicator, data)
            if signal is not None:
                individual_signals.append(signal)
        
        # THIS IS OUTSIDE THE FOR LOOP!
        logger.info(f"Collected {len(individual_signals)} signals")
        
        # Check if we got any signals
        if not individual_signals:
            logger.warning("No signals generated, returning HOLD")
            return {
                'final_signal': create_signal(
                    indicator="AGGREGATE",
                    signal="HOLD",
                    strength="NEUTRAL",
                    confidence=50,
                    reasoning="No indicators generated signals"
                ),
                'individual_signals': [],
                'summary': {
                    'total_indicators': len(self.indicators),
                    'signals_generated': 0,
                    'buy_count': 0,
                    'sell_count': 0,
                    'hold_count': 1
                }
            }
        
        # Aggregate signals
        final_signal = self.aggregate_signals(individual_signals)
        
        # Create summary
        summary = {
            'total_indicators': len(self.indicators),
            'signals_generated': len(individual_signals),
            'buy_count': sum(1 for s in individual_signals if s.signal_type == SignalType.BUY),
            'sell_count': sum(1 for s in individual_signals if s.signal_type == SignalType.SELL),
            'hold_count': sum(1 for s in individual_signals if s.signal_type == SignalType.HOLD),
            'average_confidence': sum(int(s.confidence) for s in individual_signals) / len(individual_signals),
            'aggregate_score': final_signal.score()
        }
        
        return {
            'final_signal': final_signal,
            'individual_signals': individual_signals,
            'summary': summary
        }
    
    
    def _run_indicator(
        self,
        name: str,
        indicator,
        data: Dict[str, Any]
    ) -> Optional[Signal]:
        """Safely run a single indicator"""
        try:
            logger.debug(f"Running {name} indicator")
            
            # Run indicator
            result = indicator.analyze(data)
            
            if result is None:
                logger.warning(f"{name} returned None")
                return None
            
            # Convert to Signal object
            signal = Signal.from_dict(result)
            
            logger.info(f"✅ {name}: {signal.signal_type} ({signal.strength}, {signal.confidence})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
            return None
    
    
    def aggregate_signals(self, signals: List[Signal]) -> Signal:
        """Aggregate multiple signals into one unified signal"""
        aggregate_score = self._calculate_aggregate_score(signals)
        signal_type = self._determine_signal_type(aggregate_score)
        strength = self._calculate_combined_strength(signals, signal_type)
        confidence = self._calculate_combined_confidence(signals)
        
        buy_count = sum(1 for s in signals if s.signal_type == SignalType.BUY)
        sell_count = sum(1 for s in signals if s.signal_type == SignalType.SELL)
        hold_count = sum(1 for s in signals if s.signal_type == SignalType.HOLD)
        
        reasoning = f"Aggregated from {len(signals)} indicators: "
        reasoning += f"{buy_count} BUY, {sell_count} SELL, {hold_count} HOLD. "
        reasoning += f"Score: {aggregate_score:.2f}"
        
        return create_signal(
            indicator="AGGREGATE",
            signal=signal_type.value,
            strength=strength.value,
            confidence=int(confidence.value),
            reasoning=reasoning,
            metadata={
                'num_signals': len(signals),
                'aggregate_score': aggregate_score,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'hold_count': hold_count
            }
        )
    
    
    def _calculate_aggregate_score(self, signals: List[Signal]) -> float:
        """Calculate weighted aggregate score"""
        return sum(signal.score() for signal in signals)
    
    
    def _determine_signal_type(self, aggregate_score: float) -> SignalType:
        """Determine signal type from aggregate score"""
        if aggregate_score > 0.5:
            return SignalType.BUY
        elif aggregate_score < -0.5:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    
    def _calculate_combined_strength(
        self,
        signals: List[Signal],
        signal_type: SignalType
    ) -> Strength:
        """Calculate combined strength"""
        matching_signals = [s for s in signals if s.signal_type == signal_type]
        
        if not matching_signals:
            return Strength.NEUTRAL
        
        avg_strength = sum(s.strength.to_score() for s in matching_signals) / len(matching_signals)
        return Strength.from_score(round(avg_strength))
    
    
    def _calculate_combined_confidence(self, signals: List[Signal]) -> Confidence:
        """Calculate weighted average confidence"""
        total_weighted = 0.0
        total_weight = 0.0
        
        for signal in signals:
            conf_value = int(signal.confidence)
            weight = signal.strength.to_score()
            total_weighted += conf_value * weight
            total_weight += weight
        
        if total_weight == 0:
            return Confidence(50)
        
        avg_confidence = total_weighted / total_weight
        return Confidence(int(avg_confidence))


# Test the class
if __name__ == "__main__":
    gen = SignalGenerator()
    print(f"✅ Initialized with {len(gen.indicators)} indicators")
    print("Indicators:", list(gen.indicators.keys()))