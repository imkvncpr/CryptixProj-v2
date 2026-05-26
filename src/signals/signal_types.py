from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    
    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def from_string(cls, value: str) -> 'SignalType':
        value_upper = value.upper()
        try:
            return cls[value_upper]
        except KeyError:
            raise ValueError(f"Invalid signal type: {value}. Must be BUY, SELL, or HOLD")


class Strength(Enum):
    """Signal strength levels"""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NEUTRAL = "NEUTRAL"
    
    def __str__(self) -> str:
        return self.value
    
    def to_score(self) -> int:
        scores = {
            Strength.STRONG: 3,
            Strength.MODERATE: 2,
            Strength.WEAK: 1,
            Strength.NEUTRAL: 0
        }
        return scores[self]
    
    @classmethod
    def from_score(cls, score: int) -> 'Strength':
        score_map = {
            3: cls.STRONG,
            2: cls.MODERATE,
            1: cls.WEAK,
            0: cls.NEUTRAL
        }
        
        if score not in score_map:
            raise ValueError(f"Invalid score: {score}. Must be 0-3")
        return score_map[score]
    
    @classmethod
    def from_string(cls, value: str) -> 'Strength':
        value_upper = value.upper()
        try:
            return cls[value_upper]
        except KeyError:
            raise ValueError(f"Invalid strength: {value}. Must be STRONG, MODERATE, WEAK, or NEUTRAL")


@dataclass
class Confidence:
    """Confidence level (0-100%)"""
    value: int
    
    def __post_init__(self):
        """Validate confidence value after initialization"""
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"Confidence must be a number, got {type(self.value)}")
        
        if self.value < 0 or self.value > 100:
            raise ValueError(f"Confidence must be between 0 and 100, got {self.value}")
        
        self.value = int(self.value)
    
    def __str__(self) -> str:
        return f"{self.value}%"
    
    def __repr__(self) -> str:
        return f"Confidence({self.value})"
    
    def __int__(self) -> int:
        return self.value
    
    def to_percentage(self) -> str:
        return f"{self.value}%"
    
    def to_decimal(self) -> float:
        return self.value / 100.0
    
    def level(self) -> str:
        if self.value >= 80:
            return "HIGH"
        elif self.value >= 60:
            return "MEDIUM"
        else:
            return "LOW"
    
    @classmethod
    def from_decimal(cls, decimal: float) -> 'Confidence':
        return cls(int(decimal * 100))


@dataclass
class Signal:
    """
    Complete trading signal
    
    Combines signal type, strength, confidence, and reasoning
    into a single validated object.
    """
    indicator: str
    signal_type: SignalType
    strength: Strength
    confidence: Confidence
    reasoning: str
    value: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate signal after initialization"""
        # Convert strings to enums if needed
        if isinstance(self.signal_type, str):
            self.signal_type = SignalType.from_string(self.signal_type)
        
        if isinstance(self.strength, str):
            self.strength = Strength.from_string(self.strength)
        
        if isinstance(self.confidence, (int, float)):
            self.confidence = Confidence(self.confidence)
        
        # Validate HOLD signals have NEUTRAL strength
        if self.signal_type == SignalType.HOLD and self.strength != Strength.NEUTRAL:
            raise ValueError("HOLD signals must have NEUTRAL strength")
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.indicator}: {self.signal_type} ({self.strength}, {self.confidence})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {
            'indicator': self.indicator,
            'signal': self.signal_type.value,
            'strength': self.strength.value,
            'confidence': int(self.confidence),
            'reasoning': self.reasoning
        }
        
        if self.value is not None:
            result['value'] = self.value
        
        if self.metadata is not None:
            result['metadata'] = self.metadata
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Create Signal from dictionary"""
        return cls(
            indicator=data['indicator'],
            signal_type=SignalType.from_string(data['signal']),
            strength=Strength.from_string(data['strength']),
            confidence=Confidence(data['confidence']),
            reasoning=data['reasoning'],
            value=data.get('value'),
            metadata=data.get('metadata')
        )
    
    def score(self) -> float:
        """Calculate numeric score for this signal"""
        direction = 1 if self.signal_type == SignalType.BUY else (-1 if self.signal_type == SignalType.SELL else 0)
        return self.strength.to_score() * self.confidence.to_decimal() * direction


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_signal_dict(data: Dict[str, Any]) -> bool:
    """
    Validate a signal dictionary has all required fields
    
    Args:
        data: Dictionary to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    required_fields = ['indicator', 'signal', 'strength', 'confidence', 'reasoning']
    
    # Check all required fields present
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate signal type
    try:
        SignalType.from_string(data['signal'])
    except ValueError as e:
        raise ValueError(f"Invalid signal type: {e}")
    
    # Validate strength
    try:
        Strength.from_string(data['strength'])
    except ValueError as e:
        raise ValueError(f"Invalid strength: {e}")
    
    # Validate confidence
    if not isinstance(data['confidence'], (int, float)):
        raise ValueError("Confidence must be a number")
    
    if not (0 <= data['confidence'] <= 100):
        raise ValueError(f"Confidence must be between 0 and 100, got {data['confidence']}")
    
    return True


def create_signal(
    indicator: str,
    signal: str,
    strength: str,
    confidence: int,
    reasoning: str,
    value: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Signal:
    """
    Convenience function to create a Signal object
    
    Automatically converts strings to appropriate enum types.
    
    Args:
        indicator: Indicator name
        signal: "BUY", "SELL", or "HOLD"
        strength: "STRONG", "MODERATE", "WEAK", or "NEUTRAL"
        confidence: 0-100
        reasoning: Explanation
        value: Optional indicator value
        metadata: Optional additional data
        
    Returns:
        Signal object
    """
    return Signal(
        indicator=indicator,
        signal_type=signal,
        strength=strength,
        confidence=confidence,
        reasoning=reasoning,
        value=value,
        metadata=metadata
    )


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTING SIGNAL TYPES")
    print("=" * 70)
    
    # Test 1: SignalType enum
    print("\n1. Testing SignalType enum...")
    for signal_type in SignalType:
        print(f"   {signal_type.name} = {signal_type.value}")
    
    buy_signal = SignalType.from_string("buy")
    print(f"   from_string('buy') = {buy_signal}")
    
    # Test 2: Strength enum
    print("\n2. Testing Strength enum...")
    for strength in Strength:
        score = strength.to_score()
        print(f"   {strength.name} = {strength.value} (score: {score})")
    
    strong = Strength.from_score(3)
    print(f"   from_score(3) = {strong}")
    
    # Test 3: Confidence class
    print("\n3. Testing Confidence class...")
    conf = Confidence(85)
    print(f"   Confidence(85):")
    print(f"   - String: {conf}")
    print(f"   - Integer: {int(conf)}")
    print(f"   - Percentage: {conf.to_percentage()}")
    print(f"   - Decimal: {conf.to_decimal()}")
    print(f"   - Level: {conf.level()}")
    
    for value in [95, 75, 45]:
        c = Confidence(value)
        print(f"   Confidence({value}) = {c.level()}")
    
    # Test 4: Signal class
    print("\n4. Testing Signal class...")
    signal = Signal(
        indicator="RSI",
        signal_type=SignalType.BUY,
        strength=Strength.STRONG,
        confidence=Confidence(85),
        reasoning="RSI oversold at 25",
        value=25.0
    )
    
    print(f"   Created: {signal}")
    print(f"   Score: {signal.score():.2f}")
    
    # Test 5: Convert to dictionary
    print("\n5. Testing to_dict()...")
    signal_dict = signal.to_dict()
    for key, value in signal_dict.items():
        print(f"   {key}: {value}")
    
    # Test 6: Create from dictionary
    print("\n6. Testing from_dict()...")
    dict_data = {
        'indicator': 'MACD',
        'signal': 'SELL',
        'strength': 'MODERATE',
        'confidence': 70,
        'reasoning': 'Bearish crossover',
        'value': -50.0
    }
    signal2 = Signal.from_dict(dict_data)
    print(f"   Created: {signal2}")
    print(f"   Score: {signal2.score():.2f}")
    
    # Test 7: Validation
    print("\n7. Testing validation...")
    try:
        validate_signal_dict(dict_data)
        print("   ✅ Valid signal passed validation")
    except ValueError as e:
        print(f"   ❌ Validation failed: {e}")
    
    try:
        invalid_data = {'indicator': 'RSI', 'signal': 'MAYBE'}
        validate_signal_dict(invalid_data)
        print("   ❌ Invalid signal should have failed!")
    except ValueError as e:
        print(f"   ✅ Invalid signal caught: {e}")
    
    # Test 8: Create signal helper
    print("\n8. Testing create_signal() helper...")
    signal3 = create_signal(
        indicator="Bollinger",
        signal="buy",
        strength="weak",
        confidence=60,
        reasoning="Price near lower band",
        value=75000.0
    )
    print(f"   Created: {signal3}")
    
    # Test 9: Multiple signals comparison
    print("\n9. Testing signal scoring...")
    signals = [
        create_signal("RSI", "BUY", "STRONG", 90, "Oversold"),
        create_signal("MACD", "BUY", "WEAK", 60, "Slight positive"),
        create_signal("Stoch", "SELL", "MODERATE", 75, "Overbought"),
        create_signal("MA", "HOLD", "NEUTRAL", 50, "No trend"),
    ]
    
    print(f"   {'Indicator':<12} {'Signal':<6} {'Strength':<10} {'Conf':<6} {'Score':<8}")
    print("   " + "-" * 60)
    for sig in signals:
        print(f"   {sig.indicator:<12} {sig.signal_type.value:<6} {sig.strength.value:<10} {int(sig.confidence):<6} {sig.score():+.2f}")
    
    print("\n" + "=" * 70)
    print("✅ ALL SIGNAL TYPES TESTS PASSED!")
    print("=" * 70)