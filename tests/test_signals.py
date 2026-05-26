"""
Unit tests for signal_types module

Tests all enums, classes, and helper functions
in the signal types system.
"""

import pytest
import sys
import os

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.signals.signal_types import (
    SignalType,
    Strength,
    Confidence,
    Signal,
    validate_signal_dict,
    create_signal
)


# ============================================================================
# TEST SIGNALTYPE ENUM
# ============================================================================

def test_signal_type_values():
    """Test that SignalType enum has correct values"""
    assert SignalType.BUY.value == "BUY"
    assert SignalType.SELL.value == "SELL"
    assert SignalType.HOLD.value == "HOLD"


def test_signal_type_from_string():
    """Test converting strings to SignalType"""
    # Test uppercase
    assert SignalType.from_string("BUY") == SignalType.BUY
    assert SignalType.from_string("SELL") == SignalType.SELL
    assert SignalType.from_string("HOLD") == SignalType.HOLD
    
    # Test lowercase
    assert SignalType.from_string("buy") == SignalType.BUY
    assert SignalType.from_string("sell") == SignalType.SELL
    assert SignalType.from_string("hold") == SignalType.HOLD
    
    # Test mixed case
    assert SignalType.from_string("BuY") == SignalType.BUY


def test_signal_type_from_string_invalid():
    """Test that invalid strings raise ValueError"""
    with pytest.raises(ValueError):
        SignalType.from_string("MAYBE")
    
    with pytest.raises(ValueError):
        SignalType.from_string("INVALID")
    
    with pytest.raises(ValueError):
        SignalType.from_string("")


def test_signal_type_str():
    """Test string representation"""
    assert str(SignalType.BUY) == "BUY"
    assert str(SignalType.SELL) == "SELL"
    assert str(SignalType.HOLD) == "HOLD"


# ============================================================================
# TEST STRENGTH ENUM
# ============================================================================

def test_strength_values():
    """Test Strength enum values"""
    assert Strength.STRONG.value == "STRONG"
    assert Strength.MODERATE.value == "MODERATE"
    assert Strength.WEAK.value == "WEAK"
    assert Strength.NEUTRAL.value == "NEUTRAL"


def test_strength_to_score():
    """Test converting strength to numeric score"""
    assert Strength.STRONG.to_score() == 3
    assert Strength.MODERATE.to_score() == 2
    assert Strength.WEAK.to_score() == 1
    assert Strength.NEUTRAL.to_score() == 0


def test_strength_from_score():
    """Test creating Strength from score"""
    assert Strength.from_score(3) == Strength.STRONG
    assert Strength.from_score(2) == Strength.MODERATE
    assert Strength.from_score(1) == Strength.WEAK
    assert Strength.from_score(0) == Strength.NEUTRAL


def test_strength_from_score_invalid():
    """Test invalid scores raise ValueError"""
    with pytest.raises(ValueError):
        Strength.from_score(4)
    
    with pytest.raises(ValueError):
        Strength.from_score(-1)
    
    with pytest.raises(ValueError):
        Strength.from_score(99)


def test_strength_from_string():
    """Test converting strings to Strength"""
    assert Strength.from_string("STRONG") == Strength.STRONG
    assert Strength.from_string("strong") == Strength.STRONG
    assert Strength.from_string("MODERATE") == Strength.MODERATE
    assert Strength.from_string("WEAK") == Strength.WEAK
    assert Strength.from_string("NEUTRAL") == Strength.NEUTRAL


def test_strength_from_string_invalid():
    """Test invalid strings raise ValueError"""
    with pytest.raises(ValueError):
        Strength.from_string("ULTRA")
    
    with pytest.raises(ValueError):
        Strength.from_string("INVALID")


# ============================================================================
# TEST CONFIDENCE CLASS
# ============================================================================

def test_confidence_valid_values():
    """Test creating Confidence with valid values"""
    conf_min = Confidence(0)
    assert conf_min.value == 0
    
    conf_mid = Confidence(50)
    assert conf_mid.value == 50
    
    conf_max = Confidence(100)
    assert conf_max.value == 100
    
    # Test float conversion
    conf_float = Confidence(85.7)
    assert conf_float.value == 85


def test_confidence_invalid_range():
    """Test that invalid ranges raise ValueError"""
    with pytest.raises(ValueError):
        Confidence(-1)
    
    with pytest.raises(ValueError):
        Confidence(101)
    
    with pytest.raises(ValueError):
        Confidence(999)


def test_confidence_invalid_type():
    """Test that non-numbers raise TypeError"""
    with pytest.raises(TypeError):
        Confidence("85")
    
    with pytest.raises(TypeError):
        Confidence(None)


def test_confidence_str_methods():
    """Test string representation methods"""
    conf = Confidence(85)
    
    assert str(conf) == "85%"
    assert repr(conf) == "Confidence(85)"
    assert conf.to_percentage() == "85%"


def test_confidence_conversions():
    """Test numeric conversion methods"""
    conf = Confidence(85)
    
    assert int(conf) == 85
    assert conf.to_decimal() == 0.85
    
    conf2 = Confidence.from_decimal(0.75)
    assert conf2.value == 75


def test_confidence_level():
    """Test confidence level categorization"""
    # HIGH (80-100)
    assert Confidence(100).level() == "HIGH"
    assert Confidence(80).level() == "HIGH"
    
    # MEDIUM (60-79)
    assert Confidence(79).level() == "MEDIUM"
    assert Confidence(60).level() == "MEDIUM"
    
    # LOW (0-59)
    assert Confidence(59).level() == "LOW"
    assert Confidence(0).level() == "LOW"


# ============================================================================
# TEST SIGNAL CLASS
# ============================================================================

def test_signal_creation_with_enums():
    """Test creating Signal with enum types"""
    signal = Signal(
        indicator="RSI",
        signal_type=SignalType.BUY,
        strength=Strength.STRONG,
        confidence=Confidence(85),
        reasoning="Test"
    )
    
    assert signal.indicator == "RSI"
    assert signal.signal_type == SignalType.BUY
    assert signal.strength == Strength.STRONG
    assert signal.confidence.value == 85


def test_signal_auto_conversion():
    """Test that Signal auto-converts strings to enums"""
    signal = Signal(
        indicator="MACD",
        signal_type="BUY",
        strength="MODERATE",
        confidence=70,
        reasoning="Test"
    )
    
    assert signal.signal_type == SignalType.BUY
    assert signal.strength == Strength.MODERATE
    assert isinstance(signal.confidence, Confidence)
    assert signal.confidence.value == 70


def test_signal_hold_must_be_neutral():
    """Test that HOLD signals must have NEUTRAL strength"""
    # Valid: HOLD + NEUTRAL
    signal = Signal(
        indicator="MA",
        signal_type=SignalType.HOLD,
        strength=Strength.NEUTRAL,
        confidence=50,
        reasoning="No trend"
    )
    assert signal is not None
    
    # Invalid: HOLD + STRONG
    with pytest.raises(ValueError):
        Signal(
            indicator="MA",
            signal_type=SignalType.HOLD,
            strength=Strength.STRONG,
            confidence=50,
            reasoning="Invalid"
        )


def test_signal_str():
    """Test Signal string representation"""
    signal = Signal(
        indicator="RSI",
        signal_type=SignalType.BUY,
        strength=Strength.STRONG,
        confidence=Confidence(85),
        reasoning="Test"
    )
    
    assert str(signal) == "RSI: BUY (STRONG, 85%)"


def test_signal_to_dict():
    """Test converting Signal to dictionary"""
    signal = Signal(
        indicator="RSI",
        signal_type=SignalType.BUY,
        strength=Strength.STRONG,
        confidence=Confidence(85),
        reasoning="Oversold",
        value=25.0
    )
    
    result = signal.to_dict()
    
    assert result['indicator'] == "RSI"
    assert result['signal'] == "BUY"
    assert result['strength'] == "STRONG"
    assert result['confidence'] == 85
    assert result['reasoning'] == "Oversold"
    assert result['value'] == 25.0


def test_signal_from_dict():
    """Test creating Signal from dictionary"""
    data = {
        'indicator': 'MACD',
        'signal': 'SELL',
        'strength': 'MODERATE',
        'confidence': 70,
        'reasoning': 'Bearish crossover'
    }
    
    signal = Signal.from_dict(data)
    
    assert signal.indicator == "MACD"
    assert signal.signal_type == SignalType.SELL
    assert signal.strength == Strength.MODERATE
    assert signal.confidence.value == 70


def test_signal_score():
    """Test signal score calculation"""
    # BUY STRONG 90%
    signal1 = create_signal("RSI", "BUY", "STRONG", 90, "Test")
    assert signal1.score() == pytest.approx(2.7)
    
    # SELL MODERATE 75%
    signal2 = create_signal("MACD", "SELL", "MODERATE", 75, "Test")
    assert signal2.score() == pytest.approx(-1.5)
    
    # HOLD NEUTRAL 50%
    signal3 = create_signal("MA", "HOLD", "NEUTRAL", 50, "Test")
    assert signal3.score() == pytest.approx(0.0)


# ============================================================================
# TEST HELPER FUNCTIONS
# ============================================================================

def test_validate_signal_dict_valid():
    """Test validating correct signal dictionaries"""
    valid_data = {
        'indicator': 'RSI',
        'signal': 'BUY',
        'strength': 'STRONG',
        'confidence': 85,
        'reasoning': 'Oversold'
    }
    
    result = validate_signal_dict(valid_data)
    assert result == True


def test_validate_signal_dict_missing_fields():
    """Test validation catches missing fields"""
    with pytest.raises(ValueError, match="Missing required field"):
        validate_signal_dict({
            'indicator': 'RSI',
            'signal': 'BUY',
            'confidence': 85,
            'reasoning': 'Test'
        })


def test_validate_signal_dict_invalid_signal():
    """Test validation catches invalid signal types"""
    with pytest.raises(ValueError):
        validate_signal_dict({
            'indicator': 'RSI',
            'signal': 'MAYBE',
            'strength': 'STRONG',
            'confidence': 85,
            'reasoning': 'Test'
        })


def test_validate_signal_dict_invalid_confidence():
    """Test validation catches invalid confidence"""
    with pytest.raises(ValueError):
        validate_signal_dict({
            'indicator': 'RSI',
            'signal': 'BUY',
            'strength': 'STRONG',
            'confidence': 999,
            'reasoning': 'Test'
        })


def test_create_signal_helper():
    """Test create_signal convenience function"""
    signal = create_signal(
        indicator="RSI",
        signal="buy",
        strength="strong",
        confidence=85,
        reasoning="Test"
    )
    
    assert signal.signal_type == SignalType.BUY
    assert signal.strength == Strength.STRONG
    assert signal.confidence.value == 85