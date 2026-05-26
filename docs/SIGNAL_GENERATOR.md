# Signal Generator System

## Overview

The Signal Generator is an orchestration system that combines 8 technical indicators to generate unified trading signals with confidence scoring.

## Architecture

### Components

- **SignalTypes**: Enums and classes for type-safe signal handling
- **Indicators**: 8 independent technical indicator implementations
- **SignalGenerator**: Orchestrates indicators and aggregates signals

### Indicators

1. **MACD** (Momentum)
   - Compares fast (12) and slow (26) exponential moving averages
   - Signals bullish/bearish crossovers
   - Strength based on histogram magnitude

2. **Bollinger Bands** (Volatility)
   - Price bands using 20-day SMA ± 2 standard deviations
   - Identifies overbought (>80%) and oversold (<20%) conditions
   - Position measures price within band range

3. **Stochastic** (Momentum)
   - Compares closing price within 14-day high-low range
   - %K shows raw position, %D shows 3-day smoothed version
   - Crossovers generate strong signals

## Signal Aggregation

### Scoring Logic


### Thresholds

- **BUY**: Aggregate score > 0.5
- **SELL**: Aggregate score < -0.5
- **HOLD**: Aggregate score between -0.5 and 0.5

## Usage

```python
from src.signals.signal_generator import SignalGenerator

gen = SignalGenerator()

data = {
    'high': [110, 112, 115, ...],
    'low': [105, 107, 110, ...],
    'close': [108, 111, 113, ...]
}

result = gen.generate_signals(data)

# Access results
final_signal = result['final_signal']
individual_signals = result['individual_signals']
summary = result['summary']

print(f"Signal: {final_signal.signal_type}")
print(f"Confidence: {final_signal.confidence}")
print(f"Buy Count: {summary['buy_count']}")
```

## Signal Interpretation

### BUY Signals

- **STRONG**: Multiple indicators bullish, high confidence
- **MODERATE**: Some indicators bullish, moderate confidence
- **WEAK**: Few indicators bullish, low confidence

### SELL Signals

- **STRONG**: Multiple indicators bearish, high confidence
- **MODERATE**: Some indicators bearish, moderate confidence
- **WEAK**: Few indicators bearish, low confidence

### HOLD Signals

- No strong consensus from indicators
- Wait for clearer signals

## Testing

All components have comprehensive test coverage:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_signal_types.py -v
python -m pytest tests/test_signal_generator.py -v
```

## SOLID Principles Applied

### Single Responsibility
- Each indicator has one job: analyze its domain
- SignalGenerator only orchestrates and aggregates

### Open/Closed
- Easy to add new indicators without modifying existing code
- All indicators implement same interface

### Liskov Substitution
- All indicators interchangeable with same interface
- Consistent return format across all indicators

### Interface Segregation
- Indicators only need `analyze()` method
- Minimal coupling between components

### Dependency Injection
- Indicators passed to SignalGenerator (could be extended)
- No hard-coded dependencies

## Error Handling

- All indicators handle insufficient data gracefully
- Missing data returns `None`, not crashes
- SignalGenerator continues even if individual indicator fails
- Comprehensive logging at each step

## Future Enhancements

- Add more technical indicators (RSI, MACD, etc.)
- Implement weighting system for indicator importance
- Add machine learning for signal validation
- Real-time data streaming integration
- Database storage for signal history