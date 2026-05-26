import pytest
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.signals.signal_generator import SignalGenerator
from src.signals.signal_types import SignalType, Strength


class TestMACDIntegration:
    
    def test_macd_bullish_crossover(self):
        gen = SignalGenerator()
        np.random.seed(42)
        data = {'close': list(np.random.uniform(100, 110, 50))}
        result = gen.indicators['MACD'].analyze(data)
        
        assert result is not None
        assert result['indicator'] == 'MACD'
        assert 'signal' in result
    
    def test_macd_insufficient_data(self):
        gen = SignalGenerator()
        data = {'close': [100, 101, 102]}
        result = gen.indicators['MACD'].analyze(data)
        
        assert result is None


class TestBollingerBandsIntegration:
    
    def test_bollinger_bands_oversold(self):
        gen = SignalGenerator()
        np.random.seed(100)
        close = list(np.linspace(100, 80, 50))
        data = {'close': close}
        result = gen.indicators['BOLLINGER'].analyze(data)
        
        assert result is not None
        assert result['signal'] in ['BUY', 'HOLD']
        assert 'position' in result['values']
    
    def test_bollinger_bands_overbought(self):
        gen = SignalGenerator()
        np.random.seed(101)
        close = list(np.linspace(80, 100, 50))
        data = {'close': close}
        result = gen.indicators['BOLLINGER'].analyze(data)
        
        assert result is not None
        assert result['signal'] in ['SELL', 'HOLD']
    
    def test_bollinger_bands_values(self):
        gen = SignalGenerator()
        np.random.seed(42)
        data = {'close': list(np.random.uniform(95, 105, 50))}
        result = gen.indicators['BOLLINGER'].analyze(data)
        
        assert 'upper_band' in result['values']
        assert 'middle_band' in result['values']
        assert 'lower_band' in result['values']
        assert result['values']['upper_band'] > result['values']['middle_band']
        assert result['values']['middle_band'] > result['values']['lower_band']


class TestStochasticIntegration:
    
    def test_stochastic_oversold(self):
        gen = SignalGenerator()
        np.random.seed(100)
        high = list(np.linspace(120, 110, 50))
        low = list(np.linspace(100, 90, 50))
        close = list(np.linspace(115, 92, 50))
        data = {'high': high, 'low': low, 'close': close}
        
        result = gen.indicators['STOCHASTIC'].analyze(data)
        
        assert result is not None
        assert result['signal'] in ['BUY', 'HOLD']
    
    def test_stochastic_overbought(self):
        gen = SignalGenerator()
        np.random.seed(101)
        high = list(np.linspace(110, 120, 50))
        low = list(np.linspace(90, 100, 50))
        close = list(np.linspace(92, 115, 50))
        data = {'high': high, 'low': low, 'close': close}
        
        result = gen.indicators['STOCHASTIC'].analyze(data)
        
        assert result is not None
        assert result['signal'] in ['SELL', 'HOLD']
    
    def test_stochastic_values(self):
        gen = SignalGenerator()
        np.random.seed(42)
        high = list(np.random.uniform(110, 120, 50))
        low = list(np.random.uniform(100, 110, 50))
        close = list(np.random.uniform(105, 115, 50))
        data = {'high': high, 'low': low, 'close': close}
        
        result = gen.indicators['STOCHASTIC'].analyze(data)
        
        assert 'k_percent' in result['values']
        assert 'd_percent' in result['values']
        assert 0 <= result['values']['k_percent'] <= 100
        assert 0 <= result['values']['d_percent'] <= 100


class TestSignalGeneratorIntegration:
    
    def test_generator_initialization(self):
        gen = SignalGenerator()
        assert len(gen.indicators) == 8
        assert 'MACD' in gen.indicators
        assert 'BOLLINGER' in gen.indicators
        assert 'STOCHASTIC' in gen.indicators
    
    def test_generate_signals_with_three_indicators(self):
        gen = SignalGenerator()
        np.random.seed(42)
        data = {
            'high': list(np.random.uniform(110, 120, 50)),
            'low': list(np.random.uniform(100, 110, 50)),
            'close': list(np.random.uniform(105, 115, 50))
        }
        
        result = gen.generate_signals(data)
        
        assert 'final_signal' in result
        assert 'individual_signals' in result
        assert 'summary' in result
    
    def test_final_signal_has_valid_type(self):
        gen = SignalGenerator()
        np.random.seed(42)
        data = {
            'high': list(np.random.uniform(110, 120, 50)),
            'low': list(np.random.uniform(100, 110, 50)),
            'close': list(np.random.uniform(105, 115, 50))
        }
        
        result = gen.generate_signals(data)
        final = result['final_signal']
        
        assert final.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        assert 0 <= int(final.confidence) <= 100
    
    def test_summary_statistics(self):
        gen = SignalGenerator()
        np.random.seed(42)
        data = {
            'high': list(np.random.uniform(110, 120, 50)),
            'low': list(np.random.uniform(100, 110, 50)),
            'close': list(np.random.uniform(105, 115, 50))
        }
        
        result = gen.generate_signals(data)
        summary = result['summary']
        
        assert summary['total_indicators'] == 8
        assert summary['signals_generated'] >= 3
        assert summary['buy_count'] >= 0
        assert summary['sell_count'] >= 0
        assert summary['hold_count'] >= 0
    
    def test_confidence_within_range(self):
        gen = SignalGenerator()
        np.random.seed(42)
        data = {
            'high': list(np.random.uniform(110, 120, 50)),
            'low': list(np.random.uniform(100, 110, 50)),
            'close': list(np.random.uniform(105, 115, 50))
        }
        
        result = gen.generate_signals(data)
        confidence = int(result['final_signal'].confidence)
        
        assert 0 <= confidence <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])