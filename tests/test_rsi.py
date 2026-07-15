"""
Tests for RSI Indicator
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.indicators.momentum.rsi import RSI, calculate_rsi, is_overbought, is_oversold


class TestRSIIndicator:
    """Test RSI Indicator class"""
    
    @pytest.fixture
    def rsi(self):
        """Create RSI instance"""
        return RSI(period=14)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'open': [100 + i*0.5 for i in range(30)],
            'high': [102 + i*0.5 for i in range(30)],
            'low': [98 + i*0.5 for i in range(30)],
            'close': [101 + i*0.5 for i in range(30)],
            'volume': [1000000] * 30
        }, index=dates)
        return data
    
    @pytest.fixture
    def uptrend_data(self):
        """Create strong uptrend data"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        closes = [100 + (i * 1.5) for i in range(30)]
        data = pd.DataFrame({
            'open': [c - 0.5 for c in closes],
            'high': [c + 1 for c in closes],
            'low': [c - 1 for c in closes],
            'close': closes,
            'volume': [1000000] * 30
        }, index=dates)
        return data
    
    @pytest.fixture
    def downtrend_data(self):
        """Create strong downtrend data"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        closes = [100 - (i * 1.5) for i in range(30)]
        data = pd.DataFrame({
            'open': [c + 0.5 for c in closes],
            'high': [c + 1 for c in closes],
            'low': [c - 1 for c in closes],
            'close': closes,
            'volume': [1000000] * 30
        }, index=dates)
        return data
    
    def test_rsi_initialization(self, rsi):
        """Test RSI initialization"""
        assert rsi.period == 14
        assert rsi.name == 'RSI'
    
    def test_rsi_calculate_returns_series(self, rsi, sample_data):
        """Test that calculate returns pandas Series"""
        result = rsi.calculate(sample_data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
    
    def test_rsi_values_in_range(self, rsi, sample_data):
        """Test that RSI values are between 0 and 100"""
        result = rsi.calculate(sample_data)
        # Skip NaN values (first 14 periods)
        valid_rsi = result.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_rsi_uptrend_high_values(self, rsi, uptrend_data):
        """Test RSI in uptrend gives high values"""
        result = rsi.calculate(uptrend_data)
        last_rsi = result.iloc[-1]
        # In strong uptrend, RSI should be > 50
        assert last_rsi > 50
    
    def test_rsi_downtrend_low_values(self, rsi, downtrend_data):
        """Test RSI in downtrend gives low values"""
        result = rsi.calculate(downtrend_data)
        last_rsi = result.iloc[-1]
        # In strong downtrend, RSI should be < 50
        assert last_rsi < 50
    
    def test_rsi_interpret_overbought(self, rsi):
        """Test interpretation of overbought signal"""
        result = rsi.interpret(75.0)
        assert result['signal'] == 'SELL'
        assert result['strength'] == 'MODERATE'
        assert result['confidence'] == 70
    
    def test_rsi_interpret_extreme_overbought(self, rsi):
        """Test interpretation of extreme overbought"""
        result = rsi.interpret(85.0)
        assert result['signal'] == 'SELL'
        assert result['strength'] == 'STRONG'
        assert result['confidence'] == 85
    
    def test_rsi_interpret_oversold(self, rsi):
        """Test interpretation of oversold signal"""
        result = rsi.interpret(25.0)
        assert result['signal'] == 'BUY'
        assert result['strength'] == 'MODERATE'
        assert result['confidence'] == 70
    
    def test_rsi_interpret_extreme_oversold(self, rsi):
        """Test interpretation of extreme oversold"""
        result = rsi.interpret(15.0)
        assert result['signal'] == 'BUY'
        assert result['strength'] == 'STRONG'
        assert result['confidence'] == 85
    
    def test_rsi_interpret_neutral(self, rsi):
        """Test interpretation of neutral zone"""
        result = rsi.interpret(50.0)
        assert result['signal'] == 'HOLD'
        assert result['strength'] == 'NEUTRAL'
        assert result['confidence'] == 50
    
    def test_rsi_interpret_result_format(self, rsi):
        """Test that interpret returns correct format"""
        result = rsi.interpret(50.0)
        assert 'indicator' in result
        assert 'value' in result
        assert 'signal' in result
        assert 'strength' in result
        assert 'confidence' in result
        assert 'reasoning' in result
        assert result['indicator'] == 'RSI'
    
    def test_rsi_get_latest(self, rsi, sample_data):
        """Test get_latest method"""
        rsi.calculate(sample_data)
        latest = rsi.get_latest()
        assert isinstance(latest, (int, float))
        assert 0 <= latest <= 100
    
    def test_rsi_with_different_periods(self, sample_data):
        """Test RSI with different periods"""
        rsi_7 = RSI(period=7)
        rsi_21 = RSI(period=21)
        
        result_7 = rsi_7.calculate(sample_data)
        result_21 = rsi_21.calculate(sample_data)
        
        assert len(result_7) == len(result_21) == len(sample_data)
        # Shorter period should have less NaN values
        assert result_7.isna().sum() < result_21.isna().sum()
    
    def test_rsi_with_volatile_data(self):
        """Test RSI with highly volatile data"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        # Alternating up and down
        closes = [100 if i % 2 == 0 else 90 for i in range(30)]
        data = pd.DataFrame({
            'open': closes,
            'high': [c + 10 for c in closes],
            'low': [c - 10 for c in closes],
            'close': closes,
            'volume': [1000000] * 30
        }, index=dates)
        
        rsi = RSI(period=14)
        result = rsi.calculate(data)
        valid_rsi = result.dropna()
        
        # Should have some values
        assert len(valid_rsi) > 0
        # All should be valid range
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()
    
    def test_rsi_consistency(self, sample_data):
        """Test that RSI calculation is consistent"""
        rsi1 = RSI(period=14)
        rsi2 = RSI(period=14)
        
        result1 = rsi1.calculate(sample_data)
        result2 = rsi2.calculate(sample_data)
        
        # Results should be identical
        pd.testing.assert_series_equal(result1, result2)


class TestRSIHelperFunctions:
    """Test standalone RSI helper functions"""
    
    def test_calculate_rsi_function(self):
        """Test standalone calculate_rsi function"""
        closes = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                  111, 110, 112, 114, 113, 115, 117, 116, 118, 120]
        
        rsi = calculate_rsi(closes, period=14)
        
        assert isinstance(rsi, float)
        assert 0 <= rsi <= 100
    
    def test_calculate_rsi_uptrend(self):
        """Test calculate_rsi in uptrend"""
        closes = [100 + i for i in range(20)]
        rsi = calculate_rsi(closes, period=14)
        assert rsi > 50  # Should be bullish
    
    def test_calculate_rsi_downtrend(self):
        """Test calculate_rsi in downtrend"""
        closes = [100 - i for i in range(20)]
        rsi = calculate_rsi(closes, period=14)
        assert rsi < 50  # Should be bearish
    
    def test_is_overbought_true(self):
        """Test is_overbought returns True"""
        assert is_overbought(75) is True
        assert is_overbought(80) is True
        assert is_overbought(100) is True
    
    def test_is_overbought_false(self):
        """Test is_overbought returns False"""
        assert is_overbought(65) is False
        assert is_overbought(70) is False
        assert is_overbought(50) is False
    
    def test_is_overbought_custom_threshold(self):
        """Test is_overbought with custom threshold"""
        assert is_overbought(75, threshold=80) is False
        assert is_overbought(75, threshold=70) is True
    
    def test_is_oversold_true(self):
        """Test is_oversold returns True"""
        assert is_oversold(25) is True
        assert is_oversold(20) is True
        assert is_oversold(0) is True
    
    def test_is_oversold_false(self):
        """Test is_oversold returns False"""
        assert is_oversold(35) is False
        assert is_oversold(30) is False
        assert is_oversold(50) is False
    
    def test_is_oversold_custom_threshold(self):
        """Test is_oversold with custom threshold"""
        assert is_oversold(25, threshold=20) is False
        assert is_oversold(25, threshold=30) is True


class TestRSIEdgeCases:
    """Test RSI edge cases"""
    
    def test_rsi_with_constant_prices(self):
        """Test RSI with constant prices (no volatility)"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'open': [100] * 30,
            'high': [100] * 30,
            'low': [100] * 30,
            'close': [100] * 30,
            'volume': [1000000] * 30
        }, index=dates)
        
        rsi = RSI(period=14)
        result = rsi.calculate(data)
        valid_rsi = result.dropna()
        
        # With no price changes, RSI should be 50 (neutral)
        if len(valid_rsi) > 0:
            assert valid_rsi.iloc[-1] == 50
    
    def test_rsi_with_single_spike(self):
        """Test RSI with single price spike"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        closes = [100] * 20 + [150] + [100] * 9
        data = pd.DataFrame({
            'open': closes,
            'high': [c + 5 for c in closes],
            'low': [c - 5 for c in closes],
            'close': closes,
            'volume': [1000000] * 30
        }, index=dates)
        
        rsi = RSI(period=14)
        result = rsi.calculate(data)
        valid_rsi = result.dropna()
        
        # After spike, RSI should show some bullish influence
        assert len(valid_rsi) > 0
        assert 0 <= valid_rsi.iloc[-1] <= 100
    
    def test_rsi_with_minimum_data(self):
        """Test RSI with minimum required data points"""
        dates = pd.date_range(start='2024-01-01', periods=15, freq='D')
        data = pd.DataFrame({
            'open': [100 + i for i in range(15)],
            'high': [102 + i for i in range(15)],
            'low': [98 + i for i in range(15)],
            'close': [101 + i for i in range(15)],
            'volume': [1000000] * 15
        }, index=dates)
        
        rsi = RSI(period=14)
        result = rsi.calculate(data)
        
        # Should have one valid value (at position 14)
        assert result.notna().sum() >= 1


class TestRSIIntegration:
    """Test RSI integration with data"""
    
    def test_rsi_with_real_like_data(self):
        """Test RSI with realistic price data"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Generate realistic price data with trend
        prices = [100]
        for _ in range(99):
            change = np.random.normal(0.5, 2)  # Mean 0.5, StdDev 2
            prices.append(max(prices[-1] + change, 1))  # Can't go below 1
        
        data = pd.DataFrame({
            'open': [p - 0.5 for p in prices],
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000000] * 100
        }, index=dates)
        
        rsi = RSI(period=14)
        result = rsi.calculate(data)
        
        # Should calculate for all data
        assert len(result) == 100
        # Should have valid RSI values
        valid_rsi = result.dropna()
        assert len(valid_rsi) > 80  # Most values should be valid
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])