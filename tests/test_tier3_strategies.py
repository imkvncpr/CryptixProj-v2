import pytest
import numpy as np
from src.strategies.base import BaseStrategy
from src.strategies.momentum import GoldenCrossStrategy, RSIExtremeStrategy, MACDMomentumStrategy
from src.strategies.mean_reversion import BollingerBandsStrategy, RSIReversalStrategy


class TestGoldenCrossStrategy:
    def test_import(self):
        assert GoldenCrossStrategy is not None
    
    def test_instantiate(self):
        strategy = GoldenCrossStrategy()
        assert strategy.name == "Golden Cross"
    
    def test_signal(self):
        strategy = GoldenCrossStrategy()
        prices = np.array(list(range(100, 350)), dtype=float)
        signal = strategy.calculate_signal(prices)
        assert signal is not None


class TestRSIExtremeStrategy:
    def test_import(self):
        assert RSIExtremeStrategy is not None
    
    def test_instantiate(self):
        strategy = RSIExtremeStrategy()
        assert strategy.name == "RSI Extremes"
    
    def test_signal(self):
        strategy = RSIExtremeStrategy()
        prices = np.array(list(range(100, 150)) + list(range(150, 80, -1)), dtype=float)
        signal = strategy.calculate_signal(prices)
        assert signal is not None


class TestMACDMomentumStrategy:
    def test_import(self):
        assert MACDMomentumStrategy is not None
    
    def test_instantiate(self):
        strategy = MACDMomentumStrategy()
        assert strategy.name == "MACD Momentum"
    
    def test_signal(self):
        strategy = MACDMomentumStrategy()
        prices = np.array(list(range(100, 150)) + list(range(150, 80, -1)), dtype=float)
        signal = strategy.calculate_signal(prices)
        assert signal is not None


class TestBollingerBandsStrategy:
    def test_import(self):
        assert BollingerBandsStrategy is not None
    
    def test_instantiate(self):
        strategy = BollingerBandsStrategy()
        assert strategy.name == "Bollinger Bands"
    
    def test_signal(self):
        strategy = BollingerBandsStrategy()
        prices = np.array(list(range(100, 150)), dtype=float)
        signal = strategy.calculate_signal(prices)
        assert signal is not None


class TestRSIReversalStrategy:
    def test_import(self):
        assert RSIReversalStrategy is not None
    
    def test_instantiate(self):
        strategy = RSIReversalStrategy()
        assert strategy.name == "RSI Reversal"
    
    def test_signal(self):
        strategy = RSIReversalStrategy()
        prices = np.array(list(range(100, 120)) + list(range(120, 80, -1)), dtype=float)
        signal = strategy.calculate_signal(prices)
        assert signal is not None
