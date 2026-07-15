import pytest
import numpy as np
from src.portfolio.position_sizing import PositionSizer
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.rebalancer import PortfolioRebalancer


class TestPositionSizing:
    def test_kelly_criterion(self):
        size = PositionSizer.kelly_criterion(
            win_rate=0.55,
            avg_win=1000,
            avg_loss=800,
            account_size=100000
        )
        assert size > 0
        assert size <= 25000
    
    def test_fixed_fraction(self):
        size = PositionSizer.fixed_fraction(100000, 0.02)
        assert size == 2000
    
    def test_volatility_adjusted(self):
        size = PositionSizer.volatility_adjusted(100000, 2.5, risk_pct=1.0)
        assert size > 0
    
    def test_correlation_adjusted(self):
        positions = {'BTC': 50000, 'ETH': 30000}
        correlations = {('BTC', 'ETH'): 0.75}
        adjusted = PositionSizer.correlation_adjusted(positions, correlations)
        assert adjusted['BTC'] <= positions['BTC']


class TestPortfolioOptimizer:
    def test_equal_weight(self):
        weights = PortfolioOptimizer.equal_weight_portfolio(4)
        assert np.isclose(np.sum(weights), 1.0)
        assert np.allclose(weights, 0.25)
    
    def test_risk_parity(self):
        returns = np.random.randn(100, 4) * 0.02 + 0.01
        weights = PortfolioOptimizer.risk_parity_portfolio(returns)
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)
    
    def test_min_variance(self):
        returns = np.random.randn(100, 4) * 0.02 + 0.01
        weights = PortfolioOptimizer.min_variance_portfolio(returns)
        assert np.isclose(np.sum(weights), 1.0)
    
    def test_max_sharpe(self):
        returns = np.random.randn(100, 4) * 0.02 + 0.01
        weights = PortfolioOptimizer.max_sharpe_ratio(returns)
        assert np.isclose(np.sum(weights), 1.0)


class TestPortfolioRebalancer:
    def test_check_rebalance_needed(self):
        current = np.array([0.30, 0.35, 0.35])
        target = np.array([0.33, 0.33, 0.34])
        needed = PortfolioRebalancer.check_rebalance_needed(current, target, 0.05)
        assert isinstance(needed, (bool, np.bool_))
    
    def test_calculate_trades(self):
        trades = PortfolioRebalancer.calculate_rebalance_trades(
            {'BTC': 30000, 'ETH': 35000},
            {'BTC': 0.33, 'ETH': 0.67},
            100000
        )
        assert len(trades) == 2
    
    def test_time_based_rebalance(self):
        rebalancer = PortfolioRebalancer()
        needed = rebalancer.time_based_rebalance('monthly')
        assert needed == True
    
    def test_drift_based_rebalance(self):
        current = np.array([0.30, 0.35, 0.35])
        target = np.array([0.33, 0.33, 0.34])
        needed = PortfolioRebalancer.drift_based_rebalance(current, target, 0.05)
        assert isinstance(needed, (bool, np.bool_))
