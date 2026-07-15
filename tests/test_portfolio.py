"""Tests for Portfolio Manager"""

import pytest
from src.portfolio.manager import PortfolioManager, Position


class TestPosition:
    def test_position_creation(self):
        pos = Position('bitcoin', 0.5, 45000)
        assert pos.symbol == 'bitcoin'
        assert pos.quantity == 0.5
        assert pos.entry_price == 45000
    
    def test_position_pnl_profit(self):
        pos = Position('bitcoin', 0.5, 45000)
        pos.update_price(46000)
        assert pos.pnl == 500
        assert pos.pnl_percent == pytest.approx(2.22, 0.01)
    
    def test_position_pnl_loss(self):
        pos = Position('bitcoin', 0.5, 45000)
        pos.update_price(44000)
        assert pos.pnl == -500
        assert pos.pnl_percent == pytest.approx(-2.22, 0.01)
    
    def test_position_values(self):
        pos = Position('ethereum', 5, 2500)
        assert pos.entry_value == 12500
        assert pos.current_value == 12500
        pos.update_price(2600)
        assert pos.current_value == 13000


class TestPortfolioManager:
    @pytest.fixture
    def portfolio(self):
        return PortfolioManager(initial_cash=50000)
    
    def test_initialization(self, portfolio):
        assert portfolio.initial_cash == 50000
        assert portfolio.current_cash == 50000
        assert len(portfolio.positions) == 0
        assert portfolio.portfolio_value == 50000
    
    def test_add_position(self, portfolio):
        portfolio.add_position('bitcoin', 0.1, 45000)
        assert 'bitcoin' in portfolio.positions
        assert portfolio.current_cash == 50000 - 4500
        assert len(portfolio.positions) == 1
    
    def test_add_position_insufficient_cash(self, portfolio):
        with pytest.raises(ValueError):
            portfolio.add_position('bitcoin', 10, 45000)
    
    def test_add_multiple_positions(self, portfolio):
        portfolio.add_position('bitcoin', 0.1, 45000)
        portfolio.add_position('ethereum', 5, 2500)
        assert len(portfolio.positions) == 2
        assert portfolio.current_cash == 50000 - 4500 - 12500
    
    def test_update_prices(self, portfolio):
        portfolio.add_position('bitcoin', 0.1, 45000)
        portfolio.add_position('ethereum', 5, 2500)
        portfolio.update_prices({'bitcoin': 46000, 'ethereum': 2600})
        
        btc_pos = portfolio.get_position('bitcoin')
        eth_pos = portfolio.get_position('ethereum')
        
        assert btc_pos.current_price == 46000
        assert eth_pos.current_price == 2600
    
    def test_close_position(self, portfolio):
        portfolio.add_position('bitcoin', 0.1, 45000)
        initial_cash = portfolio.current_cash
        pnl = portfolio.close_position('bitcoin', 46000)
            
        assert pnl == 100  # FIXED: was 500
        assert 'bitcoin' not in portfolio.positions
        assert portfolio.current_cash == initial_cash + 4600
    
    def test_portfolio_value(self, portfolio):
        portfolio.add_position('bitcoin', 0.1, 45000)
        portfolio.update_prices({'bitcoin': 46000})
        
        assert portfolio.total_position_value == 4600
        expected = 4600 + (50000 - 4500)
        assert portfolio.portfolio_value == expected
    
    def test_total_pnl(self, portfolio):
        portfolio.add_position('bitcoin', 0.1, 45000)
        portfolio.add_position('ethereum', 5, 2500)
        portfolio.update_prices({'bitcoin': 46000, 'ethereum': 2600})
        assert portfolio.total_pnl == 600  # FIXED: was 1000
    
    def test_total_pnl_percent(self, portfolio):
        portfolio.add_position('bitcoin', 0.1, 45000)
        portfolio.update_prices({'bitcoin': 46000})
        assert portfolio.total_pnl_percent == pytest.approx(0.2, 0.01)  # FIXED: was 1.0

    
    def test_allocation(self, portfolio):
        portfolio.add_position('bitcoin', 0.1, 45000)
        allocation = portfolio.get_allocation()
        
        assert 'bitcoin' in allocation
        assert 'cash' in allocation
        assert allocation['bitcoin'] + allocation['cash'] == pytest.approx(100.0, 0.1)
    
    def test_summary(self, portfolio):
        portfolio.add_position('bitcoin', 0.1, 45000)
        portfolio.update_prices({'bitcoin': 46000})
        
        summary = portfolio.get_summary()
        
        assert 'portfolio_value' in summary
        assert 'total_pnl' in summary
        assert 'allocation' in summary
        assert 'positions' in summary
        assert summary['num_positions'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])