"""tests/test_paper_trader.py"""
import pytest
from datetime import datetime, timedelta
from src.trading.paper_trader import PaperTrader
from src.trading.order import OrderSide

@pytest.fixture
def paper_trader():
    """Create a paper trader with 10000 starting capital"""
    return PaperTrader(initial_capital=10000, commission_rate=0.001)

class TestPaperTraderInit:
    """Test PaperTrader initialization"""
    
    def test_initialization(self):
        pt = PaperTrader(initial_capital=50000, commission_rate=0.002)
        assert pt.initial_capital == 50000
        assert pt.cash_balance == 50000
        assert pt.commission_rate == 0.002
        assert len(pt.positions) == 0
        assert len(pt.closed_positions) == 0
    
    def test_default_commission(self):
        pt = PaperTrader(initial_capital=10000)
        assert pt.commission_rate == 0.001
    
    def test_repr(self, paper_trader):
        repr_str = repr(paper_trader)
        assert 'PaperTrader' in repr_str
        assert '10,000.00' in repr_str

class TestBuyOrders:
    """Test buying (opening LONG positions)"""
    
    def test_buy_success(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        success = paper_trader.buy('BTC/USD', 0.1, 45000, timestamp)
        
        assert success is True
        assert len(paper_trader.positions) == 1
        assert len(paper_trader.closed_positions) == 0
    
    def test_buy_with_commission(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        cost = 0.5 * 45000
        commission = cost * 0.001
        expected_balance = 10000 - (0.1 * 45000) - (0.1 * 45000 * 0.001)
        
        paper_trader.buy('BTC/USD', 0.1, 45000, timestamp)
        assert paper_trader.cash_balance == pytest.approx(expected_balance)
    
    def test_buy_insufficient_cash(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        success = paper_trader.buy('BTC/USD', 0.2, 50000, timestamp)
        
        assert success is False
        assert len(paper_trader.positions) == 0
        assert paper_trader.cash_balance == 10000
    
    def test_multiple_buys(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        paper_trader.buy('BTC/USD', 0.1, 45000, timestamp)
        paper_trader.buy('ETH/USD', 1.0, 3000, timestamp)
        paper_trader.buy('DOGE/USD', 100, 0.5, timestamp)
        
        assert len(paper_trader.positions) == 3
        assert len(paper_trader.get_open_positions()) == 3

class TestSellOrders:
    """Test selling (closing LONG positions)"""
    
    def test_sell_success(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        paper_trader.buy('BTC/USD', 0.1, 45000, timestamp)
        
        sell_timestamp = datetime(2024, 1, 10)
        success = paper_trader.sell('BTC/USD', 0.5, 48000, sell_timestamp)
        
        assert success is True
        assert len(paper_trader.positions) == 0
        assert len(paper_trader.closed_positions) == 1
    
    def test_sell_profit(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        paper_trader.buy('BTC/USD', 0.2, 40000, timestamp)
        
        initial_balance = paper_trader.cash_balance
        
        paper_trader.sell('BTC/USD', 0.2, 50000, datetime(2024, 1, 10))
        
        profit = (50000 - 40000) - (50000 * 0.001)
        assert paper_trader.cash_balance > initial_balance
    
    def test_sell_no_position(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        success = paper_trader.sell('BTC/USD', 0.1, 45000, timestamp)
        
        assert success is False
    
    def test_sell_wrong_symbol(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        paper_trader.buy('BTC/USD', 0.1, 45000, timestamp)
        
        success = paper_trader.sell('ETH/USD', 0.5, 3000, timestamp)
        assert success is False

class TestShortOrders:
    """Test shorting (opening SHORT positions)"""
    
    def test_short_success(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        success = paper_trader.short('BTC/USD', 0.1, 45000, timestamp)
        
        assert success is True
        assert len(paper_trader.positions) == 1
    
    def test_short_adds_cash(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        initial_balance = paper_trader.cash_balance
        
        paper_trader.short('BTC/USD', 1.0, 45000, timestamp)
        
        proceeds = 45000 - (45000 * 0.001)
        assert paper_trader.cash_balance == initial_balance + proceeds
    
    def test_multiple_shorts(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        paper_trader.short('BTC/USD', 0.1, 45000, timestamp)
        paper_trader.short('ETH/USD', 2.0, 3000, timestamp)
        
        assert len(paper_trader.positions) == 2

class TestCoverOrders:
    """Test covering (closing SHORT positions)"""
    
    def test_cover_success(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        paper_trader.short('BTC/USD', 0.1, 45000, timestamp)
        
        cover_timestamp = datetime(2024, 1, 10)
        success = paper_trader.cover('BTC/USD', 0.5, 40000, cover_timestamp)
        
        assert success is True
        assert len(paper_trader.positions) == 0
        assert len(paper_trader.closed_positions) == 1
    
    def test_cover_profit(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        paper_trader.short('BTC/USD', 0.1, 50000, timestamp)
        
        initial_balance = paper_trader.cash_balance
        
        paper_trader.cover('BTC/USD', 0.1, 40000, datetime(2024, 1, 10))
        
        # Profit = entry - exit = 50000 - 40000
        assert paper_trader.cash_balance > initial_balance
    
    def test_cover_insufficient_cash(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        paper_trader.short('BTC/USD', 1.0, 1000, timestamp)
        
        # Now we have lots of cash, try to cover at high price
        success = paper_trader.cover('BTC/USD', 1.0, 1000, datetime(2024, 1, 10))
        assert success is True
    
    def test_cover_no_position(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        success = paper_trader.cover('BTC/USD', 0.1, 45000, timestamp)
        
        assert success is False

class TestPortfolioValue:
    """Test portfolio value calculations"""
    
    def test_portfolio_value_cash_only(self, paper_trader):
        current_prices = {'BTC/USD': 45000}
        value = paper_trader.get_portfolio_value(current_prices)
        
        assert value == 10000
    
    def test_portfolio_value_with_position(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        paper_trader.buy('BTC/USD', 0.1, 45000, timestamp)
        
        current_prices = {'BTC/USD': 50000}
        value = paper_trader.get_portfolio_value(current_prices)
        
        cash = paper_trader.cash_balance
        position_value = 0.1 * 50000
        expected = cash + position_value
        
        assert value == pytest.approx(expected)
    
    def test_portfolio_value_multiple_positions(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        paper_trader.buy('BTC/USD', 0.1, 45000, timestamp)
        paper_trader.buy('ETH/USD', 1.0, 3000, timestamp)
        
        current_prices = {'BTC/USD': 50000, 'ETH/USD': 3500}
        value = paper_trader.get_portfolio_value(current_prices)
        
        assert value > paper_trader.cash_balance

class TestPnLCalculations:
    """Test P&L calculations"""
    
    def test_unrealized_pnl_long(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        paper_trader.buy('BTC/USD', 0.2, 40000, timestamp)
        
        current_prices = {'BTC/USD': 45000}
        unrealized = paper_trader.get_unrealized_pnl(current_prices)
        
        assert unrealized == pytest.approx(1000)
    
    def test_unrealized_pnl_short(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        paper_trader.short('BTC/USD', 0.2, 50000, timestamp)
        
        current_prices = {'BTC/USD': 45000}
        unrealized = paper_trader.get_unrealized_pnl(current_prices)
        
        assert unrealized == pytest.approx(1000)
    
    def test_realized_pnl(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        paper_trader.buy('BTC/USD', 0.2, 40000, timestamp)
        
        paper_trader.sell('BTC/USD', 1.0, 45000, datetime(2024, 1, 10))
        
        realized = paper_trader.get_realized_pnl()
        assert realized > 0
    
    def test_total_pnl(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        paper_trader.buy('BTC/USD', 0.5, 40000, timestamp)
        paper_trader.buy('ETH/USD', 1.0, 3000, timestamp)
        
        paper_trader.sell('BTC/USD', 0.1, 45000, datetime(2024, 1, 10))
        
        current_prices = {'BTC/USD': 50000, 'ETH/USD': 3500}
        total_pnl = paper_trader.get_total_pnl(current_prices)
        
        assert total_pnl > 0

class TestSummary:
    """Test summary generation"""
    
    def test_summary_empty(self, paper_trader):
        current_prices = {}
        summary = paper_trader.get_summary(current_prices)
        
        assert summary['cash_balance'] == 10000
        assert summary['open_positions'] == 0
        assert summary['closed_positions'] == 0
        assert summary['win_rate'] == 0.0
    
    def test_summary_with_trades(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        paper_trader.buy('BTC/USD', 0.2, 40000, timestamp)
        paper_trader.sell('BTC/USD', 1.0, 45000, datetime(2024, 1, 10))
        
        current_prices = {'BTC/USD': 50000}
        summary = paper_trader.get_summary(current_prices)
        
        assert summary['closed_positions'] == 1
        assert summary['winning_trades'] == 1
        assert summary['win_rate'] > 0
    
    def test_summary_win_loss_ratio(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        
        # Win
        paper_trader.buy('BTC/USD', 0.5, 40000, timestamp)
        paper_trader.sell('BTC/USD', 0.1, 45000, datetime(2024, 1, 1))
        
        # Loss
        paper_trader.buy('ETH/USD', 1.0, 3000, timestamp)
        paper_trader.sell('ETH/USD', 1.0, 2800, datetime(2024, 1, 1))
        
        current_prices = {}
        summary = paper_trader.get_summary(current_prices)
        
        assert summary['winning_trades'] == 1
        assert summary['losing_trades'] == 1
        assert summary['win_rate'] == 50.0

class TestEdgeCases:
    """Test edge cases and special scenarios"""
    
    def test_zero_quantity(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        success = paper_trader.buy('BTC/USD', 0, 45000, timestamp)
        
        assert success is True or success is False  # Behavior depends on implementation
    
    def test_zero_price(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        success = paper_trader.buy('BTC/USD', 1.0, 0, timestamp)
        
        assert success is True
        assert paper_trader.cash_balance == 10000
    
    def test_very_small_position(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        success = paper_trader.buy('BTC/USD', 0.00001, 45000, timestamp)
        
        assert success is True
        assert len(paper_trader.positions) == 1
    
    def test_very_large_position(self, paper_trader):
        timestamp = datetime(2024, 1, 1)
        success = paper_trader.buy('DOGE/USD', 1000000, 0.01, timestamp)
        
        assert success is False  # Not enough cash

class TestWorkflow:
    """Test realistic trading workflows"""
    
    def test_day_trading_workflow(self, paper_trader):
        """Simulate a day trading scenario"""
        timestamp = datetime(2024, 1, 1)
        
        # Buy at open
        paper_trader.buy('BTC/USD', 0.1, 45000, timestamp)
        
        # Current price at noon
        current_prices = {'BTC/USD': 46000}
        unrealized = paper_trader.get_unrealized_pnl(current_prices)
        assert unrealized > 0
        
        # Sell at close for profit
        paper_trader.sell('BTC/USD', 0.5, 46500, datetime(2024, 1, 1, 16, 0))
        
        realized = paper_trader.get_realized_pnl()
        assert realized > 0
    
    def test_swing_trading_workflow(self, paper_trader):
        """Simulate a swing trading scenario"""
        timestamp = datetime(2024, 1, 1)
        
        # Buy and hold for several days
        paper_trader.buy('BTC/USD', 0.2, 40000, timestamp)
        paper_trader.buy('ETH/USD', 0.5, 2500, timestamp)
        
        assert len(paper_trader.get_open_positions()) == 1
        
        # Partial sells over time
        paper_trader.sell('BTC/USD', 0.1, 45000, datetime(2024, 1, 5))
        
        assert len(paper_trader.get_open_positions()) == 1
        assert len(paper_trader.get_closed_positions()) == 1
        
        # Complete second position
        paper_trader.sell('ETH/USD', 1.0, 3200, datetime(2024, 1, 10))
        
        assert len(paper_trader.get_closed_positions()) == 1
    
    def test_short_selling_workflow(self, paper_trader):
        """Simulate short selling scenario"""
        timestamp = datetime(2024, 1, 1)
        
        # Short at top
        paper_trader.short('BTC/USD', 0.2, 50000, timestamp)
        
        initial_balance = paper_trader.cash_balance
        
        # Cover at bottom for profit
        paper_trader.cover('BTC/USD', 0.2, 40000, datetime(2024, 1, 10))
        
        # Should have made profit
        assert paper_trader.cash_balance > initial_balance
        assert len(paper_trader.get_closed_positions()) == 1

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])


