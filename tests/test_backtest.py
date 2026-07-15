"""
tests/test_backtest.py
═════════════════════════════════════════════════════════════
Comprehensive tests for Backtest engine
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.backtesting.backtest import Backtest
from src.backtesting.data_handler import HistoricalDataHandler, create_sample_csv
from src.backtesting.models import Candle, BacktestTrade, BacktestResults


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """Create test data once for all tests"""
    Path('./data').mkdir(exist_ok=True)
    create_sample_csv('./data/test_bitcoin.csv', 'BTC/USD', 100, 28500)
    yield


@pytest.fixture
def data_handler():
    """Create and initialize data handler"""
    handler = HistoricalDataHandler('./data')
    handler.load_csv('BTC/USD', './data/test_bitcoin.csv')
    return handler


@pytest.fixture
def backtest(data_handler):
    """Create a backtest instance"""
    return Backtest(
        initial_capital=10000,
        data_handler=data_handler,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 4, 10),
        commission=0.001
    )


@pytest.fixture
def backtest_no_commission(data_handler):
    """Create a backtest with no commission"""
    return Backtest(
        initial_capital=10000,
        data_handler=data_handler,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 4, 10),
        commission=0.0
    )


class TestBacktestInitialization:
    """Test Backtest class initialization"""
    
    def test_backtest_creation(self, backtest):
        assert backtest.initial_capital == 10000
        assert backtest.current_cash == 10000
        assert len(backtest.closed_trades) == 0
        assert backtest.trade_id_counter == 0
    
    def test_backtest_parameters(self, backtest):
        assert backtest.start_date == datetime(2023, 1, 1)
        assert backtest.end_date == datetime(2023, 4, 10)
        assert backtest.max_risk_per_trade == 0.02
        assert backtest.max_position_size == 0.30
    
    def test_backtest_portfolio_initialized(self, backtest):
        assert isinstance(backtest.portfolio_values, dict)
        assert isinstance(backtest.daily_pnl, dict)
        assert len(backtest.portfolio_values) == 0
        assert len(backtest.daily_pnl) == 0
    
    def test_backtest_positions_initialized(self, backtest):
        assert isinstance(backtest.positions, dict)
        assert len(backtest.positions) == 0


class TestTradeExecution:
    """Test adding trades to backtest"""
    
    def test_add_single_winning_trade(self, backtest_no_commission):
        success = backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        assert success == True
        assert len(backtest_no_commission.closed_trades) == 1
        trade = backtest_no_commission.closed_trades[0]
        assert trade.trade_id == 1
        assert trade.symbol == 'BTC/USD'
        assert trade.gross_pnl == 100.0
        assert trade.is_winning_trade == True
    
    def test_add_single_losing_trade(self, backtest_no_commission):
        success = backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=29500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=28500,
            stop_loss=30000,
            take_profit=31000
        )
        
        assert success == True
        assert len(backtest_no_commission.closed_trades) == 1
        trade = backtest_no_commission.closed_trades[0]
        assert trade.gross_pnl == -100.0
        assert trade.is_losing_trade == True
    
    def test_add_multiple_trades(self, backtest_no_commission):
        for i in range(3):
            success = backtest_no_commission.add_trade(
                symbol='BTC/USD',
                entry_date=datetime(2023, 1, 1) + timedelta(days=i*10),
                entry_price=28500,
                entry_quantity=0.1,
                exit_date=datetime(2023, 1, 5) + timedelta(days=i*10),
                exit_price=29500,
                stop_loss=28000,
                take_profit=31000
            )
            assert success == True
        
        assert len(backtest_no_commission.closed_trades) == 3
        assert backtest_no_commission.trade_id_counter == 3
    
    def test_trade_with_commission(self, backtest):
        starting_cash = backtest.current_cash
        success = backtest.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        assert success == True
        cash_change = backtest.current_cash - starting_cash
        assert 90 < cash_change < 100
    
    def test_insufficient_cash_fails(self, backtest):
        success = backtest.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=50000,
            entry_quantity=1,
            exit_date=datetime(2023, 1, 5),
            exit_price=51000,
            stop_loss=49000,
            take_profit=52000
        )
        
        assert success == False
        assert len(backtest.closed_trades) == 0
    
    def test_invalid_quantity_fails(self, backtest):
        success = backtest.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=-0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        assert success == False
        assert len(backtest.closed_trades) == 0
    
    def test_invalid_dates_fails(self, backtest):
        success = backtest.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 5),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 1),
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        assert success == False
        assert len(backtest.closed_trades) == 0
    
    def test_zero_quantity_fails(self, backtest):
        success = backtest.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        assert success == False
    
    def test_equal_entry_exit_dates_fails(self, backtest):
        success = backtest.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 1),
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        assert success == False


class TestMetricsCalculation:
    """Test metric calculations"""
    
    def test_win_rate_single_winning_trade(self, backtest_no_commission):
        backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        win_rate = backtest_no_commission.calculate_win_rate()
        assert win_rate == 100.0
    
    def test_win_rate_single_losing_trade(self, backtest_no_commission):
        backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=29500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=28500,
            stop_loss=30000,
            take_profit=31000
        )
        
        win_rate = backtest_no_commission.calculate_win_rate()
        assert win_rate == 0.0
    
    def test_win_rate_mixed_trades(self, backtest_no_commission):
        for _ in range(2):
            backtest_no_commission.add_trade(
                symbol='BTC/USD',
                entry_date=datetime(2023, 1, 1),
                entry_price=28500,
                entry_quantity=0.1,
                exit_date=datetime(2023, 1, 5),
                exit_price=29500,
                stop_loss=28000,
                take_profit=31000
            )
        
        backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 6),
            entry_price=29500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 10),
            exit_price=28500,
            stop_loss=30000,
            take_profit=31000
        )
        
        win_rate = backtest_no_commission.calculate_win_rate()
        assert win_rate == pytest.approx(66.67, 0.1)
    
    def test_win_rate_no_trades(self, backtest):
        win_rate = backtest.calculate_win_rate()
        assert win_rate == 0.0
    
    def test_profit_factor_wins_and_losses(self, backtest_no_commission):
        for _ in range(2):
            backtest_no_commission.add_trade(
                symbol='BTC/USD',
                entry_date=datetime(2023, 1, 1),
                entry_price=28500,
                entry_quantity=0.1,
                exit_date=datetime(2023, 1, 5),
                exit_price=29500,
                stop_loss=28000,
                take_profit=31000
            )
        
        backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 6),
            entry_price=29500,
            entry_quantity=0.05,
            exit_date=datetime(2023, 1, 10),
            exit_price=28500,
            stop_loss=30000,
            take_profit=31000
        )
        
        profit_factor = backtest_no_commission.calculate_profit_factor()
        assert profit_factor == pytest.approx(4.0, 0.1)
    
    def test_profit_factor_all_wins(self, backtest_no_commission):
        backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        profit_factor = backtest_no_commission.calculate_profit_factor()
        assert profit_factor == float('inf')
    
    def test_profit_factor_all_losses(self, backtest_no_commission):
        backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=29500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=28500,
            stop_loss=30000,
            take_profit=31000
        )
        
        profit_factor = backtest_no_commission.calculate_profit_factor()
        assert profit_factor == 0.0
    
    def test_profit_factor_no_trades(self, backtest):
        profit_factor = backtest.calculate_profit_factor()
        assert profit_factor == 0.0
    
    def test_total_return_calculation(self, backtest_no_commission):
        backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        total_return = backtest_no_commission.calculate_total_return()
        assert total_return == pytest.approx(1.0, 0.1)
    
    def test_sharpe_ratio_with_data(self, backtest):
        backtest.portfolio_values[datetime(2023, 1, 1)] = 10000
        backtest.portfolio_values[datetime(2023, 1, 2)] = 10100
        backtest.portfolio_values[datetime(2023, 1, 3)] = 10050
        backtest.portfolio_values[datetime(2023, 1, 4)] = 10200
        
        backtest.daily_pnl[datetime(2023, 1, 1)] = 0
        backtest.daily_pnl[datetime(2023, 1, 2)] = 100
        backtest.daily_pnl[datetime(2023, 1, 3)] = 50
        backtest.daily_pnl[datetime(2023, 1, 4)] = 200
        
        sharpe = backtest.calculate_sharpe_ratio()
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_sharpe_ratio_empty(self, backtest):
        sharpe = backtest.calculate_sharpe_ratio()
        assert sharpe == 0.0
    
    def test_max_drawdown_calculation(self, backtest):
        backtest.portfolio_values[datetime(2023, 1, 1)] = 10000
        backtest.portfolio_values[datetime(2023, 1, 2)] = 11000
        backtest.portfolio_values[datetime(2023, 1, 3)] = 9500
        
        max_dd = backtest.calculate_max_drawdown()
        assert max_dd == pytest.approx(13.64, 0.1)
    
    def test_max_drawdown_no_data(self, backtest):
        max_dd = backtest.calculate_max_drawdown()
        assert max_dd == 0.0
    
    def test_max_drawdown_only_gains(self, backtest):
        backtest.portfolio_values[datetime(2023, 1, 1)] = 10000
        backtest.portfolio_values[datetime(2023, 1, 2)] = 10500
        backtest.portfolio_values[datetime(2023, 1, 3)] = 11000
        
        max_dd = backtest.calculate_max_drawdown()
        assert max_dd == 0.0


class TestPortfolioTracking:
    """Test portfolio value tracking"""
    
    def test_cash_deduction_on_entry(self, backtest_no_commission):
        starting_cash = backtest_no_commission.current_cash
        
        backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        assert backtest_no_commission.current_cash == starting_cash + 100
    
    def test_position_size_calculation(self, backtest):
        size = backtest._calculate_position_size(28500)
        assert size > 0
        assert size < 1.0
        assert not np.isnan(size)
    
    def test_portfolio_value_calculation(self, backtest):
        backtest.current_cash = 5000
        backtest.positions = {'BTC/USD': 0.1}
        
        candle = Candle(
            timestamp=datetime(2023, 1, 1),
            open=28500,
            high=29000,
            low=28400,
            close=28800,
            volume=150.5,
            symbol='BTC/USD'
        )
        
        portfolio_value = backtest._calculate_portfolio_value(candle)
        expected = 5000 + (0.1 * 28800)
        assert portfolio_value == pytest.approx(expected)


class TestIntegration:
    """Integration tests"""
    
    def test_multiple_trades_sequence(self, backtest_no_commission):
        backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 6),
            entry_price=29500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 10),
            exit_price=28500,
            stop_loss=30000,
            take_profit=31000
        )
        
        backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 11),
            entry_price=28500,
            entry_quantity=0.2,
            exit_date=datetime(2023, 1, 15),
            exit_price=30500,
            stop_loss=27500,
            take_profit=32000
        )
        
        assert len(backtest_no_commission.closed_trades) == 3
        assert backtest_no_commission.trade_id_counter == 3
        assert backtest_no_commission.closed_trades[0].is_winning_trade
        assert backtest_no_commission.closed_trades[1].is_losing_trade
        assert backtest_no_commission.closed_trades[2].is_winning_trade
    
    def test_backtest_string_representation(self, backtest):
        repr_str = repr(backtest)
        assert 'Backtest' in repr_str
        assert '10000' in repr_str  
        assert '0 trades' in repr_str
    
    def test_backtest_with_optional_trade_parameters(self, backtest_no_commission):
        success = backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        assert success
        trade = backtest_no_commission.closed_trades[0]
        assert trade.stop_loss == 28000
        assert trade.take_profit == 31000
    
    def test_backtest_with_none_optional_parameters(self, backtest_no_commission):
        success = backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500
        )
        
        assert success
        trade = backtest_no_commission.closed_trades[0]
        assert trade.stop_loss is None
        assert trade.take_profit is None


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_very_small_trade(self, backtest_no_commission):
        success = backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.001,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        assert success
        assert len(backtest_no_commission.closed_trades) == 1
    
    def test_large_trade(self, backtest_no_commission):
        success = backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=100,
            entry_quantity=50,
            exit_date=datetime(2023, 1, 5),
            exit_price=150,
            stop_loss=50,
            take_profit=200
        )
        
        assert success
        assert len(backtest_no_commission.closed_trades) == 1
    
    def test_tiny_price_movement(self, backtest_no_commission):
        success = backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=28500.01,
            stop_loss=28000,
            take_profit=31000
        )
        
        assert success
        trade = backtest_no_commission.closed_trades[0]
        assert trade.gross_pnl == pytest.approx(0.001, 0.0001)
    
    def test_same_candle_trading(self, backtest_no_commission):
        base_date = datetime(2023, 1, 1)
        
        success1 = backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=base_date,
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=base_date + timedelta(minutes=1),
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        success2 = backtest_no_commission.add_trade(
            symbol='BTC/USD',
            entry_date=base_date + timedelta(minutes=1),
            entry_price=29500,
            entry_quantity=0.1,
            exit_date=base_date + timedelta(minutes=2),
            exit_price=28500,
            stop_loss=30000,
            take_profit=31000
        )
        
        assert success1 and success2
        assert len(backtest_no_commission.closed_trades) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])