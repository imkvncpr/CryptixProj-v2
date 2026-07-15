import pytest
from datetime import datetime
from src.backtesting.models import Candle, BacktestTrade, BacktestResults


class TestCandleModel:
    """Test Candle class"""
    
    def test_candle_creation(self):
        """Test creating a candle"""
        candle = Candle(
            timestamp=datetime(2023, 1, 1),
            open=28500,
            high=29000,
            low=28400,
            close=28800,
            volume=150.5,
            symbol='BTC/USD'
        )
        
        assert candle.symbol == 'BTC/USD'
        assert candle.open == 28500
        assert candle.close == 28800
        print(candle)  # Should print formatted candle
    
    
    def test_candle_properties(self):
        """Test candle properties"""
        candle = Candle(
            timestamp=datetime(2023, 1, 1),
            open=28500,
            high=29000,
            low=28400,
            close=28800,
            volume=150.5,
            symbol='BTC/USD'
        )
        
        # Test properties
        assert candle.is_bullish == True  # close > open
        assert candle.body_size == 300  # |28800 - 28500|
        assert candle.hl_range == 600  # 29000 - 28400
        assert candle.mid == 28700  # (29000 + 28400) / 2
    
    
    def test_candle_invalid_prices(self):
        """Test that invalid candles raise errors"""
        
        # High < Low (invalid)
        with pytest.raises(ValueError):
            Candle(
                timestamp=datetime(2023, 1, 1),
                open=28500,
                high=28000,  # ❌ Less than low!
                low=28400,
                close=28800,
                volume=150.5,
                symbol='BTC/USD'
            )
    
    
    def test_candle_repr(self):
        """Test candle string representation"""
        candle = Candle(
            timestamp=datetime(2023, 1, 1),
            open=28500,
            high=29000,
            low=28400,
            close=28800,
            volume=150.5,
            symbol='BTC/USD'
        )
        
        string = str(candle)
        assert 'BTC/USD' in string
        assert '28500' in string
        print(f"Candle repr: {string}")


class TestBacktestTradeModel:
    """Test BacktestTrade class"""
    
    def test_trade_creation(self):
        """Test creating a trade"""
        trade = BacktestTrade(
            trade_id=1,
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        assert trade.trade_id == 1
        assert trade.symbol == 'BTC/USD'
        assert trade.entry_price == 28500
        print(trade)
    
    
    def test_trade_pnl_calculations(self):
        """Test P&L calculations"""
        trade = BacktestTrade(
            trade_id=1,
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        # Test calculations
        assert trade.entry_value == 2850.0  # 28500 * 0.1
        assert trade.exit_value == 2950.0   # 29500 * 0.1
        assert trade.gross_pnl == 100.0     # 2950 - 2850
        assert trade.pnl_percent == pytest.approx(3.51, 0.01)
        assert trade.is_winning_trade == True
    
    
    def test_trade_losing_trade(self):
        """Test losing trade"""
        trade = BacktestTrade(
            trade_id=2,
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=27500,  # Lower than entry
            stop_loss=27000,
            take_profit=30000
        )
        
        assert trade.gross_pnl == -100.0
        assert trade.is_losing_trade == True
        assert trade.was_stopped_out == False
    
    
    def test_trade_duration(self):
        """Test trade duration calculation"""
        trade = BacktestTrade(
            trade_id=1,
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 6),  # 5 days later
            exit_price=29500,
            stop_loss=28000,
            take_profit=31000
        )
        
        assert trade.duration_days == 5
    
    
    def test_trade_risk_reward(self):
        """Test risk/reward ratio"""
        trade = BacktestTrade(
            trade_id=1,
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500,
            stop_loss=28000,      # Risk: 500
            take_profit=31000     # Reward: 2500
        )
        
        # Risk/Reward = 2500 / 500 = 5.0
        assert trade.risk_distance == 500
        assert trade.reward_distance == 2500
        assert trade.risk_reward_ratio == 5.0
    
    
    def test_trade_invalid_dates(self):
        """Test that invalid dates raise errors"""
        with pytest.raises(ValueError):
            BacktestTrade(
                trade_id=1,
                symbol='BTC/USD',
                entry_date=datetime(2023, 1, 5),
                entry_price=28500,
                entry_quantity=0.1,
                exit_date=datetime(2023, 1, 1),  # ❌ Before entry!
                exit_price=29500,
                stop_loss=28000,
                take_profit=31000
            )


class TestIntegration:
    """Integration tests"""
    
    def test_candle_and_trade_together(self):
        """Test using candle data in a trade"""
        
        # Create entry candle
        entry_candle = Candle(
            timestamp=datetime(2023, 1, 1),
            open=28500,
            high=29000,
            low=28400,
            close=28800,
            volume=150.5,
            symbol='BTC/USD'
        )
        
        # Create exit candle
        exit_candle = Candle(
            timestamp=datetime(2023, 1, 5),
            open=29200,
            high=29500,
            low=29000,
            close=29400,
            volume=155.0,
            symbol='BTC/USD'
        )
        
        # Create trade using candle data
        trade = BacktestTrade(
            trade_id=1,
            symbol='BTC/USD',
            entry_date=entry_candle.timestamp,
            entry_price=entry_candle.close,
            entry_quantity=0.1,
            exit_date=exit_candle.timestamp,
            exit_price=exit_candle.close,
            stop_loss=entry_candle.low,
            take_profit=29800
        )
        
        assert trade.is_winning_trade
        assert trade.gross_pnl == 60.0  # (29400 - 28800) * 0.1
    
    
    def test_multiple_trades(self):
        """Test creating multiple trades"""
        
        trades = [
            BacktestTrade(
                trade_id=1,
                symbol='BTC/USD',
                entry_date=datetime(2023, 1, 1),
                entry_price=28500,
                entry_quantity=0.1,
                exit_date=datetime(2023, 1, 5),
                exit_price=29500,
                stop_loss=28000,
                take_profit=31000
            ),
            BacktestTrade(
                trade_id=2,
                symbol='BTC/USD',
                entry_date=datetime(2023, 1, 6),
                entry_price=29500,
                entry_quantity=0.1,
                exit_date=datetime(2023, 1, 10),
                exit_price=28500,  # Loss
                stop_loss=30000,
                take_profit=31500
            ),
            BacktestTrade(
                trade_id=3,
                symbol='BTC/USD',
                entry_date=datetime(2023, 1, 11),
                entry_price=28500,
                entry_quantity=0.2,
                exit_date=datetime(2023, 1, 15),
                exit_price=30000,
                stop_loss=27500,
                take_profit=31500
            )
        ]
        
        # Verify trades
        assert len(trades) == 3
        assert trades[0].is_winning_trade == True
        assert trades[1].is_losing_trade == True
        assert trades[2].is_winning_trade == True
        
        # Calculate metrics
        total_pnl = sum(t.gross_pnl for t in trades)
        winning = sum(1 for t in trades if t.is_winning_trade)
        win_rate = (winning / len(trades)) * 100
        
        print(f"\nTotal P&L: ${total_pnl:.2f}")
        print(f"Win Rate: {win_rate:.1f}%")


if __name__ == '__main__':
    # Run with: python -m pytest tests/model_test.py -v
    # Or: python tests/model_test.py
    pytest.main([__file__, '-v'])