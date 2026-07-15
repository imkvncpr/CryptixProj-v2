import pytest
from src.trading.broker.base import Broker, OrderStatus
from src.trading.broker.coinbase import CoinbaseBroker
from src.trading.risk_limiter import RiskLimiter

class TestOrderStatus:
    """Test OrderStatus constants"""
    
    def test_order_status_values(self):
        assert OrderStatus.PENDING == "PENDING"
        assert OrderStatus.FILLED == "FILLED"
        assert OrderStatus.PARTIAL == "PARTIAL"
        assert OrderStatus.CANCELLED == "CANCELLED"
        assert OrderStatus.FAILED == "FAILED"

class TestBrokerBase:
    """Test abstract Broker base class"""
    
    def test_broker_is_abstract(self):
        from abc import ABC
        assert issubclass(Broker, ABC)
    
    def test_broker_requires_implementation(self):
        with pytest.raises(TypeError):
            Broker("key", "secret")
    
    def test_broker_init(self):
        class MockBroker(Broker):
            def connect(self):
                return True
            def buy(self, s, q, p):
                return "id"
            def sell(self, s, q, p):
                return "id"
            def get_order_status(self, oid):
                return OrderStatus.FILLED
            def get_balance(self, c):
                return 1000.0
            def get_positions(self):
                return []
            def cancel_order(self, oid):
                return True
        
        broker = MockBroker("key", "secret", sandbox=True)
        assert broker.api_key == "key"
        assert broker.api_secret == "secret"
        assert broker.sandbox is True

class TestCoinbaseBroker:
    """Test CoinbaseBroker implementation"""
    
    @pytest.fixture
    def broker(self):
        return CoinbaseBroker("test_key", "test_secret", "test_pass", sandbox=True)
    
    def test_coinbase_init(self, broker):
        assert broker.api_key == "test_key"
        assert broker.passphrase == "test_pass"
        assert broker.sandbox is True
        assert "sandbox" in broker.base_url
    
    def test_coinbase_sandbox_url(self, broker):
        assert broker.base_url == "https://api-sandbox.coinbase.com"
    
    def test_coinbase_production_url(self):
        broker = CoinbaseBroker("key", "secret", "pass", sandbox=False)
        assert broker.base_url == "https://api.coinbase.com"
    
    def test_coinbase_session_created(self, broker):
        assert broker.session is not None
    
    def test_get_auth_headers(self, broker):
        headers = broker._get_auth_headers("GET", "/products")
        assert "CB-ACCESS-KEY" in headers
        assert "CB-ACCESS-SIGN" in headers
        assert "CB-ACCESS-TIMESTAMP" in headers
        assert "CB-ACCESS-PASSPHRASE" in headers
        assert "Content-Type" in headers
    
    def test_coinbase_repr(self, broker):
        repr_str = repr(broker)
        assert "CoinbaseBroker" in repr_str
        assert "Sandbox: True" in repr_str

class TestRiskLimiter:
    """Test RiskLimiter functionality"""
    
    @pytest.fixture
    def limiter(self):
        return RiskLimiter(initial_capital=10000, max_drawdown_pct=10, 
                          max_loss_per_trade=2, max_position_pct=5)
    
    def test_risklimiter_init(self, limiter):
        assert limiter.initial_capital == 10000
        assert limiter.current_balance == 10000
        assert limiter.peak_balance == 10000
    
    def test_no_drawdown_initially(self, limiter):
        assert limiter.get_current_drawdown() == 0.0
    
    def test_update_balance(self, limiter):
        limiter.update_balance(9500)
        assert limiter.current_balance == 9500
        assert limiter.peak_balance == 10000
        assert limiter.daily_loss == 500
    
    def test_update_balance_new_peak(self, limiter):
        limiter.update_balance(9500)
        limiter.update_balance(11000)
        assert limiter.current_balance == 11000
        assert limiter.peak_balance == 11000
        assert limiter.daily_loss == 0
    
    def test_current_drawdown(self, limiter):
        limiter.update_balance(9000)
        drawdown = limiter.get_current_drawdown()
        assert drawdown == pytest.approx(10.0)
    
    def test_can_trade_small_position(self, limiter):
        can_trade = limiter.can_trade("BTC", 0.01, 45000)
        assert can_trade is True
    
    def test_can_trade_position_too_large(self, limiter):
        can_trade = limiter.can_trade("BTC", 0.2, 40000)
        assert can_trade is False
    
    def test_can_trade_max_drawdown_reached(self, limiter):
        limiter.update_balance(9000)
        can_trade = limiter.can_trade("BTC", 0.01, 45000)
        assert can_trade is False
    
    def test_max_position_size(self, limiter):
        max_size = limiter.get_max_position_size()
        assert max_size == 500
    
    def test_get_summary(self, limiter):
        limiter.update_balance(9500)
        summary = limiter.get_summary()
        assert summary['current_balance'] == 9500
        assert summary['peak_balance'] == 10000
        assert summary['current_drawdown_pct'] == pytest.approx(5.0)
        assert summary['daily_loss'] == 500
    
    def test_risklimiter_repr(self, limiter):
        repr_str = repr(limiter)
        assert "RiskLimiter" in repr_str
        assert "10,000.00" in repr_str

class TestBrokerIntegration:
    """Integration tests for broker"""
    
    def test_mock_broker_workflow(self):
        class MockBroker(Broker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.orders = {}
            def connect(self):
                return True
            def buy(self, s, q, p):
                oid = f"order_{len(self.orders)}"
                self.orders[oid] = {"side": "buy", "status": OrderStatus.PENDING}
                return oid
            def sell(self, s, q, p):
                oid = f"order_{len(self.orders)}"
                self.orders[oid] = {"side": "sell", "status": OrderStatus.PENDING}
                return oid
            def get_order_status(self, oid):
                return self.orders.get(oid, {}).get("status", OrderStatus.FAILED)
            def get_balance(self, c):
                return 1000.0
            def get_positions(self):
                return list(self.orders.values())
            def cancel_order(self, oid):
                if oid in self.orders:
                    self.orders[oid]["status"] = OrderStatus.CANCELLED
                    return True
                return False
        
        broker = MockBroker("key", "secret")
        assert broker.connect() is True
        
        buy_id = broker.buy("BTC", 0.1, 45000)
        assert buy_id is not None
        assert broker.get_order_status(buy_id) == OrderStatus.PENDING
        
        sell_id = broker.sell("ETH", 1.0, 3000)
        assert sell_id is not None
        
        assert len(broker.get_positions()) == 2
        
        assert broker.cancel_order(buy_id) is True
        assert broker.get_order_status(buy_id) == OrderStatus.CANCELLED

class TestRiskLimiterIntegration:
    """Integration tests for risk limiter"""
    
    def test_risklimiter_with_trades(self):
        limiter = RiskLimiter(10000, max_loss_per_trade=10)
        
        can_trade_1 = limiter.can_trade("BTC", 0.01, 45000)
        assert can_trade_1 is True
        
        limiter.update_balance(9500)
        
        can_trade_2 = limiter.can_trade("BTC", 0.01, 45000)
        assert can_trade_2 is True
        
        limiter.update_balance(8500)
        
        can_trade_3 = limiter.can_trade("BTC", 0.01, 45000)
        assert can_trade_3 is False
    
    def test_risklimiter_daily_loss_limit(self):
        limiter = RiskLimiter(10000, max_loss_per_trade=1)
        
        limiter.update_balance(9899)
        
        can_trade = limiter.can_trade("BTC", 0.01, 45000)
        assert can_trade is False

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
