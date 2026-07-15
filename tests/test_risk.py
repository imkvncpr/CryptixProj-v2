"""Tests for Risk Manager"""

import pytest
from src.risk.manager import RiskManager, RiskLevel


class TestRiskManager:
    @pytest.fixture
    def risk_mgr(self):
        return RiskManager(portfolio_value=50000, max_risk_per_trade=0.02)
    
    def test_initialization(self, risk_mgr):
        assert risk_mgr.portfolio_value == 50000
        assert risk_mgr.max_risk_per_trade == 0.02
        assert risk_mgr.max_drawdown == 0.20
        assert risk_mgr.max_position_size == 0.30
    
    def test_position_size_calculation(self, risk_mgr):
        size = risk_mgr.calculate_position_size(45000, 44000)
        assert size == pytest.approx(0.3333, 0.01)  # FIXED: was 1.0 (capped at 30%)
        
    def test_position_size_invalid(self, risk_mgr):
        with pytest.raises(ValueError):
            risk_mgr.calculate_position_size(45000, 45000)
    
    def test_stop_loss_calculation(self, risk_mgr):
        stop_loss = risk_mgr.calculate_stop_loss(45000, 500, "STRONG")
        assert stop_loss == pytest.approx(44250, 1)
    
    def test_stop_loss_different_strengths(self, risk_mgr):
        entry = 45000
        atr = 500
        
        strong = risk_mgr.calculate_stop_loss(entry, atr, "STRONG")
        moderate = risk_mgr.calculate_stop_loss(entry, atr, "MODERATE")
        weak = risk_mgr.calculate_stop_loss(entry, atr, "WEAK")
        
        assert strong > moderate > weak
    
    def test_take_profit_calculation(self, risk_mgr):
        take_profit = risk_mgr.calculate_take_profit(45000, 44000, 2.0)
        assert take_profit == 47000
    
    def test_take_profit_different_ratios(self, risk_mgr):
        entry = 45000
        stop = 44000
        
        tp_2 = risk_mgr.calculate_take_profit(entry, stop, 2.0)
        tp_3 = risk_mgr.calculate_take_profit(entry, stop, 3.0)
        
        assert tp_3 > tp_2
    
    def test_assess_trade_risk_safe(self, risk_mgr):
        report = risk_mgr.assess_trade_risk(45000, 44000, 100, 0.05)
        
        assert report['is_safe_to_trade'] is True
        assert report['risk_level'] == RiskLevel.LOW.value
    
    def test_assess_trade_risk_unsafe(self, risk_mgr):
        report = risk_mgr.assess_trade_risk(45000, 40000, -500, 0.25)
        
        assert report['is_safe_to_trade'] is False
        assert report['risk_level'] == RiskLevel.CRITICAL.value
    
    def test_assess_trade_risk_high_drawdown(self, risk_mgr):
        report = risk_mgr.assess_trade_risk(45000, 44000, -2000, 0.21)
        
        assert report['is_safe_to_trade'] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])