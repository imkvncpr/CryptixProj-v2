"""
Risk Manager - Position sizing and risk assessment
"""

import logging
from typing import Dict
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk classification"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskManager:
    """Manages trading risk and position sizing"""
    
    def __init__(
        self,
        portfolio_value: float,
        max_risk_per_trade: float = 0.02,
        max_drawdown: float = 0.20,
        max_position_size: float = 0.30
    ):
        """Initialize Risk Manager"""
        self.portfolio_value = portfolio_value
        self.max_risk_per_trade = max_risk_per_trade
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        
        logger.info(
            f"Risk Manager initialized: "
            f"portfolio=${portfolio_value:,.2f}, "
            f"max_risk={max_risk_per_trade*100:.1f}%, "
            f"max_drawdown={max_drawdown*100:.1f}%"
        )
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        portfolio_value: float = None
    ) -> float:
        """Calculate safe position size using risk limit"""
        portfolio = portfolio_value or self.portfolio_value
        
        risk_amount = portfolio * self.max_risk_per_trade
        price_risk = entry_price - stop_loss_price
        
        if price_risk <= 0:
            raise ValueError("Stop loss must be below entry price")
        
        position_size = risk_amount / price_risk
        
        position_value = position_size * entry_price
        max_position_value = portfolio * self.max_position_size
        
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
            logger.warning(
                f"Position size capped at {self.max_position_size*100:.1f}% "
                f"of portfolio"
            )
        
        logger.debug(
            f"Position size: {position_size:.4f} coins "
            f"(Risk: ${risk_amount:.2f})"
        )
        
        return position_size  # FIXED: Now returns in both cases
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        signal_strength: str = "MODERATE"
    ) -> float:
        """Calculate stop loss based on volatility (ATR)"""
        multipliers = {
            "STRONG": 1.5,
            "MODERATE": 2.0,
            "WEAK": 3.0
        }
        
        multiplier = multipliers.get(signal_strength, 2.0)
        stop_loss = entry_price - (atr * multiplier)
        
        logger.debug(
            f"Stop loss: ${stop_loss:.2f} "
            f"({signal_strength}, ATR multiplier: {multiplier}x)"
        )
        
        return max(stop_loss, 0)
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        reward_ratio: float = 2.0
    ) -> float:
        """Calculate take profit based on risk/reward ratio"""
        risk = entry_price - stop_loss
        reward = risk * reward_ratio
        take_profit = entry_price + reward
        
        logger.debug(
            f"Take profit: ${take_profit:.2f} "
            f"(R:R {reward_ratio}:1)"
        )
        
        return take_profit
    
    def assess_trade_risk(
        self,
        entry_price: float,
        stop_loss: float,
        current_pnl: float,
        current_drawdown: float
    ) -> Dict:
        """Assess overall trade risk"""
        risk_amount = entry_price - stop_loss
        risk_percent = (risk_amount / entry_price) * 100
        
        if current_drawdown > self.max_drawdown:
            risk_level = RiskLevel.CRITICAL
        elif risk_percent > 5:
            risk_level = RiskLevel.HIGH
        elif risk_percent > 3:  # FIXED: was risk_level > 3
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        is_safe = (
            current_drawdown < self.max_drawdown and
            risk_percent < 5
        )
        
        return {
            'risk_level': risk_level.value,
            'risk_percent': round(risk_percent, 2),
            'drawdown_percent': round(current_drawdown * 100, 2),
            'max_allowed_drawdown_percent': self.max_drawdown * 100,
            'is_safe_to_trade': is_safe,
            'current_pnl': round(current_pnl, 2)
        }
    
    def update_portfolio_value(self, new_value: float):
        """Update portfolio value"""
        self.portfolio_value = new_value
        logger.debug(f"Portfolio value updated to ${new_value:,.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING RISK MANAGER")
    print("=" * 60)
    
    risk_mgr = RiskManager(portfolio_value=50000)
    
    print("\n1. Calculating position size...")
    position_size = risk_mgr.calculate_position_size(45000, 44000)
    print(f"   Position size: {position_size:.4f} BTC")
    print(f"   (Risk $1,000 / $1,000 per coin)")
    
    print("\n2. Calculating stop loss...")
    stop_loss = risk_mgr.calculate_stop_loss(45000, 500, "STRONG")
    print(f"   Stop loss: ${stop_loss:.2f}")
    
    print("\n3. Calculating take profit...")
    take_profit = risk_mgr.calculate_take_profit(45000, stop_loss, 2.0)
    print(f"   Take profit: ${take_profit:.2f}")
    
    print("\n4. Assessing trade risk...")
    risk_report = risk_mgr.assess_trade_risk(45000, stop_loss, 500, 0.05)
    print(f"   Risk Level: {risk_report['risk_level']}")
    print(f"   Safe to trade: {risk_report['is_safe_to_trade']}")
    
    print("\n" + "=" * 60)
    print("✅ RISK MANAGER TEST COMPLETE!")
    print("=" * 60)