import logging
from typing import Dict

logger = logging.getLogger(__name__)

class RiskLimiter:
    
    def __init__(self, initial_capital: float, max_drawdown_pct: float = 10.0, 
                 max_loss_per_trade: float = 2.0, max_position_pct: float = 5.0):
        self.initial_capital = initial_capital
        self.current_balance = initial_capital
        self.peak_balance = initial_capital
        self.max_drawdown_pct = max_drawdown_pct
        self.max_loss_per_trade = max_loss_per_trade
        self.max_position_pct = max_position_pct
        self.daily_loss = 0.0
        logger.info(f"RiskLimiter initialized | Max Drawdown: {max_drawdown_pct}%")
    
    def can_trade(self, symbol: str, quantity: float, price: float) -> bool:
        trade_value = quantity * price
        position_pct = (trade_value / self.current_balance) * 100
        
        if position_pct > self.max_position_pct:
            logger.warning(f"Position too large: {position_pct:.1f}%")
            return False
        
        if self.get_current_drawdown() > self.max_drawdown_pct:
            logger.warning(f"Max drawdown reached: {self.get_current_drawdown():.1f}%")
            return False
        
        if self.daily_loss > (self.initial_capital * self.max_loss_per_trade / 100):
            logger.warning(f"Daily loss limit reached")
            return False
        
        return True
    
    def update_balance(self, new_balance: float) -> None:
        self.current_balance = new_balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
        self.daily_loss = max(0, self.initial_capital - new_balance)
    
    def get_current_drawdown(self) -> float:
        if self.peak_balance == 0:
            return 0.0
        return ((self.peak_balance - self.current_balance) / self.peak_balance) * 100
    
    def get_max_position_size(self) -> float:
        """Get max position size in USD"""
        return (self.current_balance * self.max_position_pct) / 100
    
    def get_summary(self) -> Dict:
        return {
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'current_drawdown_pct': self.get_current_drawdown(),
            'daily_loss': self.daily_loss,
            'max_position_size': self.get_max_position_size()
        }
    
    def __repr__(self) -> str:
        return f"RiskLimiter | Balance: {self.current_balance:,.2f} | Drawdown: {self.get_current_drawdown():.1f}%"