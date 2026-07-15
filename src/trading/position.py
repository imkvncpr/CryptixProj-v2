from datetime import datetime
from enum import Enum
from typing import Optional
from .order import OrderSide

class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    
class Position:
    def __init__(self, position_id: int, symbol: str, entry_date: datetime,
                 entry_price: float, quantity: float, side: OrderSide):
        self.position_id = position_id
        self.symbol = symbol
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.quantity = quantity
        self.side = side
        self.entry_value = entry_price * quantity
        
        self.exit_date: Optional[datetime] = None
        self.exit_price: Optional[float] = None
        self.exit_value: Optional[float] = None
        self.pnl: Optional[float] = None
        self.pnl_percent: Optional[float] = None
        self.status = PositionStatus.OPEN
    
    def close(self, exit_price: float, exit_date: datetime) -> None:
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.exit_value = exit_price * self.quantity
        
        if self.side == OrderSide.BUY:
            self.pnl = self.exit_value - self.entry_value
        else:
            self.pnl = self.entry_value - self.exit_value
        
        self.pnl_percent = (self.pnl / self.entry_value) * 100
        self.status = PositionStatus.CLOSED
    
    def current_value(self, current_price: float) -> float:
        return current_price * self.quantity
    
    def unrealized_pnl(self, current_price: float) -> float:
        current_val = self.current_value(current_price)
        
        if self.side == OrderSide.BUY:
            return current_val - self.entry_value
        else:
            return self.entry_value - current_val
    
    def unrealized_pnl_percent(self, current_price: float) -> float:
        pnl = self.unrealized_pnl(current_price)
        return (pnl / self.entry_value) * 100
    
    @property
    def is_winning_trade(self) -> bool:
        if self.status == PositionStatus.OPEN:
            return False
        return self.pnl > 0
    
    @property
    def is_losing_trade(self) -> bool:
        if self.status == PositionStatus.OPEN:
            return False
        return self.pnl < 0
    
    @property
    def duration_days(self) -> float:
        if self.exit_date is None:
            return 0.0
        return (self.exit_date - self.entry_date).days
    
    def __repr__(self) -> str:
        side_str = "LONG" if self.side == OrderSide.BUY else "SHORT"
        status_str = self.status.value
        
        if self.status == PositionStatus.CLOSED:
            return (
                f"Position #{self.position_id} | "
                f"{side_str} {self.quantity} {self.symbol} | "
                f"Entry: {self.entry_price:.2f} | "
                f"Exit: {self.exit_price:.2f} | "
                f"P&L: {self.pnl:.2f} "
                f"({self.pnl_percent:.2f}%) | "
                f"Status: {status_str}"
)
        else:
            return (
                f"Position #{self.position_id} | "
                 "{side_str} {self.quantity} {self.symbol} | "
                 "Entry: {self.entry_price:.2f} | Status: {status_str}"
                )