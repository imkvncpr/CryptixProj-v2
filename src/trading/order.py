"""src/trading/order.py"""
from enum import Enum
from datetime import datetime
from typing import List, Tuple, Optional

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class Order:
    
    def __init__(self, order_id: int, symbol: str, side: OrderSide, 
                 order_type: OrderType, quantity: float, price: float, 
                 timestamp: datetime):
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.fill_price = None
        self.fills: List[Tuple[float, float, datetime]] = []
    
    def fill(self, quantity: float, price: float, timestamp: datetime) -> None:
        self.fills.append((quantity, price, timestamp))
        self.filled_quantity += quantity
        
        total_filled_value = sum(q * p for q, p, _ in self.fills)
        self.fill_price = total_filled_value / self.filled_quantity
        
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_quantity = self.quantity
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity
    
    def __repr__(self) -> str:
        fill_pct = (self.filled_quantity / self.quantity * 100) if self.quantity > 0 else 0
        return f"Order #{self.order_id} | {self.side.value} {self.quantity} {self.symbol} @ {self.price} | {self.status.value} ({fill_pct:.0f}%)"
