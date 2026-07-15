"""src/trading/paper_trader.py"""
import logging
from datetime import datetime
from typing import List, Dict, Optional
from .order import Order, OrderSide, OrderType, OrderStatus
from .position import Position, PositionStatus

logger = logging.getLogger(__name__)

class PaperTrader:
    """Paper trading simulator"""
    
    def __init__(self, initial_capital: float, commission_rate: float = 0.001):
        """Initialize paper trader"""
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.commission_rate = commission_rate
        
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.orders: List[Order] = []
        
        self.position_counter = 0
        self.order_counter = 0
        
        self.portfolio_history: Dict[datetime, float] = {}
        
        logger.info(f"PaperTrader initialized with initial capital")
    
    def buy(self, symbol: str, quantity: float, price: float, timestamp: datetime) -> bool:
        """Execute a buy order (open LONG position)"""
        logger.info(f"BUY {quantity} {symbol} @ {price}")
        
        cost = quantity * price
        commission = cost * self.commission_rate
        total_cost = cost + commission
        
        if self.cash_balance < total_cost:
            logger.warning(f"Insufficient cash: need {total_cost}, have {self.cash_balance}")
            return False
        
        self.cash_balance -= total_cost
        
        self.position_counter += 1
        position = Position(
            position_id=self.position_counter,
            symbol=symbol,
            entry_date=timestamp,
            entry_price=price,
            quantity=quantity,
            side=OrderSide.BUY
        )
        self.positions.append(position)
        
        logger.info(f"Position opened: {position}")
        return True
    
    def sell(self, symbol: str, quantity: float, price: float, timestamp: datetime) -> bool:
        """Close a LONG position (sell)"""
        logger.info(f"SELL {quantity} {symbol} @ {price}")
        
        position = self._find_open_position(symbol, OrderSide.BUY)
        if position is None:
            logger.warning(f"No open LONG position found for {symbol}")
            return False
        
        position.close(price, timestamp)
        self.positions.remove(position)
        self.closed_positions.append(position)
        
        proceeds = quantity * price
        commission = proceeds * self.commission_rate
        self.cash_balance += proceeds - commission
        
        logger.info(f"Position closed: {position}")
        return True
    
    def short(self, symbol: str, quantity: float, price: float, timestamp: datetime) -> bool:
        """Open a SHORT position"""
        logger.info(f"SHORT {quantity} {symbol} @ {price}")
        
        proceeds = quantity * price
        commission = proceeds * self.commission_rate
        self.cash_balance += proceeds - commission
        
        self.position_counter += 1
        position = Position(
            position_id=self.position_counter,
            symbol=symbol,
            entry_date=timestamp,
            entry_price=price,
            quantity=quantity,
            side=OrderSide.SELL
        )
        self.positions.append(position)
        
        logger.info(f"SHORT position opened: {position}")
        return True
    
    def cover(self, symbol: str, quantity: float, price: float, timestamp: datetime) -> bool:
        """Close a SHORT position"""
        logger.info(f"COVER {quantity} {symbol} @ {price}")
        
        position = self._find_open_position(symbol, OrderSide.SELL)
        if position is None:
            logger.warning(f"No open SHORT position found for {symbol}")
            return False
        
        cost = quantity * price
        commission = cost * self.commission_rate
        total_cost = cost + commission
        
        if self.cash_balance < total_cost:
            logger.warning(f"Insufficient cash to cover: need {total_cost}, have {self.cash_balance}")
            return False
        
        position.close(price, timestamp)
        self.positions.remove(position)
        self.closed_positions.append(position)
        
        self.cash_balance -= total_cost
        
        logger.info(f"SHORT position covered: {position}")
        return True
    
    def _find_open_position(self, symbol: str, side: OrderSide) -> Optional[Position]:
        """Find first open position matching symbol and side"""
        for pos in self.positions:
            if pos.symbol == symbol and pos.side == side and pos.status == PositionStatus.OPEN:
                return pos
        return None
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return [p for p in self.positions if p.status == PositionStatus.OPEN]
    
    def get_closed_positions(self) -> List[Position]:
        """Get all closed positions"""
        return self.closed_positions
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        value = self.cash_balance
        
        for position in self.get_open_positions():
            if position.symbol in current_prices:
                value += position.current_value(current_prices[position.symbol])
        
        return value
    
    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate total unrealized P&L"""
        total = 0.0
        
        for position in self.get_open_positions():
            if position.symbol in current_prices:
                total += position.unrealized_pnl(current_prices[position.symbol])
        
        return total
    
    def get_realized_pnl(self) -> float:
        """Calculate total realized P&L"""
        total = 0.0
        for p in self.closed_positions:
            if p.pnl is not None:
                total += p.pnl
        return total
    
    def get_total_pnl(self, current_prices: Dict[str, float]) -> float:
        """Get total P&L (realized + unrealized)"""
        return self.get_realized_pnl() + self.get_unrealized_pnl(current_prices)
    
    def get_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get trading summary"""
        closed = self.get_closed_positions()
        winning = sum(1 for p in closed if p.is_winning_trade)
        losing = sum(1 for p in closed if p.is_losing_trade)
        
        win_rate = (winning / len(closed) * 100) if closed else 0.0
        
        return {
            'initial_capital': self.initial_capital,
            'cash_balance': self.cash_balance,
            'open_positions': len(self.get_open_positions()),
            'closed_positions': len(closed),
            'winning_trades': winning,
            'losing_trades': losing,
            'win_rate': win_rate,
            'unrealized_pnl': self.get_unrealized_pnl(current_prices),
            'realized_pnl': self.get_realized_pnl(),
            'total_pnl': self.get_total_pnl(current_prices),
            'portfolio_value': self.get_portfolio_value(current_prices),
            'return_percent': (self.get_total_pnl(current_prices) / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return f"PaperTrader | Cash: {self.cash_balance:,.2f} | Open: {len(self.get_open_positions())} | Closed: {len(self.closed_positions)}"
