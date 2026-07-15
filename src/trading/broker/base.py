from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class OrderStatus:
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    
class Broker(ABC):
    def __init__(self, api_key: str, api_secret: str, sandbox: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.session = None
        
        logger.info(f"Broker initialized (sandbox={sandbox})")
        
    @abstractmethod
    def connect(self)-> bool:
        pass
    
    @abstractmethod
    def buy(self, symbol: str, quantity: float, price: float)-> Optional[str]:
        pass
    
    @abstractmethod
    def sell(self, symbol: str, quantity: float, price: float)-> Optional[str]:
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str)-> str:
        pass
    
    @abstractmethod
    def get_balance(self, currency: str)-> float:
        pass
    
    @abstractmethod
    def get_positions(self)-> List[Dict]:
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str)-> bool:
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__} | Sandbox: {self.sandbox}"
        
