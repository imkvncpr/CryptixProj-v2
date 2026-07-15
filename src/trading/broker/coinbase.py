import requests
import hmac
import hashlib
import time
from typing import List, Optional, Dict
from datetime import datetime
import json
import logging

from .base import Broker, OrderStatus

logger = logging.getLogger(__name__)

class CoinbaseBroker(Broker):
    def __init__(self, api_key: str, api_secret: str, passphrase: str , sandbox: bool = True):
        super().__init__(api_key, api_secret, sandbox)
        self.passphrase = passphrase
        
        if sandbox:
            self.base_url = "https://api-sandbox.coinbase.com"
        else:
            self.base_url = "https://api.coinbase.com"
            
        self.session = requests.Session()
        logger.info(f"CoinbaseBroker initialized (sandbox={sandbox})")
        
    def _get_auth_headers(self, method: str, path: str, body: str = "")-> Dict[str, str]:
        timestamp = str(time.time())
        message = f"{timestamp}{method}{path}{body}"
        
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        
        import base64
        sig_b64 =  base64.b64encode(signature).decode()
        
        return {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": sig_b64,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json"
        }
        
    def connect(self)-> bool:
        try:
            response = self.session.get(f"{self.base_url}/products")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
        
    def buy(self, symbol: str, quantity: float, price: float)-> Optional[str]:
        logger.info(f"BUY {quantity} {symbol} @ {price}")
        
        path = "/orders"
        body = json.dumps({
            "side": "buy",
            "product_id": f"{symbol}-USD",
            "price": str(price),
            "size": str(quantity)
        })
        
        try:
            headers = self._get_auth_headers("POST", path, body)
            response = self.session.post(
                f"{self.base_url}{path}",
                data=body,
                headers=headers
            )
            
            if response.status_code in [200, 201]:
                order = response.json()
                order_id = order.get('id')
                logger.info(f"Order placed: {order_id}")
                return order_id
            else:
                logger.error(f"Buy failed: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Buy error: {e}")
            return None
        
    def sell(self, symbol: str, quantity: float, price: float)-> Optional[str]:
        logger.info(f"SELL {quantity} {symbol} @ {price}")
        
        path = "/orders"
        body = json.dumps({
            "side": "sell",
            "product_id": f"{symbol}-USD",
            "price": str(price),
            "size": str(quantity)
        })
        
        try:
            headers = self._get_auth_headers("POST", path, body)
            response = self.session.post(
                f"{self.base_url}{path}",
                data=body,
                headers=headers
            )
            
            if response.status_code in [200, 201]:
                order = response.json
                order_id = order.get('id')
                logger.info(f"Order placed: {order_id}")
                return order_id
            else:
                logger.error(f"Sell failed: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Sell error: {e}")
            return None
        
    def get_order_status(self, order_id: str)-> str:
        path = f"/orders/{order_id}"
        
        try:
            headers = self._get_auth_headers("GET", path)
            response = self.session.get(
                f"{self.base_url}{path}",
                headers=headers
            )
            
            if response.status_code == 200:
                order = response.json()
                status = order.get('status')
                return status
            else:
                logger.error(f"Status check failed: {response.text}")
                return OrderStatus.FAILED
        except Exception as e:
            logger.error(f"Status error: {e}")
            return OrderStatus.FAILED
        
    def get_balance(self, currency: str)-> float:
        path = "/accounts"
        
        try:
            headers = self._get_auth_headers("GET", path)
            response = self.session.get(
                f"{self.base_url}{path}",
                headers=headers
            )
            
            if response.status_code == 200:
                accounts = response.json()
                for account in accounts:
                    if account.get('currency') == currency:
                        return float(account.get('available', 0))
                    
            logger.warning(f"Could not find {currency} balance")
            return 0.0
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return 0.0
        
    def get_positions(self) -> List[Dict]:
        path = "/orders?status=open"
        
        try:
            headers = self._get_auth_headers("GET", path)
            response = self.session.get(
                f"{self.base_url}{path}",
                headers=headers
            )
            
            if response.status_code == 200:
                orders = response.json()
                return orders
            
            logger.warning("Could not fetch positions")
            return []
        except Exception as e:
            logger.error(f"Positions error: {e}")
            return []
    
    def cancel_order(self, order_id: str) -> bool:
        path = f"/orders/{order_id}"
        
        try:
            headers = self._get_auth_headers("DELETE", path)
            response = self.session.delete(
                f"{self.base_url}{path}",
                headers=headers
            )
            
            if response.status_code in [200, 404]:
                logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                logger.error(f"Cancel failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Cancel error: {e}")
            return False
    
    def __repr__(self) -> str:
        return f"CoinbaseBroker | Sandbox: {self.sandbox}"
        
        
