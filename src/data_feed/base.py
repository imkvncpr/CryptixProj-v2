from abc import ABC, abstractmethod
from typing import List
import logging

from ..data_feed.models import OHLCV
from ..data_feed.exceptions import ValidationError

logger = logging.getLogger(__name__)

class DataFeed(ABC):
    @abstractmethod
    def get_ohlcv(self, symbol: str, days: int = 30)-> List[OHLCV]:
        pass
    
    def validate_data(self, data: List[OHLCV])->bool:
        if not data:
            raise ValidationError("Data list cannot be empty")
        
        logger.debug(f"Validating {len(data)} OHLCV candles")
        
        for i, ohlcv in enumerate(data):
            try:
                ohlcv.validate()
            except ValueError as e:
                raise ValidationError(
                    f"OHLCV at index {i} failed validation: {e}"
                )
        
        for i in range(1, len(data)):
            if data[i].timestamp < data[i-1].timestamp:
                raise ValidationError(
                    f"Timestamps not in ascending order at index {i}: "
                    f"{data[i-1].timestamp} > {data[i].timestamp}"
                )
                
        timestamps = [ohlcv.timestamp for ohlcv in data]
        if len(timestamps) != len(set(timestamps)):
            raise ValidationError("Duplicate timestamps found")
        
        logger.debug(f"Validation successful for {len(data)} candles")
        return True
                