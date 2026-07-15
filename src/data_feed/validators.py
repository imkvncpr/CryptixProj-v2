from typing import List
from datetime import timedelta
import logging

from ..data_feed.models import OHLCV
from ..data_feed.exceptions import ValidationError

logger = logging.getLogger(__name__)


def validate_ohlcv(ohlcv: OHLCV) -> bool:
    try:
        ohlcv.validate()
        logger.debug(f"OHLCV validation passed: {ohlcv.timestamp}")
        return True
    except ValueError as e:
        logger.error(f"OHLCV validation failed: {e}")
        raise ValidationError(f"OHLCV validation failed: {e}")


def validate_ohlcv_list(data: List[OHLCV]) -> bool:
    if not data:
        logger.error("OHLCV list is empty")
        raise ValidationError("OHLCV list cannot be empty")
    
    logger.debug(f"Validating {len(data)} OHLCV candles")
    
    # Validate each candle
    for i, ohlcv in enumerate(data):
        try:
            ohlcv.validate()
        except ValueError as e:
            logger.error(f"Candle {i} validation failed: {e}")
            raise ValidationError(f"Candle at index {i} failed validation: {e}")
    
    # Check timestamp ordering
    for i in range(1, len(data)):
        if data[i].timestamp < data[i-1].timestamp:
            logger.error(f"Timestamps out of order at index {i}")
            raise ValidationError(
                f"Timestamps not in ascending order at index {i}: "
                f"{data[i-1].timestamp} > {data[i].timestamp}"
            )
    
    # Check for duplicate timestamps
    timestamps = [ohlcv.timestamp for ohlcv in data]
    if len(timestamps) != len(set(timestamps)):
        logger.error("Duplicate timestamps detected")
        raise ValidationError("Duplicate timestamps found in data")
    
    logger.debug(f"✅ All {len(data)} candles validated successfully")
    return True


def validate_price_range(
    data: List[OHLCV],
    min_price: float = 0.0001,
    max_price: float = 1000000.0
) -> bool:
    logger.debug(f"Checking price range [{min_price}, {max_price}]")
    
    for i, ohlcv in enumerate(data):
        for price_type, price in [
            ('open', ohlcv.open),
            ('high', ohlcv.high),
            ('low', ohlcv.low),
            ('close', ohlcv.close)
        ]:
            if price < min_price or price > max_price:
                logger.error(
                    f"Price out of range at candle {i}: "
                    f"{price_type}={price}"
                )
                raise ValidationError(
                    f"Candle {i}: {price_type} ({price}) outside range "
                    f"[{min_price}, {max_price}]"
                )
    
    logger.debug("✅ All prices in valid range")
    return True


def validate_volume(
    data: List[OHLCV],
    min_volume: float = 0.0
) -> bool:
    logger.debug(f"Validating volume >= {min_volume}")
    
    for i, ohlcv in enumerate(data):
        if ohlcv.volume < min_volume:
            logger.warning(
                f"Low volume at candle {i}: {ohlcv.volume}"
            )
            raise ValidationError(
                f"Candle {i}: volume ({ohlcv.volume}) "
                f"below minimum ({min_volume})"
            )
    
    logger.debug("✅ All volumes valid")
    return True


def validate_no_gaps(
    data: List[OHLCV],
    expected_interval_hours: int = 1
) -> bool:
    expected_interval = timedelta(hours=expected_interval_hours)
    
    for i in range(1, len(data)):
        actual_interval = data[i].timestamp - data[i-1].timestamp
        
        if actual_interval != expected_interval:
            logger.warning(
                f"Gap detected at candle {i}: "
                f"expected {expected_interval}, got {actual_interval}"
            )
    
    return True


def is_valid_symbol(symbol: str) -> bool:
    if symbol is None or len(symbol) == 0:
        return False
    
    if len(symbol) > 50:
        return False
    
    if not symbol.replace('_', '').isalnum():
        return False
    
    return True


def detect_anomalies(data: List[OHLCV]) -> List[int]:
    if not data or len(data) < 2:
        return []
    
    anomalies = []
    
    # Calculate average range
    ranges = [ohlcv.hl_range for ohlcv in data]
    avg_range = sum(ranges) / len(ranges) if ranges else 0
    
    for i, ohlcv in enumerate(data):
        
        # Check for extreme moves (>3x average)
        if avg_range > 0 and ohlcv.hl_range > 3 * avg_range:
            logger.warning(f"Anomaly detected at candle {i}: extreme move")
            anomalies.append(i)
        
        # Check for zero volume
        if ohlcv.volume == 0:
            logger.warning(f"Anomaly detected at candle {i}: zero volume")
            anomalies.append(i)
        
        # Check for identical OHLC (suspicious)
        if (ohlcv.open == ohlcv.high == ohlcv.low == ohlcv.close):
            logger.warning(f"Anomaly detected at candle {i}: identical OHLC")
            anomalies.append(i)
    
    return anomalies


def check_data_quality(data: List[OHLCV]) -> dict:

    issues = []
    warnings = []
    
    # Basic validation
    try:
        validate_ohlcv_list(data)
    except ValidationError as e:
        issues.append(str(e))
    
    # Price range validation
    try:
        validate_price_range(data)
    except ValidationError as e:
        issues.append(str(e))
    
    # Volume validation
    try:
        validate_volume(data)
    except ValidationError as e:
        issues.append(str(e))
    
    # Gap validation
    try:
        validate_no_gaps(data)
    except ValidationError as e:
        warnings.append(str(e))
    
    # Detect anomalies
    anomalies = detect_anomalies(data)
    if anomalies:
        warnings.append(f"Detected {len(anomalies)} anomalous candles")
    
    # Calculate statistics
    if data:
        all_prices = []
        for ohlcv in data:
            all_prices.extend([ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close])
        
        volumes = [ohlcv.volume for ohlcv in data]
        
        price_min = min(all_prices)
        price_max = max(all_prices)
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        volume_min = min(volumes) if volumes else 0
        volume_max = max(volumes) if volumes else 0
        
        time_start = data[0].timestamp
        time_end = data[-1].timestamp
    else:
        price_min = price_max = avg_volume = volume_min = volume_max = 0
        time_start = time_end = None
    
    # Calculate quality score
    quality_score = 100
    for _ in issues:
        quality_score -= 20  # Major penalty
    for _ in warnings:
        quality_score -= 5   # Minor penalty
    quality_score = max(0, quality_score)
    
    return {
        'is_valid': len(issues) == 0,
        'total_candles': len(data),
        'time_range': (time_start, time_end),
        'price_range': (price_min, price_max),
        'avg_volume': avg_volume,
        'volume_range': (volume_min, volume_max),
        'issues': issues,
        'warnings': warnings,
        'quality_score': quality_score,
        'anomalies': anomalies
    }
        
        

            
