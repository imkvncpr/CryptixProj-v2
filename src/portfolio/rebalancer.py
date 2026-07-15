import numpy as np
from typing import Dict
from datetime import datetime


class PortfolioRebalancer:
    def __init__(self):
        self.last_rebalance_date = None

    @staticmethod
    def check_rebalance_needed(
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        threshold: float = 0.05
    ) -> bool:
        drift = np.abs(current_weights - target_weights)
        max_drift = np.max(drift)

        return max_drift > threshold

    @staticmethod
    def calculate_rebalance_trades(
        current_values: Dict[str, float],
        target_weights: Dict[str, float],
        total_portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calculate the dollar amount to buy (+) or sell (-)
        for each asset in order to reach the target allocation.
        """
        trades = {}

        for asset, target_weight in target_weights.items():
            target_value = target_weight * total_portfolio_value
            current_value = current_values.get(asset, 0.0)

            # Positive -> Buy
            # Negative -> Sell
            trades[asset] = target_value - current_value

        return trades

    def time_based_rebalance(
        self,
        rebalance_frequency: str = "monthly"
    ) -> bool:

        if self.last_rebalance_date is None:
            self.last_rebalance_date = datetime.now()
            return True

        days_since = (datetime.now() - self.last_rebalance_date).days

        frequency_days = {
            "daily": 1,
            "weekly": 7,
            "monthly": 30,
            "quarterly": 90,
            "annually": 365
        }

        threshold = frequency_days.get(
            rebalance_frequency.lower(),
            30
        )

        if days_since >= threshold:
            self.last_rebalance_date = datetime.now()
            return True

        return False

    @staticmethod
    def drift_based_rebalance(
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        max_drift: float = 0.05
    ) -> bool:

        deviation = np.abs(current_weights - target_weights)
        max_deviation = np.max(deviation)

        return max_deviation > max_drift