"""
Feature engineering pipeline for computing derived indicators and temporal features.

Transforms raw sensor readings into feature vectors suitable for LightGBM inference.
"""

import numpy as np
from datetime import datetime
from typing import List, Deque
from collections import deque
import logging

from data.schema import SensorReading, FeatureVector


logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Compute feature vectors from sensor readings with rolling buffer."""

    def __init__(self, buffer_size: int = 6):
        """Initialize feature engineer with rolling buffer.

        Args:
            buffer_size: Number of historical readings to maintain
        """
        self.buffer: Deque[SensorReading] = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        logger.info(f"FeatureEngineer initialized with buffer_size={buffer_size}")

    def compute(self, readings: List[SensorReading]) -> FeatureVector:
        """Compute feature vector from sensor readings.

        Requires at least 1 reading; pads with first reading if buffer not full.

        Args:
            readings: List of SensorReading objects (typically last 1-6)

        Returns:
            Complete FeatureVector ready for model inference
        """
        if not readings:
            raise ValueError("Must provide at least one reading")

        # Ensure we have readings to work with
        if len(readings) < 3:
            # Pad history at the front so the newest supplied reading remains latest.
            readings = [readings[0]] * (3 - len(readings)) + readings

        latest = readings[-1]

        # Extract hour of day for daily periodicity
        hour_of_day = latest.timestamp.hour if hasattr(latest.timestamp, 'hour') else 12

        # Compute temporal derivatives (changes over 1 hour / ~4 readings)
        delta_N = (latest.N - readings[0].N) if len(readings) > 1 else 0.0
        delta_P = (latest.P - readings[0].P) if len(readings) > 1 else 0.0
        delta_K = (latest.K - readings[0].K) if len(readings) > 1 else 0.0
        delta_moisture = (latest.moisture - readings[0].moisture) if len(readings) > 1 else 0.0

        # Rolling averages (over last 3 readings)
        ec_values = [r.EC for r in readings[-3:]]
        rolling_avg_EC = float(np.mean(ec_values))

        n_values = [r.N for r in readings[-3:]]
        rolling_avg_N = float(np.mean(n_values))

        # Derived health indicators
        ec_per_moisture = latest.EC / (latest.moisture + 1e-9)
        pH_error = abs(latest.pH - 5.5)  # Distance from optimal
        N_K_ratio = latest.N / (latest.K + 1e-9)
        moisture_x_EC = latest.moisture * latest.EC

        # Deviation score: normalized distance from optimal ranges
        deviation_score = self._compute_deviation_score(latest)

        return FeatureVector(
            N=float(latest.N),
            P=float(latest.P),
            K=float(latest.K),
            EC=float(latest.EC),
            pH=float(latest.pH),
            moisture=float(latest.moisture),
            temp=float(latest.temp),
            growth_stage=int(latest.growth_stage),
            hour_of_day=int(hour_of_day),
            delta_N=float(delta_N),
            delta_P=float(delta_P),
            delta_K=float(delta_K),
            delta_moisture=float(delta_moisture),
            rolling_avg_EC=float(rolling_avg_EC),
            rolling_avg_N=float(rolling_avg_N),
            EC_per_moisture=float(ec_per_moisture),
            pH_error=float(pH_error),
            N_K_ratio=float(N_K_ratio),
            moisture_x_EC=float(moisture_x_EC),
            deviation_score=float(deviation_score)
        )

    @staticmethod
    def _compute_deviation_score(reading: SensorReading) -> float:
        """Compute scalar deviation score from optimal ranges.

        Args:
            reading: Current sensor reading

        Returns:
            Scalar combining normalized deviations from all optimal ranges
        """
        optimal_ranges = {
            'N': (100, 150),
            'P': (30, 60),
            'K': (150, 200),
            'EC': (600, 1200),
            'pH': (5.0, 6.0),
            'moisture': (55, 75),
            'temp': (22, 32)
        }

        score = 0.0
        for param, (min_val, max_val) in optimal_ranges.items():
            param_value = getattr(reading, param)
            range_size = max_val - min_val
            if param_value < min_val:
                score += ((min_val - param_value) / range_size) ** 2
            elif param_value > max_val:
                score += ((param_value - max_val) / range_size) ** 2

        return float(np.sqrt(score) / len(optimal_ranges))  # Normalized by number of parameters

    @staticmethod
    def feature_names() -> List[str]:
        """Return ordered list of feature names matching to_array() order.

        CRITICAL: Must match exactly with FeatureVector.to_array() order.
        """
        return FeatureVector.feature_names()
