"""
1D Kalman filter for sensor smoothing across multiple channels.

Implements steady-state Kalman filtering for N, P, K, EC, pH, moisture, and temp.
"""

import logging
from typing import Dict
import numpy as np


logger = logging.getLogger(__name__)


class KalmanSmoother:
    """1D Kalman filter for smoothing sensor readings."""

    def __init__(self, process_noise_q: float = 0.01, measurement_noise_r: float = 0.1):
        """Initialize Kalman filters for each sensor channel.

        Args:
            process_noise_q: Process noise variance (system dynamics)
            measurement_noise_r: Measurement noise variance (sensor noise)
        """
        self.Q = process_noise_q
        self.R = measurement_noise_r

        # State: estimated value, P: estimation error covariance
        self.channels = ['N', 'P', 'K', 'EC', 'pH', 'moisture', 'temp']
        self.state: Dict[str, float] = {}
        self.P: Dict[str, float] = {}

        logger.info(f"KalmanSmoother initialized: Q={process_noise_q}, R={measurement_noise_r}")

    def update(self, reading):
        """Update filter with new noisy measurement.

        Args:
            reading: SensorReading object

        Returns:
            SensorReading with smoothed values
        """
        from data.schema import SensorReading

        smoothed_values = {}

        for channel in self.channels:
            measurement = getattr(reading, channel)

            # Initialize state on first call
            if channel not in self.state:
                self.state[channel] = measurement
                self.P[channel] = self.R
                smoothed_values[channel] = measurement
                continue

            # Predict
            x_prior = self.state[channel]
            P_prior = self.P[channel] + self.Q

            # Update (measurement update)
            K_gain = P_prior / (P_prior + self.R)  # Kalman gain
            innovation = measurement - x_prior
            self.state[channel] = x_prior + K_gain * innovation
            self.P[channel] = (1 - K_gain) * P_prior

            smoothed_values[channel] = self.state[channel]

        # Return smoothed reading
        return SensorReading(
            timestamp=reading.timestamp,
            N=smoothed_values['N'],
            P=smoothed_values['P'],
            K=smoothed_values['K'],
            EC=smoothed_values['EC'],
            pH=smoothed_values['pH'],
            moisture=smoothed_values['moisture'],
            temp=smoothed_values['temp'],
            growth_stage=reading.growth_stage
        )
