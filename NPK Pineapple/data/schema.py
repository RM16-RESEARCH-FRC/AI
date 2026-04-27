"""
Data schema definitions and dataclasses for the pineapple AI fertigation system.

All modules use these standardized types instead of raw dicts or tuples.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List
import numpy as np


@dataclass
class SensorReading:
    """Raw sensor reading with timestamp and all measured parameters."""
    timestamp: datetime
    N: float          # mg/kg
    P: float          # mg/kg
    K: float          # mg/kg
    EC: float         # μS/cm
    pH: float         # 0-14
    moisture: float   # %
    temp: float       # °C
    growth_stage: int # 0=vegetative, 1=pre_flowering, 2=fruiting


@dataclass
class FeatureVector:
    """Complete feature vector for model inference.

    Combines raw sensor readings, temporal derivatives, and derived indicators.
    Must be converted to numpy array via to_array() for ONNX inference.
    """
    # Raw sensor readings
    N: float
    P: float
    K: float
    EC: float
    pH: float
    moisture: float
    temp: float
    growth_stage: int
    hour_of_day: int

    # Temporal features (computed from rolling buffer)
    delta_N: float           # 1-hour change in N (mg/kg)
    delta_P: float           # 1-hour change in P
    delta_K: float           # 1-hour change in K
    delta_moisture: float    # 1-hour change in moisture %
    rolling_avg_EC: float    # 3-cycle moving average of EC
    rolling_avg_N: float     # 3-cycle moving average of N

    # Derived health indicators
    EC_per_moisture: float   # EC normalized by moisture (interaction term)
    pH_error: float          # abs(pH - 5.5), distance from mid-optimal
    N_K_ratio: float         # N / (K + 1e-9), nutrient balance
    moisture_x_EC: float     # moisture * EC interaction term
    deviation_score: float   # scalar: total normalized distance from optimal ranges

    def to_array(self) -> np.ndarray:
        """Convert to float32 numpy array in consistent feature order.

        CRITICAL: Order must match model training feature order exactly.
        Returns array of shape (1, n_features) ready for ONNX input.
        """
        features = [
            self.N, self.P, self.K, self.EC, self.pH, self.moisture, self.temp,
            self.growth_stage, self.hour_of_day,
            self.delta_N, self.delta_P, self.delta_K, self.delta_moisture,
            self.rolling_avg_EC, self.rolling_avg_N,
            self.EC_per_moisture, self.pH_error, self.N_K_ratio, self.moisture_x_EC,
            self.deviation_score
        ]
        return np.array([[float(f) for f in features]], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        """Return ordered list of feature names matching to_array() order."""
        return [
            'N', 'P', 'K', 'EC', 'pH', 'moisture', 'temp', 'growth_stage', 'hour_of_day',
            'delta_N', 'delta_P', 'delta_K', 'delta_moisture',
            'rolling_avg_EC', 'rolling_avg_N',
            'EC_per_moisture', 'pH_error', 'N_K_ratio', 'moisture_x_EC', 'deviation_score'
        ]


@dataclass
class ActionVector:
    """Control action output from model (before safety constraints)."""
    delta_N: float        # mg/kg to add (negative means remove via water)
    delta_P: float        # mg/kg to add
    delta_K: float        # mg/kg to add
    irrigation_ml: float  # mL of water to add
    pH_adj: float         # adjustment: negative = acidify, positive = alkalize

    def to_dict(self) -> dict:
        """Convert to dict for JSON logging."""
        return {
            'delta_N': float(self.delta_N),
            'delta_P': float(self.delta_P),
            'delta_K': float(self.delta_K),
            'irrigation_ml': float(self.irrigation_ml),
            'pH_adj': float(self.pH_adj)
        }


@dataclass
class SystemState:
    """Complete system state at one control loop cycle."""
    reading: SensorReading      # Raw sensor data
    features: FeatureVector     # Computed features
    action: ActionVector        # Control action taken
    timestamp: datetime         # When this state was computed
    constrained: bool           # True if safety layer modified the action
    raw_action: ActionVector = None  # Unconstrained model output (for diagnostics)

    def __post_init__(self):
        """Ensure timestamp is set."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
