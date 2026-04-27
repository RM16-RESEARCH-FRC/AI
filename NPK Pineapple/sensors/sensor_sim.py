"""
Simulated sensor reader for closed-loop testing without hardware.

Implements same interface as real sensor for seamless code switching.
"""

import logging
from abc import ABC, abstractmethod

from data.schema import SensorReading, ActionVector
from data.simulator import SoilSimulator


logger = logging.getLogger(__name__)


class SensorReader(ABC):
    """Abstract base class for sensor readers."""

    @abstractmethod
    def read(self) -> SensorReading:
        """Read sensor and return current state."""
        pass


class SensorSim(SensorReader):
    """Simulated sensor using internal soil dynamics."""

    def __init__(self, config: dict, seed: int = 42):
        """Initialize simulated sensor with soil simulator.

        Args:
            config: System configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.simulator = SoilSimulator(seed=seed)
        self.last_action = None

        logger.info("SensorSim initialized")

    def read(self) -> SensorReading:
        """Read simulated sensor.

        If inject_action() was called, uses that action; otherwise uses no-op.

        Returns:
            SensorReading from simulator
        """
        if self.last_action is None:
            self.last_action = ActionVector(
                delta_N=0, delta_P=0, delta_K=0,
                irrigation_ml=0, pH_adj=0
            )

        reading = self.simulator.step(self.last_action, dt_minutes=15)
        self.last_action = None  # Clear for next cycle
        return reading

    def inject_action(self, action: ActionVector) -> None:
        """Inject control action to be applied on next read().

        Args:
            action: ActionVector to apply
        """
        self.last_action = action
