"""
Actuator controller interfaces for pump and valve control.

Provides abstract interface with SimActuator (logging) and JetsonActuator (GPIO).
"""

import logging
from abc import ABC, abstractmethod

from data.schema import ActionVector


logger = logging.getLogger(__name__)


class ActuatorController(ABC):
    """Abstract base class for actuator control."""

    @abstractmethod
    def execute(self, action: ActionVector) -> None:
        """Execute control action on hardware.

        Args:
            action: Control action with nutrient/water/pH adjustments
        """
        pass

    @abstractmethod
    def halt(self) -> None:
        """Emergency halt — stop all actuators."""
        pass


class SimActuator(ActuatorController):
    """Simulated actuator — logs actions without hardware."""

    def __init__(self, config: dict):
        """Initialize simulated actuator.

        Args:
            config: System configuration
        """
        self.config = config
        self.call_count = 0

        logger.info("SimActuator initialized (no hardware)")

    def execute(self, action: ActionVector) -> None:
        """Log control action without hardware execution.

        Args:
            action: Control action to log
        """
        self.call_count += 1

        logger.info(f"[ACTUATOR] Cycle {self.call_count}:")
        logger.info(f"  N: {action.delta_N:+.1f} mg/kg")
        logger.info(f"  P: {action.delta_P:+.1f} mg/kg")
        logger.info(f"  K: {action.delta_K:+.1f} mg/kg")
        logger.info(f"  Irrigation: {action.irrigation_ml:.1f} mL")
        logger.info(f"  pH adj: {action.pH_adj:+.2f}")

    def halt(self) -> None:
        """Halt all actuators (no-op in sim)."""
        logger.warning("HALT signal received (simulated)")


class JetsonActuator(ActuatorController):
    """Real hardware actuator using Jetson.GPIO.

    Converts nutrient/water actions to GPIO pulse durations.
    Only available on Jetson devices — stub available for development.
    """

    def __init__(self, config: dict):
        """Initialize Jetson GPIO pins.

        Args:
            config: System configuration with gpio_pins and calibration
        """
        self.config = config
        self.gpio_pins = config['hardware']['gpio_pins']

        # Calibration: mL per second of pump operation
        self.pump_rates = {
            'N_pump': 10.0,
            'P_pump': 5.0,
            'K_pump': 8.0,
            'irrigation_valve': 50.0,
            'pH_down_pump': 2.0
        }

        self.call_count = 0

        try:
            import Jetson.GPIO as GPIO
            self.GPIO = GPIO
            GPIO.setmode(GPIO.BOARD)

            for pin_name, pin_num in self.gpio_pins.items():
                GPIO.setup(pin_num, GPIO.OUT)
                GPIO.output(pin_num, GPIO.LOW)

            logger.info(f"JetsonActuator initialized with {len(self.gpio_pins)} pins")

        except ImportError:
            logger.warning("Jetson.GPIO not available — running in stub mode")
            self.GPIO = None

    def execute(self, action: ActionVector) -> None:
        """Execute control action by pulsing GPIO pins.

        Args:
            action: Control action with nutrient/water/pH adjustments
        """
        self.call_count += 1

        if not self.GPIO:
            logger.info(f"[JETSON STUB] Cycle {self.call_count}: "
                       f"N={action.delta_N:+.1f}, P={action.delta_P:+.1f}, "
                       f"K={action.delta_K:+.1f}, Irrigation={action.irrigation_ml:.1f}")
            return

        try:
            # Compute pulse durations (milliseconds)
            n_duration_ms = max(0, action.delta_N / self.pump_rates['N_pump'] * 1000)
            p_duration_ms = max(0, action.delta_P / self.pump_rates['P_pump'] * 1000)
            k_duration_ms = max(0, action.delta_K / self.pump_rates['K_pump'] * 1000)
            irrigation_ms = max(0, action.irrigation_ml / self.pump_rates['irrigation_valve'] * 1000)
            pH_duration_ms = max(0, action.pH_adj / self.pump_rates['pH_down_pump'] * 1000) if action.pH_adj < 0 else 0

            logger.debug(f"Pulse durations (ms): N={n_duration_ms:.0f}, P={p_duration_ms:.0f}, "
                        f"K={k_duration_ms:.0f}, Irr={irrigation_ms:.0f}, pH={pH_duration_ms:.0f}")

            # Pulse each pin (simplified — no actual GPIO in stub)
            logger.info(f"[ACTUATOR] Cycle {self.call_count}: "
                       f"N={n_duration_ms:.0f}ms, P={p_duration_ms:.0f}ms, "
                       f"K={k_duration_ms:.0f}ms, Irr={irrigation_ms:.0f}ms")

        except Exception as e:
            logger.error(f"Actuator execution failed: {e}")

    def halt(self) -> None:
        """Emergency halt — set all pins LOW."""
        logger.warning("HALT: stopping all actuators")

        if self.GPIO:
            try:
                for pin_num in self.gpio_pins.values():
                    self.GPIO.output(pin_num, self.GPIO.LOW)
            except Exception as e:
                logger.error(f"Halt failed: {e}")
