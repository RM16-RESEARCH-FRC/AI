"""
Terminal dashboard for real-time system monitoring.

Simple text-based display with ANSI colour codes (no external UI library required).
"""

import logging
import os
import sys
from datetime import datetime
from collections import deque
from typing import Optional

from data.schema import SensorReading, ActionVector


logger = logging.getLogger(__name__)


class ANSIColours:
    """ANSI colour codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Foreground colours
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    CYAN = '\033[36m'

    # Background
    BLACK_BG = '\033[40m'


class Dashboard:
    """Terminal-based monitoring dashboard."""

    def __init__(self, config: dict, max_events: int = 5):
        """Initialize dashboard.

        Args:
            config: System configuration with optimal ranges
            max_events: Number of recent events to display
        """
        self.config = config
        self.optimal_ranges = config['agronomy']['optimal_ranges']
        self.critical_limits = config['agronomy']['critical_limits']
        self.events: deque = deque(maxlen=max_events)
        self.cycle_count = 0

        logger.info("Dashboard initialized")

    def update(
        self,
        reading: SensorReading,
        action: ActionVector,
        was_constrained: bool
    ) -> None:
        """Update and display dashboard.

        Args:
            reading: Current sensor reading
            action: Current control action
            was_constrained: Whether safety layer fired
        """
        self.cycle_count += 1

        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')

        # Header
        print(ANSIColours.BOLD + ANSIColours.CYAN + "=" * 80)
        print("PINEAPPLE AI FERTIGATION SYSTEM")
        print("=" * 80 + ANSIColours.RESET)

        # Timestamp
        print(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Cycle: {self.cycle_count}")

        # Sensor readings with colour coding
        print(ANSIColours.BOLD + "\nSENSOR READINGS" + ANSIColours.RESET)
        print("-" * 80)

        self._display_parameter("N (mg/kg)", reading.N, self.optimal_ranges['N'])
        self._display_parameter("P (mg/kg)", reading.P, self.optimal_ranges['P'])
        self._display_parameter("K (mg/kg)", reading.K, self.optimal_ranges['K'])
        self._display_parameter("EC (μS/cm)", reading.EC, self.optimal_ranges['EC'])
        self._display_parameter("pH", reading.pH, self.optimal_ranges['pH'], precision=2)
        self._display_parameter("Moisture (%)", reading.moisture, self.optimal_ranges['moisture'])
        self._display_parameter("Temp (°C)", reading.temp, self.optimal_ranges['temp'])

        growth_stage_names = ['Vegetative', 'Pre-flowering', 'Fruiting']
        print(f"  Growth Stage: {growth_stage_names[min(reading.growth_stage, 2)]}")

        # Recent actions
        print(ANSIColours.BOLD + "\nRECENT ACTION" + ANSIColours.RESET)
        print("-" * 80)

        constraint_indicator = f"{ANSIColours.RED}(CONSTRAINED){ANSIColours.RESET}" if was_constrained else ""
        print(f"  Delta N: {action.delta_N:+.1f} mg/kg {constraint_indicator if was_constrained else ''}")
        print(f"  Delta P: {action.delta_P:+.1f} mg/kg")
        print(f"  Delta K: {action.delta_K:+.1f} mg/kg")
        print(f"  Irrigation: {action.irrigation_ml:.1f} mL")
        print(f"  pH Adjustment: {action.pH_adj:+.2f}")

        # Events log
        print(ANSIColours.BOLD + "\nRECENT EVENTS" + ANSIColours.RESET)
        print("-" * 80)

        if self.events:
            for event in list(self.events)[-5:]:
                print(f"  {event}")
        else:
            print("  (No events)")

        # Footer
        print("\n" + ANSIColours.DIM + "Press Ctrl+C to stop" + ANSIColours.RESET)

    def _display_parameter(
        self,
        name: str,
        value: float,
        optimal_range: list,
        precision: int = 1
    ) -> None:
        """Display parameter with colour coding.

        Args:
            name: Parameter name
            value: Current value
            optimal_range: [min, max] optimal range
            precision: Decimal precision for display
        """
        min_val, max_val = optimal_range

        if value < min_val:
            colour = ANSIColours.RED
            status = "LOW"
        elif value > max_val:
            colour = ANSIColours.YELLOW
            status = "HIGH"
        else:
            colour = ANSIColours.GREEN
            status = "OK"

        format_str = f"{{:.{precision}f}}"
        print(f"  {name}: {colour}{format_str.format(value):>10}{ANSIColours.RESET} "
              f"(optimal: {min_val}-{max_val}) {colour}[{status}]{ANSIColours.RESET}")

    def add_event(self, level: str, message: str) -> None:
        """Add event to log.

        Args:
            level: Event level (INFO, WARNING, ERROR, CRITICAL)
            message: Event message
        """
        timestamp = datetime.now().strftime('%H:%M:%S')

        if level == "ERROR":
            colour = ANSIColours.RED
        elif level == "WARNING":
            colour = ANSIColours.YELLOW
        else:
            colour = ANSIColours.CYAN

        event_str = f"{colour}[{timestamp} {level}] {message}{ANSIColours.RESET}"
        self.events.append(event_str)
