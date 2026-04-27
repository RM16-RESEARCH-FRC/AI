"""
Safety constraint enforcement layer.

Hard constraint enforcement to prevent dangerous nutrient/water/pH actions.
Applied after model prediction and before actuator execution.
"""

import logging
import numpy as np
from typing import Tuple

from data.schema import SensorReading, ActionVector


logger = logging.getLogger(__name__)


class SafetyLayer:
    """Enforce hard constraints on control actions."""

    def __init__(self, config: dict):
        """Initialize safety layer with configuration limits.

        Args:
            config: System configuration dict with critical_limits and correction_limits_per_cycle
        """
        self.critical_limits = config['agronomy']['critical_limits']
        self.correction_limits = config['agronomy']['correction_limits_per_cycle']
        self.optimal_ranges = config['agronomy']['optimal_ranges']

        logger.info("SafetyLayer initialized")
        logger.info(f"  EC max: {self.critical_limits['EC_max']}")
        logger.info(f"  Moisture min: {self.critical_limits['moisture_min']}")
        logger.info(f"  pH range: [{self.critical_limits['pH_min']}, {self.critical_limits['pH_max']}]")

    def enforce(self, reading: SensorReading, action: ActionVector) -> Tuple[ActionVector, bool]:
        """Apply hard constraints to control action.

        Args:
            reading: Current sensor reading
            action: Model-predicted action (may be unconstrained)

        Returns:
            Tuple of (constrained_action, was_constrained)
            was_constrained=True if any constraint fired
        """
        constrained_action = ActionVector(
            delta_N=action.delta_N,
            delta_P=action.delta_P,
            delta_K=action.delta_K,
            irrigation_ml=action.irrigation_ml,
            pH_adj=action.pH_adj
        )

        was_constrained = False

        # Rule 1: High EC (salt buildup) — emergency flush
        if reading.EC >= self.critical_limits['EC_max']:
            logger.warning(f"EC CRITICAL: {reading.EC:.0f} >= {self.critical_limits['EC_max']} — FLUSH")
            constrained_action.delta_N = 0
            constrained_action.delta_P = 0
            constrained_action.delta_K = 0
            constrained_action.irrigation_ml = 100  # Gentle flush
            was_constrained = True

        # Rule 2: Low moisture — emergency irrigation
        if reading.moisture <= self.critical_limits['moisture_min']:
            logger.warning(f"Moisture CRITICAL: {reading.moisture:.1f}% <= {self.critical_limits['moisture_min']}% — IRRIGATE")
            constrained_action.irrigation_ml = self.correction_limits['max_irrigation_ml']
            was_constrained = True

        # Rule 3: Low pH — emergency alkalization
        if reading.pH < self.critical_limits['pH_min']:
            logger.warning(f"pH CRITICAL: {reading.pH:.2f} < {self.critical_limits['pH_min']} — ALKALIZE")
            constrained_action.pH_adj = self.correction_limits['max_pH_adj']
            constrained_action.delta_N = 0  # Block nutrients during pH fix
            constrained_action.delta_P = 0
            constrained_action.delta_K = 0
            was_constrained = True

        # Rule 4: High pH — emergency acidification
        if reading.pH > self.critical_limits['pH_max']:
            logger.warning(f"pH CRITICAL: {reading.pH:.2f} > {self.critical_limits['pH_max']} — ACIDIFY")
            constrained_action.pH_adj = -self.correction_limits['max_pH_adj']
            was_constrained = True

        # Rule 5: Clip all actions to correction limits (per-cycle max changes)
        constrained_action.delta_N = np.clip(
            constrained_action.delta_N,
            -self.correction_limits['max_N_adj'],
            self.correction_limits['max_N_adj']
        )
        constrained_action.delta_P = np.clip(
            constrained_action.delta_P,
            -self.correction_limits['max_P_adj'],
            self.correction_limits['max_P_adj']
        )
        constrained_action.delta_K = np.clip(
            constrained_action.delta_K,
            -self.correction_limits['max_K_adj'],
            self.correction_limits['max_K_adj']
        )
        constrained_action.irrigation_ml = np.clip(
            constrained_action.irrigation_ml,
            0,
            self.correction_limits['max_irrigation_ml']
        )
        constrained_action.pH_adj = np.clip(
            constrained_action.pH_adj,
            -self.correction_limits['max_pH_adj'],
            self.correction_limits['max_pH_adj']
        )

        # Rule 6: Never allow negative nutrient additions (no extraction)
        # NOTE: Model may predict negative, but we allow it via the limits above
        # If you want to outright forbid, uncomment:
        # constrained_action.delta_N = max(0, constrained_action.delta_N)
        # constrained_action.delta_P = max(0, constrained_action.delta_P)
        # constrained_action.delta_K = max(0, constrained_action.delta_K)

        return constrained_action, was_constrained
