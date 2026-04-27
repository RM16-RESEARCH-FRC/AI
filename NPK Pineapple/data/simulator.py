"""
Soil and plant simulator for generating training data without real hardware.

Physically-grounded simulation of nitrogen, phosphorus, potassium, moisture,
EC, pH, and temperature dynamics over a pineapple growth cycle.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
import logging

from data.schema import SensorReading, ActionVector


logger = logging.getLogger(__name__)


# Depletion rates (mg/kg per hour) by growth stage
N_DEPLETION_RATE = {0: 2.0, 1: 3.5, 2: 5.0}      # vegetative, pre-flower, fruiting
P_DEPLETION_RATE = {0: 0.8, 1: 1.2, 2: 1.8}
K_DEPLETION_RATE = {0: 1.5, 1: 2.5, 2: 4.0}

# Physical constants
MOISTURE_LOSS_COEFF = 0.15      # % per degree per hour
EC_NUTRIENT_COEFF = 0.3         # EC rises per nutrient unit
SENSOR_NOISE_SIGMA = 0.03       # 3% gaussian noise


class SoilSimulator:
    """Physically-grounded soil simulator for closed-loop testing."""

    def __init__(self, seed: int = 42):
        """Initialize simulator with random but reasonable state.

        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)

        # Initialize state near optimal ranges
        self.N = self.rng.uniform(100, 150)
        self.P = self.rng.uniform(30, 60)
        self.K = self.rng.uniform(150, 200)
        self.EC = self.rng.uniform(600, 1200)
        self.pH = self.rng.uniform(5.0, 6.0)
        self.moisture = self.rng.uniform(55, 75)
        self.temp = self.rng.uniform(22, 32)
        self.growth_stage = 0  # Start vegetative

        self.step_count = 0
        self.stage_start_step = 0

        logger.info(f"SoilSimulator initialized: N={self.N:.1f}, P={self.P:.1f}, K={self.K:.1f}, "
                   f"pH={self.pH:.2f}, moisture={self.moisture:.1f}%")

    def step(self, action: ActionVector, dt_minutes: int = 15) -> SensorReading:
        """Simulate one time step with given control action.

        Args:
            action: Control action (nutrient additions, irrigation, pH adjustment)
            dt_minutes: Time step duration in minutes

        Returns:
            SensorReading with new state and sensor noise added
        """
        dt_hours = dt_minutes / 60.0

        # Update growth stage every ~720 steps (180 hours ≈ 7.5 days)
        # Vegetative: 90 days (660 steps), Pre-flower: 30 days (220 steps), Fruiting: 60 days (440 steps)
        if self.step_count - self.stage_start_step > 660:  # 90 days
            self.growth_stage = 1  # Pre-flowering
            if self.step_count - self.stage_start_step > 880:  # 120 days
                self.growth_stage = 2  # Fruiting

        # Nutrient depletion (growth stage dependent)
        stage = int(np.clip(self.growth_stage, 0, 2))
        self.N -= N_DEPLETION_RATE[stage] * dt_hours
        self.P -= P_DEPLETION_RATE[stage] * dt_hours
        self.K -= K_DEPLETION_RATE[stage] * dt_hours

        # Apply nutrient additions from action
        self.N += action.delta_N
        self.P += action.delta_P
        self.K += action.delta_K

        # Moisture dynamics: natural loss from evapotranspiration, gain from irrigation
        self.moisture -= MOISTURE_LOSS_COEFF * (self.temp - 20) * dt_hours
        self.moisture += action.irrigation_ml * 0.1  # ~10% moisture per 100ml irrigation

        # EC dynamics: rises with nutrients, falls with dilution
        nutrient_ec = (self.N / 100 + self.P / 30 + self.K / 150) * EC_NUTRIENT_COEFF
        self.EC = nutrient_ec + (1200 - nutrient_ec) * np.exp(-0.1 * (self.moisture - 60))

        # pH dynamics: natural drift toward 6.2, adjustment from action
        self.pH += 0.01 * (6.2 - self.pH) * dt_hours  # Natural drift
        self.pH += action.pH_adj * 0.5  # Adjustment from additives

        # Temperature: slow drift with small daily variation
        self.temp += 0.1 * (25 - self.temp) * dt_hours + self.rng.normal(0, 0.5)

        # Bound state variables to physical limits
        self.N = np.clip(self.N, 0, 300)
        self.P = np.clip(self.P, 0, 100)
        self.K = np.clip(self.K, 0, 300)
        self.EC = np.clip(self.EC, 0, 2000)
        self.pH = np.clip(self.pH, 3.5, 8.5)
        self.moisture = np.clip(self.moisture, 5, 95)
        self.temp = np.clip(self.temp, 15, 40)

        # Add sensor noise (3% of reading)
        reading = SensorReading(
            timestamp=datetime.now(),
            N=np.clip(self.N + self.rng.normal(0, self.N * SENSOR_NOISE_SIGMA), 0, 300),
            P=np.clip(self.P + self.rng.normal(0, self.P * SENSOR_NOISE_SIGMA), 0, 100),
            K=np.clip(self.K + self.rng.normal(0, self.K * SENSOR_NOISE_SIGMA), 0, 300),
            EC=np.clip(self.EC + self.rng.normal(0, self.EC * SENSOR_NOISE_SIGMA), 0, 2000),
            pH=np.clip(self.pH + self.rng.normal(0, 0.1), 3.5, 8.5),
            moisture=np.clip(self.moisture + self.rng.normal(0, self.moisture * SENSOR_NOISE_SIGMA), 5, 95),
            temp=np.clip(self.temp + self.rng.normal(0, 0.2), 15, 40),
            growth_stage=stage
        )

        self.step_count += 1
        return reading

    def generate_dataset(self, n_steps: int, policy: str = "optimal") -> List[Tuple[SensorReading, ActionVector]]:
        """Generate synthetic training dataset with labeled actions.

        Args:
            n_steps: Number of steps to simulate
            policy: "optimal" for healthy actions, "random" for diversity

        Returns:
            List of (SensorReading, ActionVector) tuples representing optimal corrections
        """
        dataset = []

        for i in range(n_steps):
            # Compute optimal action based on deviations from target ranges
            action = self._compute_optimal_action(policy)

            # Step simulation
            reading = self.step(action, dt_minutes=15)

            dataset.append((reading, action))

            if (i + 1) % 5000 == 0:
                logger.info(f"Generated {i + 1}/{n_steps} training samples")

        logger.info(f"Dataset generation complete: {n_steps} samples")
        return dataset

    def _compute_optimal_action(self, policy: str) -> ActionVector:
        """Compute optimal control action based on current state.

        Args:
            policy: "optimal" for near-optimal actions, "random" for diversity

        Returns:
            ActionVector with recommended corrections
        """
        if policy == "random":
            # 70% optimal, 30% historical random for diversity
            if self.rng.rand() > 0.7:
                return ActionVector(
                    delta_N=self.rng.uniform(-20, 30),
                    delta_P=self.rng.uniform(-10, 15),
                    delta_K=self.rng.uniform(-20, 30),
                    irrigation_ml=self.rng.uniform(0, 300),
                    pH_adj=self.rng.uniform(-0.3, 0.3)
                )

        # Default optimal policy: maintain set points
        optimal_N, optimal_P, optimal_K = 125, 45, 175
        optimal_pH, optimal_moisture = 5.5, 65
        optimal_EC = 900

        delta_N = (optimal_N - self.N) * 0.15  # Proportional correction
        delta_P = (optimal_P - self.P) * 0.15
        delta_K = (optimal_K - self.K) * 0.15

        # Irrigation based on moisture deficit
        if self.moisture < 50:
            irrigation = 200
        elif self.moisture > 75:
            irrigation = 0
        else:
            irrigation = 100

        # pH correction
        pH_error = optimal_pH - self.pH
        pH_adj = pH_error * 0.2

        return ActionVector(
            delta_N=float(np.clip(delta_N, -30, 30)),
            delta_P=float(np.clip(delta_P, -15, 15)),
            delta_K=float(np.clip(delta_K, -30, 30)),
            irrigation_ml=float(np.clip(irrigation, 0, 500)),
            pH_adj=float(np.clip(pH_adj, -0.5, 0.5))
        )
