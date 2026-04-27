"""
Test suite for pineapple AI fertigation system.

Tests simulator, features, safety layer, and full cycle execution.
"""

import sys
import os
import pytest
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.schema import SensorReading, ActionVector, FeatureVector
from data.simulator import SoilSimulator
from features.engineer import FeatureEngineer
from control.safety import SafetyLayer
from sensors.sensor_sim import SensorSim
import yaml


# Fixtures
@pytest.fixture
def config():
    """Load test configuration."""
    with open("config/system_config.yaml", 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def simulator():
    """Create sim instance."""
    return SoilSimulator(seed=42)


@pytest.fixture
def feature_engineer():
    """Create feature engineer."""
    return FeatureEngineer(buffer_size=6)


# Tests
class TestSimulator:
    """Test SoilSimulator physical dynamics."""

    def test_simulator_bounds(self, simulator):
        """Test that simulator state stays within physical bounds over many steps."""
        action = ActionVector(
            delta_N=0, delta_P=0, delta_K=0,
            irrigation_ml=100, pH_adj=0
        )

        for i in range(1000):
            reading = simulator.step(action, dt_minutes=15)

            # Check bounds
            assert 0 <= reading.N <= 300
            assert 0 <= reading.P <= 100
            assert 0 <= reading.K <= 300
            assert 0 <= reading.EC <= 2000
            assert 3.5 <= reading.pH <= 8.5
            assert 5 <= reading.moisture <= 95
            assert 15 <= reading.temp <= 40
            assert reading.growth_stage in [0, 1, 2]

    def test_simulator_nutrient_depletion(self, simulator):
        """Test that nutrients deplete without replenishment."""
        initial_N = simulator.N

        # Apply zero nutrient action
        action = ActionVector(
            delta_N=0, delta_P=0, delta_K=0,
            irrigation_ml=100, pH_adj=0
        )

        for _ in range(10):
            simulator.step(action, dt_minutes=15)

        # N should have decreased
        assert simulator.N < initial_N

    def test_dataset_generation(self, simulator):
        """Test dataset generation produces consistent samples."""
        dataset = simulator.generate_dataset(n_steps=1000, policy='optimal')

        assert len(dataset) == 1000

        # Check each sample
        for reading, action in dataset:
            assert isinstance(reading, SensorReading)
            assert isinstance(action, ActionVector)
            assert reading.N > 0
            assert action.irrigation_ml >= 0


class TestFeatureEngineer:
    """Test feature engineering pipeline."""

    def test_feature_vector_shape(self, feature_engineer):
        """Test that feature vector has correct shape."""
        reading = SensorReading(
            timestamp=datetime.now(),
            N=125, P=45, K=175, EC=900, pH=5.5,
            moisture=65, temp=27, growth_stage=0
        )

        features = feature_engineer.compute([reading])

        assert isinstance(features, FeatureVector)
        arr = features.to_array()
        assert arr.shape == (1, 20)  # 20 features
        assert arr.dtype == np.float32

    def test_feature_names_match_array(self, feature_engineer):
        """Test that feature names match array order."""
        names = FeatureVector.feature_names()
        assert len(names) == 20
        assert 'N' in names
        assert 'pH_error' in names
        assert 'deviation_score' in names

    def test_feature_temporal_computation(self, feature_engineer):
        """Test that temporal features are computed correctly."""
        reading1 = SensorReading(
            timestamp=datetime.now(),
            N=100, P=40, K=170, EC=900, pH=5.5,
            moisture=65, temp=27, growth_stage=0
        )

        reading2 = SensorReading(
            timestamp=datetime.now(),
            N=110, P=45, K=175, EC=900, pH=5.5,
            moisture=65, temp=27, growth_stage=0
        )

        features = feature_engineer.compute([reading1, reading2])

        # Delta should reflect change
        assert features.delta_N > 0
        assert features.delta_P > 0
        assert features.delta_K > 0


class TestSafetyLayer:
    """Test safety constraint enforcement."""

    def test_high_ec_flush(self, config):
        """Test that high EC triggers flush."""
        safety = SafetyLayer(config)

        reading = SensorReading(
            timestamp=datetime.now(),
            N=200, P=70, K=250, EC=1600,  # High EC
            pH=5.5, moisture=65, temp=27, growth_stage=0
        )

        action = ActionVector(
            delta_N=20, delta_P=10, delta_K=20,
            irrigation_ml=0, pH_adj=0
        )

        constrained, was_constrained = safety.enforce(reading, action)

        assert was_constrained
        assert constrained.delta_N == 0
        assert constrained.delta_P == 0
        assert constrained.delta_K == 0
        assert constrained.irrigation_ml > 0

    def test_low_moisture_irrigation(self, config):
        """Test that low moisture forces irrigation."""
        safety = SafetyLayer(config)

        reading = SensorReading(
            timestamp=datetime.now(),
            N=125, P=45, K=175, EC=900, pH=5.5,
            moisture=10,  # Low
            temp=27, growth_stage=0
        )

        action = ActionVector(
            delta_N=0, delta_P=0, delta_K=0,
            irrigation_ml=0, pH_adj=0
        )

        constrained, was_constrained = safety.enforce(reading, action)

        assert was_constrained
        assert constrained.irrigation_ml > 0

    def test_low_pH_alkalization(self, config):
        """Test that low pH triggers alkalization."""
        safety = SafetyLayer(config)

        reading = SensorReading(
            timestamp=datetime.now(),
            N=125, P=45, K=175, EC=900, pH=4.2,  # Low
            moisture=65, temp=27, growth_stage=0
        )

        action = ActionVector(
            delta_N=20, delta_P=10, delta_K=20,
            irrigation_ml=100, pH_adj=0
        )

        constrained, was_constrained = safety.enforce(reading, action)

        assert was_constrained
        assert constrained.pH_adj > 0

    def test_action_clipping(self, config):
        """Test that actions are clipped to limits."""
        safety = SafetyLayer(config)

        reading = SensorReading(
            timestamp=datetime.now(),
            N=125, P=45, K=175, EC=900, pH=5.5,
            moisture=65, temp=27, growth_stage=0
        )

        action = ActionVector(
            delta_N=1000,  # Way too high
            delta_P=1000,
            delta_K=1000,
            irrigation_ml=2000,
            pH_adj=10
        )

        constrained, was_constrained = safety.enforce(reading, action)

        assert constrained.delta_N <= config['agronomy']['correction_limits_per_cycle']['max_N_adj']
        assert constrained.delta_P <= config['agronomy']['correction_limits_per_cycle']['max_P_adj']
        assert constrained.irrigation_ml <= config['agronomy']['correction_limits_per_cycle']['max_irrigation_ml']


class TestSensorSim:
    """Test simulated sensor."""

    def test_sensor_read(self, config):
        """Test that sensor can read and return SensorReading."""
        sensor = SensorSim(config)

        reading = sensor.read()

        assert isinstance(reading, SensorReading)
        assert reading.N > 0
        assert reading.timestamp is not None

    def test_sensor_action_injection(self, config):
        """Test that sensor action injection works."""
        sensor = SensorSim(config)

        action = ActionVector(
            delta_N=30, delta_P=15, delta_K=30,
            irrigation_ml=200, pH_adj=0.2
        )

        sensor.inject_action(action)
        reading = sensor.read()

        # State should reflect the action
        assert isinstance(reading, SensorReading)


class TestFullCycle:
    """Test complete sense-think-act cycle."""

    def test_100_cycles_no_crash(self, config):
        """Test that 100 full cycles run without crashing."""
        from control.actuator import SimActuator
        from dashboard.monitor import Dashboard

        # Create minimal components
        sensor = SensorSim(config)
        actuator = SimActuator(config)
        dashboard = Dashboard(config)

        # For this test, we'll skip predictor (would need trained model)
        # Just test the sensor and safety layer feedback loop

        safety = SafetyLayer(config)
        feature_engineer = FeatureEngineer()

        reading_count = 0

        for cycle in range(100):
            # Read sensor
            reading = sensor.read()

            # Compute features
            features = feature_engineer.compute([reading])

            # Dummy prediction (replace with real predictor if trained)
            action = ActionVector(
                delta_N=10, delta_P=5, delta_K=10,
                irrigation_ml=50, pH_adj=0
            )

            # Apply safety
            safe_action, was_constrained = safety.enforce(reading, action)

            # Execute
            actuator.execute(safe_action)

            # Feed back to simulator
            sensor.inject_action(safe_action)

            reading_count += 1

        assert reading_count == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
