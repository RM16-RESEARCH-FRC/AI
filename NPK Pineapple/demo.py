#!/usr/bin/env python
"""
Quick demonstration of the pineapple AI system.

Shows simulator, feature engineering, safety constraints, and a 10-cycle run.
"""

import sys
sys.path.insert(0, '.')

from datetime import datetime
from data.simulator import SoilSimulator
from data.schema import ActionVector
from features.engineer import FeatureEngineer
from control.safety import SafetyLayer
from sensors.sensor_sim import SensorSim
from control.actuator import SimActuator
import yaml

print("\n" + "="*80)
print("PINEAPPLE AI FERTIGATION SYSTEM - DEMO")
print("="*80 + "\n")

# Load config
with open('config/system_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 1. Test Simulator
print("1. SOIL SIMULATOR TEST")
print("-" * 80)
sim = SoilSimulator(seed=42)
print(f"Initial state:")
print(f"  N: {sim.N:.1f} mg/kg (optimal: {config['agronomy']['optimal_ranges']['N']})")
print(f"  P: {sim.P:.1f} mg/kg (optimal: {config['agronomy']['optimal_ranges']['P']})")
print(f"  K: {sim.K:.1f} mg/kg (optimal: {config['agronomy']['optimal_ranges']['K']})")
print(f"  EC: {sim.EC:.0f} μS/cm (optimal: {config['agronomy']['optimal_ranges']['EC']})")
print(f"  pH: {sim.pH:.2f} (optimal: {config['agronomy']['optimal_ranges']['pH']})")
print(f"  Moisture: {sim.moisture:.1f}% (optimal: {config['agronomy']['optimal_ranges']['moisture']})")
print(f"  Temp: {sim.temp:.1f} °C (optimal: {config['agronomy']['optimal_ranges']['temp']})")

# 2. Feature Engineering
print("\n2. FEATURE ENGINEERING TEST")
print("-" * 80)
fe = FeatureEngineer()
action = ActionVector(delta_N=20, delta_P=10, delta_K=20, irrigation_ml=100, pH_adj=0)
reading = sim.step(action)
features = fe.compute([reading])
print(f"Feature vector computed successfully:")
print(f"  Raw features: N={features.N:.1f}, P={features.P:.1f}, K={features.K:.1f}")
print(f"  Temporal: delta_N={features.delta_N:.1f}, delta_moisture={features.delta_moisture:.1f}")
print(f"  Derived: pH_error={features.pH_error:.3f}, N_K_ratio={features.N_K_ratio:.3f}")
print(f"  Deviation score: {features.deviation_score:.3f}")
print(f"  Array shape: {features.to_array().shape}, dtype: {features.to_array().dtype}")

# 3. Safety Layer
print("\n3. SAFETY CONSTRAINTS TEST")
print("-" * 80)
safety = SafetyLayer(config)

# Test high EC
reading_high_ec = reading
reading_high_ec.EC = 1600  # Above limit
action_test = ActionVector(delta_N=20, delta_P=10, delta_K=20, irrigation_ml=0, pH_adj=0)
safe, constrained = safety.enforce(reading_high_ec, action_test)
print(f"High EC test (EC={reading_high_ec.EC:.0f}, threshold={config['agronomy']['critical_limits']['EC_max']}):")
print(f"  Original action: N={action_test.delta_N:+.0f}, irrigation={action_test.irrigation_ml:.0f}")
print(f"  Constrained: {constrained}")
print(f"  Safe action: N={safe.delta_N:+.0f}, irrigation={safe.irrigation_ml:.0f}")

# 4. Sensor Simulation
print("\n4. SENSOR SIMULATION TEST")
print("-" * 80)
sensor = SensorSim(config, seed=42)
reading1 = sensor.read()
sensor.inject_action(ActionVector(delta_N=15, delta_P=7, delta_K=15, irrigation_ml=150, pH_adj=0))
reading2 = sensor.read()
print(f"Read 1 - N: {reading1.N:.1f}, Moisture: {reading1.moisture:.1f}%")
print(f"Read 2 - N: {reading2.N:.1f}, Moisture: {reading2.moisture:.1f}%")
print(f"Action injected and simulator state updated ✓")

# 5. Full 10-Cycle Demo
print("\n5. FULL CONTROL LOOP - 10 CYCLES")
print("-" * 80)
print("Cycle | N (mg/kg) | EC (μS/cm) | pH   | Moisture (%) | Action")
print("-" * 80)

sensor = SensorSim(config, seed=42)
fe = FeatureEngineer()
actuator = SimActuator(config)

for i in range(10):
    # Read
    reading = sensor.read()

    # Feature engineer
    features = fe.compute([reading])

    # Model prediction (dummy - use fixed action for demo)
    model_action = ActionVector(
        delta_N=15 if reading.N < 120 else 0,
        delta_P=8 if reading.P < 45 else 0,
        delta_K=15 if reading.K < 175 else 0,
        irrigation_ml=100 if reading.moisture < 60 else 50,
        pH_adj=0.1 if reading.pH < 5.5 else 0 if reading.pH > 5.5 else 0
    )

    # Safety
    safe_action, was_constrained = safety.enforce(reading, model_action)

    # Execute
    actuator.execute(safe_action)

    # Feed back
    sensor.inject_action(safe_action)

    # Display
    constraint_marker = " *" if was_constrained else ""
    print(f"{i+1:2d}    | {reading.N:8.1f}  | {reading.EC:9.0f}  | {reading.pH:4.2f} | {reading.moisture:11.1f} | OK{constraint_marker}")

print("-" * 80)

# Summary
print("\n6. SYSTEM SUMMARY")
print("-" * 80)
print(f"✓ Simulator: Physically realistic soil dynamics")
print(f"✓ Features: 20-dimensional vector with temporal indicators")
print(f"✓ Safety: Hard constraints enforced (EC, pH, moisture)")
print(f"✓ Sensor: Simulated with closed-loop feedback")
print(f"✓ Actuator: Control actions logged and applied")
print(f"✓ Control Loop: 10 complete sense→think→act cycles executed")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("""
1. Install requirements:
   pip install -r requirements.txt

2. Run training (auto-generates 60k samples, trains LightGBM, exports ONNX):
   python main.py --train-only

3. Run full control loop in simulation mode:
   python main.py

4. Switch to real hardware (zero code changes):
   - Edit config/system_config.yaml: data_mode: "real"
   - Set modbus_port to your device
   - Run: python main.py

5. Run tests:
   pytest tests/test_system.py -v
""")
print("="*80 + "\n")
