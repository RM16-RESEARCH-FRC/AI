# Pineapple AI Fertigation System

Complete ML-based nutrient and water control system for *Ananas comosus* (pineapple) cultivation on NVIDIA Jetson devices.

## Features

- **Closed-loop AI control** using LightGBM + ONNX Runtime
- **Zero-code data mode switching** between simulation and real hardware
- **Hard safety constraints** preventing dangerous nutrient/water/pH actions
- **Modbus RTU sensor interface** with Jetson GPIO actuator control
- **Complete logging** to SQLite for diagnostics and retraining
- **Terminal dashboard** for real-time monitoring
- **Fully offline** — no cloud dependencies, runs completely on Jetson

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | LightGBM (multi-output regression) |
| Inference | ONNX Runtime (CPU, Jetson-compatible) |
| Scheduling | APScheduler |
| Sensors | Modbus RTU over RS485 |
| Actuators | Jetson GPIO + relays |
| Data | SQLite |
| Config | YAML |

## Project Structure

```
pineapple_ai/
├── config/system_config.yaml          # All tuneable parameters
├── data/
│   ├── simulator.py                   # Soil + plant simulator
│   ├── schema.py                      # Dataclass definitions
│   └── logger.py                      # SQLite data logger
├── features/engineer.py               # Feature engineering
├── model/
│   ├── train.py                       # LightGBM training
│   ├── export.py                      # ONNX export
│   └── validate.py                    # ONNX validation
├── inference/
│   ├── predictor.py                   # ONNX Runtime wrapper
│   └── kalman.py                      # Sensor smoothing
├── control/
│   ├── safety.py                      # Hard constraints
│   ├── actuator.py                    # GPIO/relay control
│   └── scheduler.py                   # Main control loop
├── io/
│   ├── sensor_sim.py                  # Simulated sensor
│   └── sensor_real.py                 # Real sensor (Modbus)
├── dashboard/monitor.py               # Terminal display
├── main.py                            # Entry point
├── requirements.txt
└── README.md
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Simulation Mode (No Hardware)

```bash
python main.py
```

This automatically:
1. Generates synthetic training data if no model exists
2. Trains LightGBM models
3. Exports to ONNX
4. Runs control loop with simulated sensor
5. Logs all readings and actions to SQLite

### Training Only

```bash
python main.py --train-only
```

Trains models and exits.

### ML Model Results

Training now writes inspectable model results under `model/`:

- `model/model_metrics.csv` — one-row-per-target summary with train/test MAE, RMSE, and R2
- `model/model_report.json` — structured version of the same results plus dataset details
- `model/sample_predictions.csv` — actual vs predicted values for sample test rows
- `model/feature_importance.csv` — LightGBM feature importance by target
- `model/*_model.txt` — trained LightGBM model files

To generate only the LightGBM training results without ONNX export:

```bash
python -m model.train
```

How to read the main metrics:

- **MAE**: average absolute error in the target's unit. Lower is better.
- **RMSE**: error metric that penalizes large misses. Lower is better.
- **R2**: explained variance. `1.0` is perfect; values near `0` are weak.

The demo (`python demo.py`) is a system demonstration with a fixed dummy action policy. It is useful for checking simulator, feature engineering, safety rules, and actuator feedback, but it is **not** proof that the ML models are good. Use the files above for ML model performance.

### Real Hardware Mode

Once you have real sensor hardware (Modbus RTU over RS485):

1. Edit `config/system_config.yaml`:
   ```yaml
   system:
     data_mode: "real"  # Switch from "sim"
   hardware:
     modbus_port: "/dev/ttyUSB0"  # Set correct port
   ```

2. Run system:
   ```bash
   python main.py
   ```

**Zero code changes required** — system automatically uses real sensor and Jetson GPIO.

### Data Export for Retraining

After 2 weeks of real sensor data:

```bash
python main.py --export-data
```

Exports readings and actions to CSV. Retrain with:

```bash
python main.py --train-only
```

Then restart.

## Control Loop Architecture

```
Read Sensor
    ↓
Kalman Filter (smooth noise)
    ↓
Feature Engineering (20 features)
    ↓
ONNX Runtime Inference (5 outputs)
    ↓
Safety Layer (hard constraints)
    ↓
Actuator Control (GPIO pulses)
    ↓
Simulator Feedback / Real Hardware
    ↓
SQLite Logging
    ↓
Dashboard Update
```

## Configuration Parameters

Key tuneable parameters in `config/system_config.yaml`:

### Agronomy

- **Optimal ranges**: N, P, K (mg/kg), EC (μS/cm), pH, moisture (%), temp (°C)
- **Growth stages**: Vegetative (0), Pre-flowering (1), Fruiting (2)
- **Critical limits**: EC max, moisture min, pH range
- **Correction limits**: Max nutrient/water/pH adjustments per cycle

### Model

- **LightGBM hyperparameters**: n_estimators, learning_rate, max_depth, etc.
- **Targets**: Control outputs (delta_N, delta_P, delta_K, irrigation_ml, pH_adj)
- **ONNX path**: Where to save/load models

### System

- **Sensor interval**: 15 minutes (configurable)
- **Data mode**: "sim" or "real"
- **Log database**: SQLite path
- **Log level**: DEBUG, INFO, WARNING, ERROR, CRITICAL

### Hardware

- **Modbus port**: /dev/ttyUSB0 (or COM3 on Windows)
- **GPIO pins**: Jetson pin numbers for N/P/K pumps, irrigation valve, pH control

## Safety Constraints

The SafetyLayer enforces hard rules that **cannot be disabled**:

1. **EC ≥ 1500 μS/cm** → Emergency flush (irrigation only, no nutrients)
2. **Moisture ≤ 15%** → Force full irrigation
3. **pH < 4.5** → Emergency alkalization, block nutrients
4. **pH > 7.0** → Emergency acidification
5. **All actions clipped** to correction_limits_per_cycle
6. **Never negative irrigation** (no water extraction)

## Database Schema

### sensor_readings
- timestamp, N, P, K, EC, pH, moisture, temp, growth_stage

### actions
- timestamp, delta_N, delta_P, delta_K, irrigation_ml, pH_adj, constrained, raw_prediction (JSON)

### system_events
- timestamp, level, message

## Monitoring

Real-time terminal dashboard displays:
- Current sensor readings with colour coding (green=optimal, red=critical)
- Recent control actions
- Safety constraint violations
- System events log

Press Ctrl+C to gracefully shutdown.

## Testing

```bash
pytest tests/test_system.py -v
```

Includes tests for:
- Simulator physical bounds
- Feature engineering consistency
- Safety constraint enforcement
- Full 100-cycle execution

## Model Details

### Training

- Generates 60,000 synthetic samples covering:
  - All growth stages
  - Normal operation + edge cases (stress, EC spikes, drought)
  - Both optimal and random policies (70/30 mix)
- Time-ordered train/test split (80/20) to preserve temporal structure
- 5 independent LightGBM regressors for each target

### Inference

- ONNX Runtime on CPU (< 50ms per cycle on Jetson)
- Kalman filtering for sensor noise
- Feature vector includes 20 engineered indicators
- Warm-up JIT compilation on startup

## Real Hardware Integration

### Sensor (Modbus RTU)

Register map (farmbot_anchor_dataset standard):
- 0x0001: N (scale 0.1)
- 0x0002: P (scale 0.1)
- 0x0003: K (scale 0.1)
- 0x0010: EC (scale 1.0)
- 0x0011: pH (scale 0.01)
- 0x0020: Moisture (scale 0.1)
- 0x0021: Temp (scale 0.1)

Graceful fallback on sensor failure (up to 5 retries, then halt).

### Actuators (Jetson GPIO)

Controlled via GPIO pulse duration:
- N pump: 11 (10 mL/sec calibration)
- P pump: 13 (5 mL/sec)
- K pump: 15 (8 mL/sec)
- Irrigation valve: 16 (50 mL/sec)
- pH adjustment: 18 (2 mL/sec)

Calibration constants configurable in `control/actuator.py`.

## Development Workflow

1. **Develop and test in sim mode** — no hardware needed
2. **When ready for real hardware**: Change `data_mode: "real"` in config
3. **Collect real data** for 2 weeks
4. **Export and retrain** with real data
5. **Deploy updated model** — no code changes

## Known Limitations

- Single Jetson device only (no distributed control)
- Modbus RTU latency ~100ms (acceptable for 15-min cycles)
- LightGBM does not support GPU on Jetson (CPU sufficient for inference scale)
- Growth stage must be set manually in config (not auto-detected)

## Future Enhancements

- Multi-zone control with zone-specific models
- Real-time model retraining with online learning
- Integration with Jetson L4T camera for plant health monitoring
- Mobile app for remote monitoring
- Cloud backup of control decisions (diagnostics only)

## Credits

Built for FARMBOT AI FRC project.

## License

Proprietary — All rights reserved.
