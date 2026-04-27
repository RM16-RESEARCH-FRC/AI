"""
Main entry point for pineapple AI fertigation system.

Initializes all components, trains models if needed, and starts control loop.
"""

import sys
import os
import logging
import argparse
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.schema import SensorReading, ActionVector, FeatureVector
from data.simulator import SoilSimulator
from features.engineer import FeatureEngineer
from inference.predictor import Predictor
from inference.kalman import KalmanSmoother
from control.safety import SafetyLayer
from control.scheduler import ControlLoop
from control.actuator import SimActuator, JetsonActuator
from sensors.sensor_sim import SensorSim
from dashboard.monitor import Dashboard
from data.logger import DataLogger
from model.train import train_pipeline
from model.export import export_to_onnx


# Setup logging
def setup_logging(config: dict) -> None:
    """Configure logging to file and stdout."""
    log_level = getattr(logging, config['system']['log_level'], logging.INFO)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "pineapple.log"),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str = "config/system_config.yaml") -> dict:
    """Load system configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ensure_model_exists(config: dict) -> None:
    """Check if ONNX model exists; train if not."""
    model_pattern = config['model']['onnx_path'].replace('.onnx', '_*.onnx')
    model_dir = Path(config['model']['onnx_path']).parent

    # Check if any model files exist
    if list(model_dir.glob('pineapple_model_*.onnx')):
        logging.info(f"Models found in {model_dir}")
        return

    logging.warning("No models found — training from scratch...")

    models, feature_names = train_pipeline(config)
    export_to_onnx(models, feature_names, config['model']['onnx_path'], config)

    logging.info("Model training and export complete")


def init_components(config: dict, args):
    """Initialize all system components."""
    logging.info("Initializing system components...")

    # Override data mode if specified
    data_mode = args.mode or config['system']['data_mode']
    logging.info(f"Data mode: {data_mode}")

    # Sensor reader
    if data_mode == "sim":
        sensor = SensorSim(config, seed=42)
    elif data_mode == "real":
        try:
            from sensors.sensor_real import SensorReal
            sensor = SensorReal(config)
        except ImportError:
            logging.warning("sensor_real not available — falling back to sim")
            sensor = SensorSim(config)
    else:
        raise ValueError(f"Unknown data mode: {data_mode}")

    # ONNX predictor
    predictor = Predictor(model_dir="model")
    predictor.warm_up(n_iterations=5)

    # Actuator
    if data_mode == "sim":
        actuator = SimActuator(config)
    else:
        try:
            actuator = JetsonActuator(config)
        except Exception as e:
            logging.warning(f"JetsonActuator failed: {e} — using SimActuator")
            actuator = SimActuator(config)

    # Dashboard
    dashboard = Dashboard(config)

    # Control loop
    control_loop = ControlLoop(config, sensor, predictor, actuator, dashboard)

    logging.info("Component initialization complete")
    return control_loop


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pineapple AI Fertigation System")
    parser.add_argument('--train-only', action='store_true', help='Train model and exit')
    parser.add_argument('--validate', action='store_true', help='Validate ONNX and exit')
    parser.add_argument('--mode', choices=['sim', 'real'], help='Override data mode')
    parser.add_argument('--export-data', action='store_true', help='Export logs to CSV and exit')

    args = parser.parse_args()

    # Load config
    config = load_config()
    setup_logging(config)

    logging.info("=" * 80)
    logging.info("PINEAPPLE AI FERTIGATION SYSTEM")
    logging.info("=" * 80)

    # Handle special modes
    if args.train_only:
        logging.info("Training mode: training model and exiting")
        models, feature_names = train_pipeline(config)
        export_to_onnx(models, feature_names, config['model']['onnx_path'], config)
        logging.info("Training complete")
        return

    if args.export_data:
        logging.info("Exporting data to CSV...")
        db_path = config['system']['log_db_path']
        logger = DataLogger(db_path)

        logger.export_csv('sensor_readings', 'exports/sensor_readings.csv')
        logger.export_csv('actions', 'exports/actions.csv')
        logger.export_csv('system_events', 'exports/system_events.csv')

        logger.close()
        logging.info("Export complete")
        return

    # Normal operation: ensure model exists
    ensure_model_exists(config)

    # Initialize components
    control_loop = init_components(config, args)

    # Run scheduler
    sensor_interval = config['system']['sensor_interval_minutes']
    control_loop.schedule_and_run(interval_minutes=sensor_interval)


if __name__ == "__main__":
    main()
