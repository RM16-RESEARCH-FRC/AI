"""
Main control loop scheduler using APScheduler.

Orchestrates sense → think → act → log cycles at regular intervals.
"""

import logging
import time
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor

from data.schema import SystemState
from features.engineer import FeatureEngineer
from inference.predictor import Predictor
from inference.kalman import KalmanSmoother
from control.safety import SafetyLayer
from data.logger import DataLogger


logger = logging.getLogger(__name__)


class ControlLoop:
    """Main control loop for fertigation system."""

    def __init__(
        self,
        config: dict,
        sensor_reader,
        predictor: Predictor,
        actuator,
        dashboard
    ):
        """Initialize control loop with all components.

        Args:
            config: System configuration
            sensor_reader: SensorReader instance (sim or real)
            predictor: Predictor instance (ONNX runtime)
            actuator: ActuatorController instance (sim or Jetson)
            dashboard: Dashboard monitor instance
        """
        self.config = config
        self.sensor = sensor_reader
        self.predictor = predictor
        self.actuator = actuator
        self.dashboard = dashboard

        self.kalman = KalmanSmoother(process_noise_q=0.01, measurement_noise_r=0.1)
        self.feature_engineer = FeatureEngineer(buffer_size=6)
        self.safety = SafetyLayer(config)
        self.logger_db = DataLogger(config['system']['log_db_path'])

        self.cycle_count = 0
        self.start_time = datetime.now()
        self.last_cycle_time = 0.0

        logger.info("ControlLoop initialized")

    def run_cycle(self) -> None:
        """Execute one complete sense → think → act → log cycle.

        Any exception is caught, logged, and the loop continues.
        """
        cycle_start_time = time.time()

        try:
            # 1. Read sensors
            raw_reading = self.sensor.read()

            # 2. Smooth with Kalman filter
            smooth_reading = self.kalman.update(raw_reading)

            # 3. Add to feature buffer
            reading_list = [smooth_reading]
            features = self.feature_engineer.compute(reading_list)

            # 4. Model prediction
            raw_action = self.predictor.predict(features)

            # 5. Safety enforcement
            safe_action, was_constrained = self.safety.enforce(smooth_reading, raw_action)

            # 6. Execute actuators
            self.actuator.execute(safe_action)

            # 7. Feed back to simulator (if in sim mode)
            if hasattr(self.sensor, 'inject_action'):
                self.sensor.inject_action(safe_action)

            # 8. Log everything
            self.logger_db.log_reading(raw_reading)
            self.logger_db.log_action(datetime.now(), safe_action, was_constrained, raw_action)

            # 9. Update dashboard
            self.dashboard.update(smooth_reading, safe_action, was_constrained)

            # Compute cycle time
            self.last_cycle_time = time.time() - cycle_start_time
            self.cycle_count += 1

            if self.cycle_count % 10 == 0:
                logger.info(f"Cycle {self.cycle_count}: {self.last_cycle_time*1000:.1f}ms "
                           f"({self.logger_db.get_reading_count()} readings logged)")

        except Exception as e:
            logger.error(f"Cycle {self.cycle_count} failed: {e}", exc_info=True)
            self.logger_db.log_event("ERROR", f"Cycle failed: {str(e)}")

    def schedule_and_run(self, interval_minutes: int = 15) -> None:
        """Schedule control loop to run periodically.

        Args:
            interval_minutes: How often to run cycles (default 15 min)
        """
        logger.info(f"Starting scheduler: {interval_minutes}-minute cycle interval")

        scheduler = BlockingScheduler(
            executors={'default': ThreadPoolExecutor(max_workers=1)},
            job_defaults={'coalesce': True, 'max_instances': 1}
        )

        scheduler.add_job(
            self.run_cycle,
            'interval',
            minutes=interval_minutes,
            id='control_loop'
        )

        try:
            logger.info("Control loop running (press Ctrl+C to stop)")
            scheduler.start()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            scheduler.shutdown()
            self.shutdown()

    def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down control loop...")

        # Emergency halt
        self.actuator.halt()

        # Flush logger
        self.logger_db.close()

        # Compute uptime
        uptime = datetime.now() - self.start_time
        logger.info(f"Shutdown complete. Uptime: {uptime}")
        logger.info(f"Total cycles: {self.cycle_count}")
        logger.info(f"Total readings logged: {self.logger_db.get_reading_count()}")
