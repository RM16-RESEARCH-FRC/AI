"""
SQLite logging backend for sensor readings and control actions.

Persists all readings, actions, and system events for diagnostics and retraining.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path

from data.schema import SensorReading, ActionVector


logger = logging.getLogger(__name__)


class DataLogger:
    """SQLite-backed logger for sensor and action data."""

    def __init__(self, db_path: str = "data/readings.db"):
        """Initialize database connection and create tables if needed.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        self._create_tables()
        logger.info(f"DataLogger initialized: {self.db_path}")

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        # Sensor readings table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                N REAL, P REAL, K REAL,
                EC REAL, pH REAL, moisture REAL, temp REAL,
                growth_stage INTEGER
            )
        """)

        # Actions table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                delta_N REAL, delta_P REAL, delta_K REAL,
                irrigation_ml REAL, pH_adj REAL,
                constrained INTEGER,
                raw_prediction TEXT
            )
        """)

        # System events table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT,
                message TEXT
            )
        """)

        self.conn.commit()

    def log_reading(self, reading: SensorReading) -> None:
        """Log sensor reading to database.

        Args:
            reading: SensorReading object
        """
        try:
            self.cursor.execute("""
                INSERT INTO sensor_readings
                (timestamp, N, P, K, EC, pH, moisture, temp, growth_stage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                reading.timestamp.isoformat(),
                reading.N, reading.P, reading.K,
                reading.EC, reading.pH, reading.moisture, reading.temp,
                reading.growth_stage
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to log reading: {e}")

    def log_action(
        self,
        timestamp: datetime,
        action: ActionVector,
        constrained: bool,
        raw_action: ActionVector = None
    ) -> None:
        """Log control action to database.

        Args:
            timestamp: When action was computed
            action: Control action taken
            constrained: Whether safety layer modified it
            raw_action: Unconstrained model output (for diagnostics)
        """
        try:
            raw_json = None
            if raw_action:
                raw_json = json.dumps(raw_action.to_dict())

            self.cursor.execute("""
                INSERT INTO actions
                (timestamp, delta_N, delta_P, delta_K, irrigation_ml, pH_adj, constrained, raw_prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp.isoformat(),
                action.delta_N, action.delta_P, action.delta_K,
                action.irrigation_ml, action.pH_adj,
                int(constrained),
                raw_json
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to log action: {e}")

    def log_event(self, level: str, message: str) -> None:
        """Log system event.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Event message
        """
        try:
            self.cursor.execute("""
                INSERT INTO system_events (timestamp, level, message)
                VALUES (?, ?, ?)
            """, (datetime.now().isoformat(), level, message))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def export_csv(self, table: str, output_path: str) -> None:
        """Export table to CSV for external analysis.

        Args:
            table: Table name to export
            output_path: CSV file path
        """
        try:
            self.cursor.execute(f"SELECT * FROM {table}")
            rows = self.cursor.fetchall()

            if not rows:
                logger.warning(f"No rows in table {table}")
                return

            # Get column names
            self.cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in self.cursor.fetchall()]

            # Write CSV
            with open(output_path, 'w') as f:
                f.write(','.join(columns) + '\n')
                for row in rows:
                    f.write(','.join(str(v) for v in row) + '\n')

            logger.info(f"Exported {len(rows)} rows from {table} to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")

    def get_latest_reading(self) -> dict:
        """Get most recent sensor reading for diagnostics."""
        try:
            self.cursor.execute("SELECT * FROM sensor_readings ORDER BY id DESC LIMIT 1")
            row = self.cursor.fetchone()
            if row:
                cols = ['id', 'timestamp', 'N', 'P', 'K', 'EC', 'pH', 'moisture', 'temp', 'growth_stage']
                return dict(zip(cols, row))
        except Exception as e:
            logger.error(f"Failed to get latest reading: {e}")
        return None

    def get_reading_count(self) -> int:
        """Get total number of readings logged."""
        try:
            self.cursor.execute("SELECT COUNT(*) FROM sensor_readings")
            return self.cursor.fetchone()[0]
        except Exception:
            return 0

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("DataLogger connection closed")
