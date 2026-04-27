"""
Real hardware sensor reader via Modbus RTU over RS485.

Reads NPK, EC, pH, moisture, and temperature from industrial soil sensor.
Implements same interface as SensorSim for seamless switching.
"""

import logging
import time
from typing import Optional

from pymodbus.client import ModbusSerialClient as ModbusClient
from pymodbus.exceptions import ModbusException

from data.schema import SensorReading
from io.sensor_sim import SensorReader


logger = logging.getLogger(__name__)


class SensorFailureException(Exception):
    """Raised when sensor fails repeatedly."""
    pass


class SensorReal(SensorReader):
    """Real hardware sensor reader via Modbus RTU."""

    def __init__(self, config: dict):
        """Initialize Modbus connection.

        Args:
            config: System configuration with serial port and baudrate
        """
        self.config = config
        self.port = config['hardware']['modbus_port']
        self.baudrate = config['hardware']['modbus_baudrate']

        # Register map (hardware-specific)
        self.registers = {
            'N': 0x0001,        # scale 0.1
            'P': 0x0002,        # scale 0.1
            'K': 0x0003,        # scale 0.1
            'EC': 0x0010,       # scale 1.0
            'pH': 0x0011,       # scale 0.01
            'moisture': 0x0020, # scale 0.1
            'temp': 0x0021      # scale 0.1
        }

        self.client = ModbusClient(
            method='rtu',
            port=self.port,
            baudrate=self.baudrate,
            timeout=2,
            stopbits=1,
            bytesize=8,
            parity='N'
        )

        self.failure_count = 0
        self.max_failures = 5
        self.last_valid_reading: Optional[SensorReading] = None

        try:
            if not self.client.connect():
                logger.warning(f"Failed to connect to {self.port} at {self.baudrate} baud")
            else:
                logger.info(f"Connected to sensor at {self.port}")
        except Exception as e:
            logger.error(f"Modbus connection error: {e}")

    def read(self) -> SensorReading:
        """Read sensor via Modbus RTU.

        Returns:
            SensorReading from sensor, or last valid reading on failure

        Raises:
            SensorFailureException if max consecutive failures exceeded
        """
        from datetime import datetime

        try:
            # Read all registers
            response = self.client.read_holding_registers(
                address=0x0000,
                count=0x0030,
                slave=1
            )

            if response.isError():
                logger.error(f"Modbus read error: {response}")
                self.failure_count += 1
            else:
                # Extract values with scale factors
                registers = response.registers

                N = registers[self.registers['N']] * 0.1
                P = registers[self.registers['P']] * 0.1
                K = registers[self.registers['K']] * 0.1
                EC = registers[self.registers['EC']] * 1.0
                pH = registers[self.registers['pH']] * 0.01
                moisture = registers[self.registers['moisture']] * 0.1
                temp = registers[self.registers['temp']] * 0.1

                # Create reading
                reading = SensorReading(
                    timestamp=datetime.now(),
                    N=N, P=P, K=K,
                    EC=EC, pH=pH,
                    moisture=moisture, temp=temp,
                    growth_stage=self.config['agronomy']['growth_stages'].get('vegetative', 0)
                )

                self.last_valid_reading = reading
                self.failure_count = 0
                logger.debug(f"Sensor read: N={N:.1f}, EC={EC:.0f}, pH={pH:.2f}")

                return reading

        except ModbusException as e:
            logger.error(f"Modbus exception: {e}")
            self.failure_count += 1
        except Exception as e:
            logger.error(f"Sensor read error: {e}")
            self.failure_count += 1

        # Failure handling
        if self.failure_count >= self.max_failures:
            logger.critical(f"Sensor failed {self.max_failures} times — halting")
            raise SensorFailureException(f"Sensor failure after {self.max_failures} attempts")

        # Return last valid reading if available
        if self.last_valid_reading:
            logger.warning(f"Returning last valid reading (failure #{self.failure_count})")
            return self.last_valid_reading

        # Fallback: return zero reading
        from datetime import datetime
        logger.warning("No valid readings available — returning zeros")
        return SensorReading(
            timestamp=datetime.now(),
            N=0, P=0, K=0, EC=0, pH=7, moisture=50, temp=25,
            growth_stage=0
        )

    def close(self) -> None:
        """Close Modbus connection."""
        if self.client:
            self.client.close()
            logger.info("Modbus connection closed")
