from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
NPK_ROOT = PROJECT_ROOT / "NPK Pineapple"
NPK_MODEL_DIR = NPK_ROOT / "model"
LEAF_MODEL_PATH = PROJECT_ROOT / "LEAF_DETECTION" / "leaf_detection.onnx"
STATIC_DIR = DASHBOARD_DIR / "static"
SIMULATION_TICK_SECONDS = 8.0
SIMULATION_WAVE_SECONDS = 60.0

OPTIMAL_RANGES = {
    "N": [100, 150],
    "P": [30, 60],
    "K": [150, 200],
    "EC": [600, 1200],
    "pH": [5.0, 6.0],
    "moisture": [55, 75],
    "temp": [22, 32],
}

DEFAULT_SENSORS = {
    "N": 124.0,
    "P": 42.0,
    "K": 176.0,
    "EC": 880.0,
    "pH": 5.6,
    "moisture": 63.0,
    "temp": 27.0,
    "growth_stage": 2,
}


@lru_cache(maxsize=1)
def get_leaf_model_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(LEAF_MODEL_PATH),
        "status": "missing",
        "classes": [],
        "input": None,
        "output": None,
        "error": None,
    }
    if not LEAF_MODEL_PATH.exists():
        info["error"] = "LEAF_DETECTION/leaf_detection.onnx was not found."
        return info

    try:
        import ast
        import onnxruntime as ort

        session = ort.InferenceSession(str(LEAF_MODEL_PATH), providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        custom = session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(custom.get("names", "{}"))
        info.update(
            {
                "status": "loaded",
                "classes": [str(value) for _, value in sorted(names.items())],
                "input": {"name": input_meta.name, "shape": input_meta.shape},
                "output": {"name": output_meta.name, "shape": output_meta.shape},
                "task": custom.get("task"),
                "model": custom.get("description"),
            }
        )
    except Exception as exc:
        info["status"] = "unavailable"
        info["error"] = str(exc)
    return info


class TelemetryPayload(BaseModel):
    sensors: dict[str, float | int] | None = None
    predictions: dict[str, Any] | None = None
    vision: dict[str, Any] | None = None
    streams: dict[str, str] | None = None
    system: dict[str, Any] | None = None
    events: list[dict[str, Any] | str] = Field(default_factory=list)


class NpkModelService:
    def __init__(self) -> None:
        self.status = "not_loaded"
        self.error: str | None = None
        self.predictor = None
        self.feature_engineer = None
        self.SensorReading = None
        self.history: deque[Any] = deque(maxlen=6)
        self._load()

    def _load(self) -> None:
        if not NPK_ROOT.exists():
            self.status = "missing"
            self.error = f"NPK project not found at {NPK_ROOT}"
            return

        try:
            sys.path.insert(0, str(NPK_ROOT))
            from data.schema import SensorReading  # type: ignore
            from features.engineer import FeatureEngineer  # type: ignore
            from inference.predictor import Predictor  # type: ignore

            self.SensorReading = SensorReading
            self.feature_engineer = FeatureEngineer(buffer_size=6)
            self.predictor = Predictor(model_dir=str(NPK_MODEL_DIR))

            if not getattr(self.predictor, "sessions", None):
                self.status = "unavailable"
                self.error = "No NPK ONNX sessions loaded. Check onnxruntime and model files."
            else:
                self.status = "loaded"
                self.error = None
        except Exception as exc:  # Keep dashboard usable without ML deps.
            self.status = "unavailable"
            self.error = str(exc)

    def predict(self, sensors: dict[str, Any]) -> dict[str, Any]:
        if self.status != "loaded" or not self.predictor or not self.feature_engineer:
            return {
                "source": "fallback",
                "model_status": self.status,
                "model_error": self.error,
                "npk_action": self._fallback_action(sensors),
            }

        try:
            reading = self.SensorReading(
                timestamp=datetime.now(),
                N=float(sensors.get("N", DEFAULT_SENSORS["N"])),
                P=float(sensors.get("P", DEFAULT_SENSORS["P"])),
                K=float(sensors.get("K", DEFAULT_SENSORS["K"])),
                EC=float(sensors.get("EC", DEFAULT_SENSORS["EC"])),
                pH=float(sensors.get("pH", DEFAULT_SENSORS["pH"])),
                moisture=float(sensors.get("moisture", DEFAULT_SENSORS["moisture"])),
                temp=float(sensors.get("temp", DEFAULT_SENSORS["temp"])),
                growth_stage=int(sensors.get("growth_stage", DEFAULT_SENSORS["growth_stage"])),
            )
            self.history.append(reading)
            features = self.feature_engineer.compute(list(self.history))
            action = self.predictor.predict(features)
            return {
                "source": "NPK Pineapple ONNX",
                "model_status": self.status,
                "model_error": None,
                "npk_action": action.to_dict(),
                "features": {
                    "deviation_score": round(float(features.deviation_score), 4),
                    "pH_error": round(float(features.pH_error), 3),
                    "N_K_ratio": round(float(features.N_K_ratio), 3),
                    "EC_per_moisture": round(float(features.EC_per_moisture), 2),
                },
            }
        except Exception as exc:
            self.status = "runtime_error"
            self.error = str(exc)
            return {
                "source": "fallback",
                "model_status": self.status,
                "model_error": self.error,
                "npk_action": self._fallback_action(sensors),
            }

    @staticmethod
    def _fallback_action(sensors: dict[str, Any]) -> dict[str, float]:
        targets = {"N": 125.0, "P": 45.0, "K": 175.0, "moisture": 65.0, "pH": 5.5}
        return {
            "delta_N": round((targets["N"] - float(sensors.get("N", targets["N"]))) * 0.25, 2),
            "delta_P": round((targets["P"] - float(sensors.get("P", targets["P"]))) * 0.25, 2),
            "delta_K": round((targets["K"] - float(sensors.get("K", targets["K"]))) * 0.25, 2),
            "irrigation_ml": round(max(0.0, targets["moisture"] - float(sensors.get("moisture", targets["moisture"]))) * 18.0, 2),
            "pH_adj": round((targets["pH"] - float(sensors.get("pH", targets["pH"]))) * 0.35, 3),
        }


class DashboardState:
    def __init__(self, model_service: NpkModelService) -> None:
        now = iso_now()
        self.lock = threading.Lock()
        self.model_service = model_service
        self.simulation_enabled = True
        self.state: dict[str, Any] = {
            "updated_at": now,
            "last_jetson_seen_at": None,
            "connection": "simulated",
            "sensors": DEFAULT_SENSORS.copy(),
            "predictions": model_service.predict(DEFAULT_SENSORS.copy()),
            "vision": {
                "leaf_status": "No Jetson data yet",
                "leaf_confidence": None,
                "leaf_severity": "unknown",
                "leaf_detection_count": 0,
                "leaf_detections": [],
                "fruit_count": 0,
                "ripeness": "Unknown",
                "ripeness_confidence": None,
                "estimated_weight_kg": None,
                "detections": [],
            },
            "streams": {
                "usb_cam": "",
            },
            "system": {
                "jetson_name": "waiting-for-jetson",
                "fps_usb": None,
                "latency_ms": None,
            },
            "events": [
                {"time": now, "level": "info", "message": "Dashboard started in simulation mode."}
            ],
            "history": [],
            "model": {
                "npk_model_dir": str(NPK_MODEL_DIR),
                "status": model_service.status,
                "error": model_service.error,
                "leaf": get_leaf_model_info(),
            },
            "optimal_ranges": OPTIMAL_RANGES,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            self._refresh_connection_locked()
            self._refresh_model_locked()
            state = dict(self.state)
            state["events"] = list(self.state["events"])[-30:]
            state["history"] = list(self.state["history"])[-120:]
            state["simulation_enabled"] = self.simulation_enabled
            return state

    def ingest(self, payload: TelemetryPayload, source: str = "jetson") -> dict[str, Any]:
        with self.lock:
            now = iso_now()
            if payload.sensors:
                self.state["sensors"].update(payload.sensors)

            if payload.predictions:
                predictions = dict(payload.predictions)
                predictions.setdefault("source", "jetson")
                predictions.setdefault("model_status", "provided")
                self.state["predictions"] = predictions
            elif payload.sensors:
                self.state["predictions"] = self.model_service.predict(self.state["sensors"])

            if payload.vision:
                self.state["vision"].update(payload.vision)

            if payload.streams:
                self.state["streams"].update(payload.streams)

            if payload.system:
                self.state["system"].update(payload.system)

            for event in payload.events:
                self._add_event_locked(event)

            self.state["updated_at"] = now
            if source == "jetson":
                self.state["last_jetson_seen_at"] = now
                self.state["connection"] = "connected"
                self.simulation_enabled = False

            self._append_history_locked(now)
            self._refresh_model_locked()

        return self.snapshot()

    def toggle_simulation(self, enabled: bool) -> dict[str, Any]:
        with self.lock:
            self.simulation_enabled = enabled
            self.state["connection"] = "simulated" if enabled else "waiting"
            if not enabled:
                self.state["last_jetson_seen_at"] = None
            self._add_event_locked(
                {"level": "info", "message": f"Simulation {'enabled' if enabled else 'disabled'}."}
            )

        return self.snapshot()

    def simulate_tick(self) -> None:
        with self.lock:
            if not self.simulation_enabled:
                return

            t = time.time() / SIMULATION_WAVE_SECONDS
            sensors = {
                "N": 124 + math.sin(t) * 9 + random.uniform(-1.8, 1.8),
                "P": 43 + math.cos(t / 1.4) * 4 + random.uniform(-0.8, 0.8),
                "K": 176 + math.sin(t / 1.7) * 13 + random.uniform(-2.2, 2.2),
                "EC": 895 + math.sin(t / 1.2) * 110 + random.uniform(-18, 18),
                "pH": 5.55 + math.sin(t / 2.0) * 0.23 + random.uniform(-0.03, 0.03),
                "moisture": 64 + math.cos(t / 1.6) * 7 + random.uniform(-1.3, 1.3),
                "temp": 27 + math.sin(t / 2.5) * 2.1 + random.uniform(-0.4, 0.4),
                "growth_stage": 2,
            }
            self.state["sensors"] = {key: round(value, 2) if isinstance(value, float) else value for key, value in sensors.items()}
            self.state["predictions"] = self.model_service.predict(self.state["sensors"])
            self.state["vision"] = {
                "leaf_status": random.choice(["healthy", "healthy", "mealybug_wilt", "fruit_rot", "root_rot"]),
                "leaf_confidence": round(random.uniform(0.74, 0.96), 2),
                "leaf_severity": random.choice(["clear", "clear", "medium", "high"]),
                "leaf_detection_count": random.choice([0, 1, 1, 2]),
                "leaf_detections": [],
                "fruit_count": random.choice([1, 1, 2, 3]),
                "ripeness": random.choice(["Semi-ripe", "Ripe", "Unripe"]),
                "ripeness_confidence": round(random.uniform(0.68, 0.94), 2),
                "estimated_weight_kg": round(random.uniform(0.9, 1.8), 2),
                "detections": [],
            }
            self.state["system"].update(
                {
                    "jetson_name": "simulator",
                    "fps_usb": round(random.uniform(18, 28), 1),
                    "latency_ms": round(random.uniform(18, 55), 1),
                }
            )
            now = iso_now()
            self.state["updated_at"] = now
            self.state["connection"] = "simulated"
            self._append_history_locked(now)
            self._refresh_model_locked()

    def _append_history_locked(self, timestamp: str) -> None:
        row = {
            "time": timestamp,
            **{key: self.state["sensors"].get(key) for key in ["N", "P", "K", "EC", "pH", "moisture", "temp"]},
        }
        self.state["history"].append(row)
        self.state["history"] = self.state["history"][-120:]

    def _add_event_locked(self, event: dict[str, Any] | str) -> None:
        if isinstance(event, str):
            normalized = {"time": iso_now(), "level": "info", "message": event}
        else:
            normalized = {
                "time": event.get("time") or iso_now(),
                "level": event.get("level", "info"),
                "message": event.get("message", ""),
            }
        if normalized["message"]:
            self.state["events"].append(normalized)
            self.state["events"] = self.state["events"][-30:]

    def _refresh_connection_locked(self) -> None:
        if self.simulation_enabled:
            self.state["connection"] = "simulated"
            return

        last_seen = self.state.get("last_jetson_seen_at")
        if not last_seen:
            self.state["connection"] = "waiting"
            return

        try:
            last_dt = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - last_dt).total_seconds()
            self.state["connection"] = "stale" if age > 15 else "connected"
        except ValueError:
            self.state["connection"] = "unknown"

    def _refresh_model_locked(self) -> None:
        self.state["model"] = {
            "npk_model_dir": str(NPK_MODEL_DIR),
            "status": self.model_service.status,
            "error": self.model_service.error,
            "leaf": get_leaf_model_info(),
        }


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


model_service = NpkModelService()
dashboard_state = DashboardState(model_service)
app = FastAPI(title="FARMBOT AI Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
def get_state() -> dict[str, Any]:
    return dashboard_state.snapshot()


@app.post("/api/telemetry")
def post_telemetry(payload: TelemetryPayload) -> dict[str, Any]:
    return dashboard_state.ingest(payload, source="jetson")


@app.post("/api/simulation/{enabled}")
def set_simulation(enabled: bool) -> dict[str, Any]:
    return dashboard_state.toggle_simulation(enabled)


def simulation_loop() -> None:
    while True:
        dashboard_state.simulate_tick()
        time.sleep(SIMULATION_TICK_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FARMBOT AI PC dashboard.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard port.")
    parser.add_argument("--no-sim", action="store_true", help="Start with simulation disabled.")
    args = parser.parse_args()

    if args.no_sim:
        dashboard_state.toggle_simulation(False)

    thread = threading.Thread(target=simulation_loop, daemon=True)
    thread.start()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
