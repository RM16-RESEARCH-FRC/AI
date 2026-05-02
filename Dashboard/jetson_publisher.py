from __future__ import annotations

import argparse
import json
import math
import platform
import random
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


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
SIM_SENSOR_PERIOD_SECONDS = 60.0


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_sensor_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    sensor_path = Path(path)
    if not sensor_path.exists():
        raise FileNotFoundError(sensor_path)
    return json.loads(sensor_path.read_text(encoding="utf-8"))


def local_ip_for_dashboard(dashboard_url: str) -> str:
    parsed = urlparse(dashboard_url)
    host = parsed.hostname
    if not host:
        return socket.gethostbyname(socket.gethostname())

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((host, parsed.port or 80))
            return probe.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def camera_stream_url(args: argparse.Namespace, path: str, override: str) -> str:
    if override:
        return override

    host = args.camera_host or local_ip_for_dashboard(args.dashboard_url)
    return f"http://{host}:{args.camera_port}{path}"


def simulated_sensors() -> dict[str, float | int]:
    t = time.time() / SIM_SENSOR_PERIOD_SECONDS
    return {
        "N": round(124 + math.sin(t) * 8 + random.uniform(-1.5, 1.5), 2),
        "P": round(43 + math.cos(t / 1.4) * 4 + random.uniform(-0.7, 0.7), 2),
        "K": round(176 + math.sin(t / 1.8) * 12 + random.uniform(-2.0, 2.0), 2),
        "EC": round(900 + math.sin(t / 1.2) * 100 + random.uniform(-15, 15), 2),
        "pH": round(5.55 + math.sin(t / 2.0) * 0.2 + random.uniform(-0.03, 0.03), 2),
        "moisture": round(64 + math.cos(t / 1.6) * 7 + random.uniform(-1.0, 1.0), 2),
        "temp": round(27 + math.sin(t / 2.5) * 2 + random.uniform(-0.3, 0.3), 2),
        "growth_stage": 2,
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    sensors = load_sensor_json(args.sensor_json) if args.sensor_json else simulated_sensors()
    leaf_result = {}
    if args.leaf_image:
        from leaf_inference import LeafDiseaseDetector

        leaf_result = LeafDiseaseDetector(args.leaf_model_path).predict_image(args.leaf_image)

    return {
        "sensors": sensors or DEFAULT_SENSORS,
        "streams": {
            "usb_cam": camera_stream_url(args, "/usb.mjpg", args.usb_stream_url),
            "depth_cam": camera_stream_url(args, "/depth.mjpg", args.depth_stream_url),
        },
        "vision": {
            "leaf_status": args.leaf_status,
            "leaf_confidence": args.leaf_confidence,
            "leaf_severity": args.leaf_severity,
            "leaf_detection_count": args.leaf_detection_count,
            "fruit_count": args.fruit_count,
            "ripeness": args.ripeness,
            "ripeness_confidence": args.ripeness_confidence,
            "estimated_weight_kg": args.estimated_weight_kg,
            **leaf_result,
        },
        "system": {
            "jetson_name": args.jetson_name or platform.node() or "jetson",
            "publisher_time": iso_now(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish Jetson telemetry to the PC dashboard.")
    parser.add_argument("--dashboard-url", required=True, help="Example: http://PC_IP:8000")
    parser.add_argument("--interval", type=float, default=8.0, help="Seconds between telemetry posts.")
    parser.add_argument("--sensor-json", help="Optional JSON file containing N/P/K/EC/pH/moisture/temp values.")
    parser.add_argument("--camera-host", default="", help="Jetson camera host/IP. Auto-detected when omitted.")
    parser.add_argument("--camera-port", type=int, default=8090, help="Jetson camera stream server port.")
    parser.add_argument("--usb-stream-url", default="", help="Override URL for USB stream.")
    parser.add_argument("--depth-stream-url", default="", help="Override URL for depth stream.")
    parser.add_argument("--jetson-name", default="")
    parser.add_argument("--leaf-status", default="pending")
    parser.add_argument("--leaf-confidence", type=float)
    parser.add_argument("--leaf-severity", default="unknown")
    parser.add_argument("--leaf-detection-count", type=int, default=0)
    parser.add_argument("--leaf-image", help="Optional image path for one-shot leaf model inference before publishing.")
    parser.add_argument("--leaf-model-path", default="../LEAF_DETECTION/leaf_detection.onnx")
    parser.add_argument("--fruit-count", type=int, default=0)
    parser.add_argument("--ripeness", default="pending")
    parser.add_argument("--ripeness-confidence", type=float)
    parser.add_argument("--estimated-weight-kg", type=float)
    args = parser.parse_args()

    endpoint = args.dashboard_url.rstrip("/") + "/api/telemetry"
    print(f"Publishing telemetry to {endpoint}")
    while True:
        payload = build_payload(args)
        try:
            response = requests.post(endpoint, json=payload, timeout=5)
            response.raise_for_status()
            print(f"{iso_now()} posted telemetry")
        except requests.RequestException as exc:
            print(f"{iso_now()} post failed: {exc}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
