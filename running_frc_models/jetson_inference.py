#!/usr/bin/env python3

import argparse
import ast
import math
import random
import socket
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import onnxruntime as ort
import requests


MODEL_DIR = Path("/home/rm/running_frc_models")
FRUIT_MODEL = MODEL_DIR / "detection_model.onnx"
LEAF_MODEL = MODEL_DIR / "leaf_detection.onnx"

FRUIT_CLASSES = ["pineapple"]
LEAF_CLASSES = ["fruit_rot", "healthy", "mealybug_wilt", "root_rot"]
SIM_SENSOR_PERIOD_SECONDS = 60.0

latest_frame = None
latest_jpeg = None
latest_fruit_boxes = []
latest_leaf_boxes = []
lock = threading.Lock()


def iso_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def local_ip_for_dashboard(dashboard_url):
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


def fake_sensors():
    t = time.time() / SIM_SENSOR_PERIOD_SECONDS
    return {
        "N": round(124 + math.sin(t) * 8 + random.uniform(-1, 1), 2),
        "P": round(43 + math.cos(t) * 4 + random.uniform(-1, 1), 2),
        "K": round(176 + math.sin(t) * 12 + random.uniform(-2, 2), 2),
        "EC": round(900 + math.sin(t) * 100 + random.uniform(-15, 15), 2),
        "pH": round(5.55 + math.sin(t) * 0.2 + random.uniform(-0.03, 0.03), 2),
        "moisture": round(64 + math.cos(t) * 7 + random.uniform(-1, 1), 2),
        "temp": round(27 + math.sin(t) * 2 + random.uniform(-0.3, 0.3), 2),
        "growth_stage": 2,
    }


def fake_npk():
    return {
        "delta_N": round(random.uniform(-5, 5), 2),
        "delta_P": round(random.uniform(-2, 2), 2),
        "delta_K": round(random.uniform(-5, 5), 2),
        "irrigation_ml": round(random.uniform(0, 50), 2),
        "pH_adj": round(random.uniform(-0.5, 0.5), 3),
    }


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/usb.mjpg":
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        while True:
            with lock:
                frame = latest_jpeg

            if frame is not None:
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except Exception:
                    break

            time.sleep(0.03)

    def log_message(self, *args):
        pass


def draw_boxes(frame, boxes, color):
    for box in boxes:
        x1, y1, x2, y2 = box["box"]
        label = box["label"]
        conf = box["confidence"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def camera_loop(camera_id, skip_camera=False):
    global latest_frame, latest_jpeg

    if skip_camera:
        print("Camera skipped; publishing a blank test frame")
        while True:
            frame = np.full((480, 640, 3), 32, dtype=np.uint8)
            cv2.putText(
                frame,
                "Camera skipped",
                (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (210, 210, 210),
                2,
            )
            ok, jpeg = cv2.imencode(".jpg", frame)
            if ok:
                with lock:
                    latest_frame = frame.copy()
                    latest_jpeg = jpeg.tobytes()
            time.sleep(0.5)

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Camera failed")
        return

    print("Camera opened")

    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        with lock:
            latest_frame = frame.copy()
            fboxes = list(latest_fruit_boxes)
            lboxes = list(latest_leaf_boxes)

        display = frame.copy()
        draw_boxes(display, fboxes, (0, 255, 0))
        draw_boxes(display, lboxes, (0, 0, 255))

        ok, jpeg = cv2.imencode(".jpg", display)
        if ok:
            with lock:
                latest_jpeg = jpeg.tobytes()

        time.sleep(0.01)


def model_input_size(session):
    shape = session.get_inputs()[0].shape
    target_h = shape[2] if len(shape) > 2 and isinstance(shape[2], int) else 640
    target_w = shape[3] if len(shape) > 3 and isinstance(shape[3], int) else target_h
    return int(target_w), int(target_h)


def preprocess(frame, session):
    target_w, target_h = model_input_size(session)
    original_h, original_w = frame.shape[:2]
    scale = min(target_w / original_w, target_h / original_h)
    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2
    canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized

    image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(image, (2, 0, 1))[None, ...]
    return tensor, {
        "scale": scale,
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "original_w": original_w,
        "original_h": original_h,
    }


def run_model(session, frame):
    tensor, meta = preprocess(frame, session)
    output = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: tensor},
    )[0]
    return output, meta


def parse_class_names(session, fallback):
    raw = session.get_modelmeta().custom_metadata_map.get("names", "")
    try:
        parsed = ast.literal_eval(raw)
        return [str(parsed[i]) for i in sorted(parsed)]
    except Exception:
        return list(fallback)


def nms(boxes, scores, class_ids, iou_threshold=0.45):
    if not boxes:
        return []

    boxes_np = np.array(boxes, dtype=np.float32)
    scores_np = np.array(scores, dtype=np.float32)
    class_np = np.array(class_ids, dtype=np.int32)
    keep = []

    for class_id in np.unique(class_np):
        indexes = np.where(class_np == class_id)[0]
        class_boxes = boxes_np[indexes]
        class_scores = scores_np[indexes]
        x1, y1, x2, y2 = class_boxes[:, 0], class_boxes[:, 1], class_boxes[:, 2], class_boxes[:, 3]
        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = class_scores.argsort()[::-1]

        while order.size > 0:
            current = int(order[0])
            keep.append(int(indexes[current]))
            xx1 = np.maximum(x1[current], x1[order[1:]])
            yy1 = np.maximum(y1[current], y1[order[1:]])
            xx2 = np.minimum(x2[current], x2[order[1:]])
            yy2 = np.minimum(y2[current], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            union = areas[current] + areas[order[1:]] - inter
            iou = inter / np.maximum(union, 1e-9)
            order = order[np.where(iou <= iou_threshold)[0] + 1]

    return sorted(keep, key=lambda idx: scores[idx], reverse=True)


def parse_yolo(output, meta, classes, threshold):
    predictions = np.squeeze(output, axis=0) if output.ndim == 3 and output.shape[0] == 1 else np.squeeze(output)
    if predictions.ndim != 2:
        return []
    expected_columns = len(classes) + 4
    if predictions.shape[0] <= expected_columns and predictions.shape[1] != expected_columns:
        predictions = predictions.T

    boxes = []
    scores = []
    class_ids = []
    scale = meta["scale"]
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]
    frame_w = meta["original_w"]
    frame_h = meta["original_h"]

    for row in predictions:
        class_scores = row[4:]
        if len(class_scores) == 0:
            continue

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if confidence < threshold:
            continue

        cx, cy, width, height = [float(value) for value in row[:4]]
        x1 = (cx - width / 2 - pad_x) / scale
        y1 = (cy - height / 2 - pad_y) / scale
        x2 = (cx + width / 2 - pad_x) / scale
        y2 = (cy + height / 2 - pad_y) / scale

        x1 = max(0.0, min(float(frame_w), x1))
        y1 = max(0.0, min(float(frame_h), y1))
        x2 = max(0.0, min(float(frame_w), x2))
        y2 = max(0.0, min(float(frame_h), y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        label = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
        boxes.append(
            {
                "box": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "label": label,
                "class_name": label,
                "class_id": class_id,
                "confidence": round(confidence, 4),
            }
        )
        scores.append(confidence)
        class_ids.append(class_id)

    return [boxes[index] for index in nms([box["bbox_xyxy"] for box in boxes], scores, class_ids)]


def leaf_severity(label, confidence):
    if label == "healthy":
        return "clear"
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


def ort_providers():
    preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    return [provider for provider in preferred if provider in available] or ["CPUExecutionProvider"]


def inference_loop(url, stream_host, port, interval):
    global latest_fruit_boxes, latest_leaf_boxes

    print("Loading models...")

    providers = ort_providers()
    fruit = ort.InferenceSession(str(FRUIT_MODEL), providers=providers)
    leaf = ort.InferenceSession(str(LEAF_MODEL), providers=providers)
    fruit_classes = parse_class_names(fruit, FRUIT_CLASSES)
    leaf_classes = parse_class_names(leaf, LEAF_CLASSES)

    print(f"Models loaded with providers: {', '.join(providers)}")

    endpoint = url.rstrip("/") + "/api/telemetry"
    stream_url = f"http://{stream_host}:{port}/usb.mjpg"

    while True:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(1)
            continue

        try:
            fruit_raw, fruit_meta = run_model(fruit, frame)
            leaf_raw, leaf_meta = run_model(leaf, frame)
            fruit_boxes = parse_yolo(fruit_raw, fruit_meta, fruit_classes, 0.25)
            leaf_boxes = parse_yolo(leaf_raw, leaf_meta, leaf_classes, 0.35)

            with lock:
                latest_fruit_boxes = fruit_boxes
                latest_leaf_boxes = leaf_boxes

            best_leaf = leaf_boxes[0] if leaf_boxes else None
            payload = {
                "sensors": fake_sensors(),
                "predictions": {"source": "Jetson", "npk_action": fake_npk()},
                "streams": {"usb_cam": stream_url},
                "vision": {
                    "fruit_count": len(fruit_boxes),
                    "ripeness_confidence": max([box["confidence"] for box in fruit_boxes], default=0),
                    "detections": fruit_boxes,
                    "leaf_status": best_leaf["label"] if best_leaf else "none",
                    "leaf_confidence": best_leaf["confidence"] if best_leaf else None,
                    "leaf_severity": leaf_severity(best_leaf["label"], best_leaf["confidence"]) if best_leaf else "clear",
                    "leaf_detection_count": len(leaf_boxes),
                    "leaf_detections": leaf_boxes,
                },
                "system": {"jetson_name": "jetson-agx"},
            }

            requests.post(endpoint, json=payload, timeout=3)
            print(f"Fruit={len(fruit_boxes)} Leaf={len(leaf_boxes)}")

        except Exception as exc:
            print("ERR:", exc)

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard-url", required=True)
    parser.add_argument("--jetson-ip", default="", help="Jetson IP for the dashboard stream URL. Auto-detected when omitted.")
    parser.add_argument("--usb-camera", type=int, default=0)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--interval", type=float, default=8.0)
    parser.add_argument("--skip-camera", action="store_true", help="Publish a blank test frame instead of opening a USB camera.")
    args = parser.parse_args()

    stream_host = args.jetson_ip or local_ip_for_dashboard(args.dashboard_url)
    threading.Thread(target=camera_loop, args=(args.usb_camera, args.skip_camera), daemon=True).start()
    threading.Thread(target=inference_loop, args=(args.dashboard_url, stream_host, args.port, args.interval), daemon=True).start()

    print(f"Stream at http://{stream_host}:{args.port}/usb.mjpg")
    HTTPServer(("0.0.0.0", args.port), StreamHandler).serve_forever()


if __name__ == "__main__":
    main()
