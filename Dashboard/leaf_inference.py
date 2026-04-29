from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "LEAF_DETECTION" / "leaf_detection.onnx"


class LeafDiseaseDetector:
    """Small ONNX Runtime wrapper for the YOLO leaf disease model."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, confidence: float = 0.35, iou: float = 0.45) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou = iou
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.names = self._parse_names(metadata.get("names", ""))

    def predict_array(self, image_bgr: np.ndarray) -> dict[str, Any]:
        original_h, original_w = image_bgr.shape[:2]
        tensor, scale, pad_x, pad_y = self._preprocess(image_bgr)
        raw = self.session.run([self.output.name], {self.input.name: tensor})[0]
        detections = self._postprocess(raw, original_w, original_h, scale, pad_x, pad_y)
        best = detections[0] if detections else None
        return {
            "leaf_status": best["class_name"] if best else "No disease detected",
            "leaf_confidence": best["confidence"] if best else None,
            "leaf_severity": self._severity(best["class_name"], best["confidence"]) if best else "clear",
            "leaf_detection_count": len(detections),
            "leaf_model": self.model_path.name,
            "leaf_detections": detections,
        }

    def predict_image(self, image_path: str | Path) -> dict[str, Any]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image-file inference") from exc

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.predict_array(image)

    def _preprocess(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for leaf inference preprocessing") from exc

        target_h = int(self.input.shape[2] or 640)
        target_w = int(self.input.shape[3] or 640)
        h, w = image_bgr.shape[:2]
        scale = min(target_w / w, target_h / h)
        resized_w, resized_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - resized_w) // 2
        pad_y = (target_h - resized_h) // 2
        canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return tensor[None, ...], scale, float(pad_x), float(pad_y)

    def _postprocess(
        self,
        raw: np.ndarray,
        original_w: int,
        original_h: int,
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> list[dict[str, Any]]:
        predictions = np.squeeze(raw)
        if predictions.ndim != 2:
            return []
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes = []
        scores = []
        class_ids = []
        for row in predictions:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < self.confidence:
                continue

            cx, cy, width, height = [float(v) for v in row[:4]]
            x1 = (cx - width / 2 - pad_x) / scale
            y1 = (cy - height / 2 - pad_y) / scale
            x2 = (cx + width / 2 - pad_x) / scale
            y2 = (cy + height / 2 - pad_y) / scale
            boxes.append([
                max(0.0, min(original_w, x1)),
                max(0.0, min(original_h, y1)),
                max(0.0, min(original_w, x2)),
                max(0.0, min(original_h, y2)),
            ])
            scores.append(score)
            class_ids.append(class_id)

        kept = self._nms(np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32), self.iou)
        detections = []
        for idx in kept:
            class_id = class_ids[idx]
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.names.get(class_id, str(class_id)),
                    "confidence": round(float(scores[idx]), 4),
                    "bbox_xyxy": [round(float(v), 2) for v in boxes[idx]],
                }
            )
        return sorted(detections, key=lambda item: item["confidence"], reverse=True)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        if len(boxes) == 0:
            return []

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / np.maximum(union, 1e-9)
            order = order[np.where(iou <= iou_threshold)[0] + 1]

        return keep

    @staticmethod
    def _parse_names(raw_names: str) -> dict[int, str]:
        try:
            parsed = ast.literal_eval(raw_names)
        except (SyntaxError, ValueError):
            return {}
        return {int(key): str(value) for key, value in parsed.items()}

    @staticmethod
    def _severity(class_name: str, confidence: float) -> str:
        if class_name == "healthy":
            return "clear"
        if confidence >= 0.75:
            return "high"
        if confidence >= 0.5:
            return "medium"
        return "low"
