"""
FRC Fruit Pineapple - Configuration and Utilities Package
"""

from .paths import *
from .jetson_utils import (
    quantize_yolo_model,
    quantize_pytorch_model,
    convert_to_onnx,
    get_model_info
)

__version__ = "1.0.0"
__all__ = [
    "quantize_yolo_model",
    "quantize_pytorch_model",
    "convert_to_onnx",
    "get_model_info",
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "OUTPUTS_DIR",
]
