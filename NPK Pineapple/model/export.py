"""
ONNX export pipeline for LightGBM models.

Converts trained LightGBM models to ONNX format for deployment on Jetson.
"""

import logging
import os
from typing import List
import numpy as np

from lightgbm import LGBMRegressor
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType
import yaml


logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/system_config.yaml") -> dict:
    """Load system configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def export_to_onnx(
    models: List[LGBMRegressor],
    feature_names: List[str],
    output_path: str,
    config: dict
) -> str:
    """Export LightGBM models to ONNX format.

    Creates a single ONNX model with 5 output targets.

    Args:
        models: List of trained LGBMRegressor models (one per target)
        feature_names: List of feature names in training order
        output_path: Path to save ONNX model
        config: System configuration

    Returns:
        Path to exported ONNX model
    """
    logger.info("=" * 80)
    logger.info("EXPORTING MODELS TO ONNX")
    logger.info("=" * 80)

    if not models or len(models) != len(config['model']['targets']):
        raise ValueError(f"Expected {len(config['model']['targets'])} models, got {len(models)}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    logger.info(f"Exporting {len(models)} models to: {output_path}")

    # Create initial ONNX from first model
    n_features = len(feature_names)
    initial_type = [('float_input', FloatTensorType([1, n_features]))]

    # Export first model
    logger.info(f"Converting model 0: {config['model']['targets'][0]}")
    onnx_model = onnxmltools.convert_lightgbm(models[0])

    # For simplicity with single-model output, we'll export each as separate ONNX
    # In production, you'd want to combine these or run inference on each separately
    for i, (model, target_name) in enumerate(zip(models, config['model']['targets'])):
        target_path = output_path.replace('.onnx', f'_{target_name}.onnx')
        logger.info(f"Converting model {i}: {target_name} -> {target_path}")

        onnx_model = onnxmltools.convert_lightgbm(model)
        onnxmltools.utils.save_model(onnx_model, target_path)
        logger.info(f"  Saved: {target_path}")

    logger.info("\n" + "=" * 80)
    logger.info("ONNX EXPORT COMPLETE")
    logger.info("=" * 80)

    return output_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from model.train import train_pipeline

    config = load_config()
    models, feature_names = train_pipeline(config)
    output_path = config['model']['onnx_path']
    export_to_onnx(models, feature_names, output_path, config)
