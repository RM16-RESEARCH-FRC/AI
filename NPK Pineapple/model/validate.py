"""
ONNX model validation pipeline.

Compares ONNX Runtime inference outputs against LightGBM native outputs
to verify export accuracy.
"""

import numpy as np
import logging
import onnxruntime as ort
from typing import List
import yaml


logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/system_config.yaml") -> dict:
    """Load system configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_onnx(
    models_lgb: List,
    onnx_path: str,
    X_test: np.ndarray,
    Y_test_dict: dict,
    config: dict,
    max_deviation: float = 0.005
) -> bool:
    """Validate ONNX models against LightGBM outputs.

    Args:
        models_lgb: List of trained LightGBM regressors
        onnx_path: Path to ONNX model(s)
        X_test: Test feature matrix
        Y_test_dict: Dict of {target_name: Y_test_array}
        config: System configuration
        max_deviation: Maximum allowed relative deviation (0.5% default)

    Returns:
        True if validation passes, False otherwise
    """
    logger.info("=" * 80)
    logger.info("VALIDATING ONNX MODELS")
    logger.info("=" * 80)

    all_passed = True

    for i, (model_lgb, target_name) in enumerate(zip(models_lgb, config['model']['targets'])):
        target_path = onnx_path.replace('.onnx', f'_{target_name}.onnx')

        logger.info(f"\nValidating: {target_name}")
        logger.info(f"  Loading ONNX from: {target_path}")

        try:
            # Load and run ONNX model
            session = ort.InferenceSession(target_path, providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            onnx_predictions = session.run([output_name], {input_name: X_test.astype(np.float32)})[0]
            lgb_predictions = model_lgb.predict(X_test)

            # Flatten if needed
            if len(onnx_predictions.shape) > 1:
                onnx_predictions = onnx_predictions.flatten()

            # Compute metrics
            mae = np.mean(np.abs(onnx_predictions - lgb_predictions))
            max_abs_error = np.max(np.abs(onnx_predictions - lgb_predictions))
            relative_error = mae / (np.mean(np.abs(lgb_predictions)) + 1e-9)

            logger.info(f"  MAE: {mae:.6f}")
            logger.info(f"  Max error: {max_abs_error:.6f}")
            logger.info(f"  Relative error: {relative_error:.4f}")

            if relative_error > max_deviation:
                logger.warning(f"  FAILED: Relative error {relative_error:.4f} > threshold {max_deviation:.4f}")
                all_passed = False
            else:
                logger.info(f"  PASSED")

        except Exception as e:
            logger.error(f"  ERROR loading/running ONNX: {e}")
            all_passed = False

    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("VALIDATION PASSED")
    else:
        logger.warning("VALIDATION FAILED")
    logger.info("=" * 80)

    return all_passed


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from model.train import train_pipeline
    from model.export import export_to_onnx

    config = load_config()

    # Train models
    models, feature_names = train_pipeline(config)

    # Export to ONNX
    output_path = config['model']['onnx_path']
    export_to_onnx(models, feature_names, output_path, config)

    logger.info("Validation would require test set - skipping in standalone execution")
