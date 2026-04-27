"""
ONNX Runtime inference wrapper for pineapple control predictions.

Loads ONNX models and provides simple predict() interface returning ActionVector.
"""

import logging
import os
from typing import List
import numpy as np
import onnxruntime as ort

from data.schema import FeatureVector, ActionVector


logger = logging.getLogger(__name__)


class Predictor:
    """ONNX Runtime inference engine for control actions."""

    def __init__(self, model_dir: str = "model", target_names: List[str] = None):
        """Initialize predictor by loading ONNX models.

        Args:
            model_dir: Directory containing ONNX model files
            target_names: List of target names matching model file names
        """
        if target_names is None:
            target_names = ['delta_N', 'delta_P', 'delta_K', 'irrigation_ml', 'pH_adj']

        self.model_dir = model_dir
        self.target_names = target_names
        self.sessions = {}

        logger.info(f"Loading ONNX models from: {model_dir}")

        for target in target_names:
            model_path = os.path.join(model_dir, f"pineapple_model_{target}.onnx")

            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_path}")
                continue

            try:
                session = ort.InferenceSession(
                    model_path,
                    providers=['CPUExecutionProvider']
                )
                self.sessions[target] = session
                logger.info(f"  Loaded: {target}")
            except Exception as e:
                logger.error(f"  Failed to load {target}: {e}")

        logger.info(f"Loaded {len(self.sessions)}/{len(target_names)} models")

    def predict(self, feature_vector: FeatureVector) -> ActionVector:
        """Run inference on feature vector.

        Args:
            feature_vector: FeatureVector with all computed features

        Returns:
            ActionVector with control actions
        """
        if not self.sessions:
            raise RuntimeError("No models loaded")

        # Convert to numpy array (shape: 1 x n_features)
        X = feature_vector.to_array()

        predictions = {}

        for target, session in self.sessions.items():
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            try:
                output = session.run([output_name], {input_name: X})[0]
                # Handle different output shapes
                if isinstance(output, np.ndarray):
                    if output.ndim > 1:
                        output = output.flatten()
                    predictions[target] = float(output[0])
                else:
                    predictions[target] = float(output)

            except Exception as e:
                logger.error(f"Inference failed for {target}: {e}")
                predictions[target] = 0.0

        # Return ActionVector in consistent order
        return ActionVector(
            delta_N=predictions.get('delta_N', 0.0),
            delta_P=predictions.get('delta_P', 0.0),
            delta_K=predictions.get('delta_K', 0.0),
            irrigation_ml=predictions.get('irrigation_ml', 0.0),
            pH_adj=predictions.get('pH_adj', 0.0)
        )

    def warm_up(self, n_iterations: int = 10) -> None:
        """Run dummy predictions to JIT-load the sessions.

        Args:
            n_iterations: Number of dummy predictions to run
        """
        logger.info(f"Warming up inference engines ({n_iterations} iterations)...")

        dummy_fv = FeatureVector(
            N=125, P=45, K=175, EC=900, pH=5.5, moisture=65, temp=27,
            growth_stage=0, hour_of_day=12,
            delta_N=0, delta_P=0, delta_K=0, delta_moisture=0,
            rolling_avg_EC=900, rolling_avg_N=125,
            EC_per_moisture=13.8, pH_error=0.05, N_K_ratio=0.71,
            moisture_x_EC=58500, deviation_score=0.1
        )

        for i in range(n_iterations):
            try:
                self.predict(dummy_fv)
            except Exception as e:
                logger.warning(f"Warm-up iteration {i} failed: {e}")

        logger.info("Warm-up complete")
