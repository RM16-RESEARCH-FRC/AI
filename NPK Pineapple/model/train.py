"""
LightGBM model training pipeline for pineapple fertigation predictions.

Trains separate multi-output regression models for nutrient and water control.
"""

import numpy as np
import logging
from typing import List, Tuple
import yaml
import os
import csv
import json
from datetime import datetime

from lightgbm import LGBMRegressor
from data.simulator import SoilSimulator
from features.engineer import FeatureEngineer
from data.schema import SensorReading, ActionVector, FeatureVector


logger = logging.getLogger(__name__)


def _save_training_results(
    metrics: List[dict],
    sample_predictions: List[dict],
    feature_importance: List[dict],
    config: dict,
    train_size: int,
    test_size: int,
    n_features: int,
) -> None:
    """Persist model evaluation outputs for inspection after training."""
    os.makedirs("model", exist_ok=True)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "total_rows": train_size + test_size,
            "train_rows": train_size,
            "test_rows": test_size,
            "split": "time_ordered_80_20",
            "features": n_features,
            "targets": config["model"]["targets"],
        },
        "interpretation": {
            "mae": "Mean absolute prediction error in the same unit as the target. Lower is better.",
            "rmse": "Root mean squared error, which penalizes larger mistakes. Lower is better.",
            "r2": "Explained variance score. 1.0 is perfect; values near 0 mean weak predictive value.",
        },
        "metrics": metrics,
    }

    with open("model/model_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    metric_fields = [
        "target", "unit", "train_mae", "test_mae", "test_rmse",
        "train_r2", "test_r2", "test_mean", "test_min", "test_max",
        "model_path",
    ]
    with open("model/model_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metrics)

    if sample_predictions:
        with open("model/sample_predictions.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sample_predictions[0].keys()))
            writer.writeheader()
            writer.writerows(sample_predictions)

    if feature_importance:
        with open("model/feature_importance.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["target", "feature", "importance"])
            writer.writeheader()
            writer.writerows(feature_importance)

    logger.info("Saved model evaluation results:")
    logger.info("  model/model_report.json")
    logger.info("  model/model_metrics.csv")
    logger.info("  model/sample_predictions.csv")
    logger.info("  model/feature_importance.csv")


def _target_unit(target: str) -> str:
    """Return display unit for each model target."""
    return {
        "delta_N": "mg/kg",
        "delta_P": "mg/kg",
        "delta_K": "mg/kg",
        "irrigation_ml": "mL",
        "pH_adj": "pH adjustment",
    }.get(target, "")


def load_config(config_path: str = "config/system_config.yaml") -> dict:
    """Load system configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_pipeline(config: dict) -> Tuple[List[LGBMRegressor], List[str]]:
    """Complete training pipeline from sim data to trained models.

    Args:
        config: System configuration dict

    Returns:
        Tuple of (list of trained models, feature names)
    """
    logger.info("=" * 80)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 80)

    # 1. Generate simulation dataset
    logger.info(f"Generating synthetic dataset with {config['system']['sim_num_rows']} samples...")
    simulator = SoilSimulator(seed=42)
    dataset = simulator.generate_dataset(
        n_steps=config['system']['sim_num_rows'],
        policy='random'  # Mix optimal and random for diversity
    )

    # 2. Extract features and targets
    logger.info("Computing features for all samples...")
    feature_engineer = FeatureEngineer(buffer_size=6)

    X_list = []
    Y_dict = {target: [] for target in config['model']['targets']}

    for reading, action in dataset:
        features = feature_engineer.compute([reading])
        X_list.append(features.to_array()[0])

        for target in config['model']['targets']:
            Y_dict[target].append(getattr(action, target))

    X = np.array(X_list, dtype=np.float32)
    logger.info(f"Feature matrix shape: {X.shape}")

    # 3. Split dataset (80/20 time-ordered split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]

    logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # 4. Train LightGBM models (one per target)
    models = []
    feature_names = FeatureVector.feature_names()
    metrics = []
    feature_importance = []
    sample_predictions = [{"test_row": i} for i in range(min(50, X_test.shape[0]))]

    for target in config['model']['targets']:
        Y = np.array(Y_dict[target], dtype=np.float32)
        Y_train, Y_test = Y[:split_idx], Y[split_idx:]

        logger.info(f"\nTraining model for target: {target}")
        logger.info(f"  Y range (train): [{Y_train.min():.2f}, {Y_train.max():.2f}]")
        logger.info(f"  Y range (test): [{Y_test.min():.2f}, {Y_test.max():.2f}]")

        model = LGBMRegressor(
            **config['model']['lgb_params'],
            verbose=-1,
            random_state=42,
            objective='regression'
        )

        model.fit(
            X_train, Y_train,
            eval_set=[(X_test, Y_test)],
            callbacks=[],
            eval_metric='mae'
        )

        # Evaluate on test set
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_mae = np.mean(np.abs(train_pred - Y_train))
        test_mae = np.mean(np.abs(test_pred - Y_test))
        test_rmse = np.sqrt(np.mean((test_pred - Y_test) ** 2))
        train_r2 = model.score(X_train, Y_train)
        test_r2 = model.score(X_test, Y_test)

        logger.info(f"  Train MAE: {train_mae:.4f}, R2: {train_r2:.4f}")
        logger.info(f"  Test MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

        # Save model to text format for inspection
        model_txt_path = f"model/{target}_model.txt"
        model.booster_.save_model(model_txt_path)
        logger.info(f"  Saved: {model_txt_path}")

        metrics.append({
            "target": target,
            "unit": _target_unit(target),
            "train_mae": round(float(train_mae), 6),
            "test_mae": round(float(test_mae), 6),
            "test_rmse": round(float(test_rmse), 6),
            "train_r2": round(float(train_r2), 6),
            "test_r2": round(float(test_r2), 6),
            "test_mean": round(float(np.mean(Y_test)), 6),
            "test_min": round(float(np.min(Y_test)), 6),
            "test_max": round(float(np.max(Y_test)), 6),
            "model_path": model_txt_path,
        })

        for feature, importance in zip(feature_names, model.feature_importances_):
            feature_importance.append({
                "target": target,
                "feature": feature,
                "importance": int(importance),
            })

        for i, row in enumerate(sample_predictions):
            row[f"{target}_actual"] = round(float(Y_test[i]), 6)
            row[f"{target}_predicted"] = round(float(test_pred[i]), 6)
            row[f"{target}_error"] = round(float(test_pred[i] - Y_test[i]), 6)

        models.append(model)

    _save_training_results(
        metrics=metrics,
        sample_predictions=sample_predictions,
        feature_importance=feature_importance,
        config=config,
        train_size=X_train.shape[0],
        test_size=X_test.shape[0],
        n_features=X.shape[1],
    )

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)

    return models, feature_names


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    config = load_config()
    models, feature_names = train_pipeline(config)
    logger.info(f"Trained {len(models)} models with {len(feature_names)} features")
