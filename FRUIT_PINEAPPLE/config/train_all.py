#!/usr/bin/env python3
"""
Execute notebooks directly to train models
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\Asus\Desktop\FARMBOT AI\FRC\FRUIT_PINEAPPLE")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "config"))

# Import paths at module level
from config.paths import (
    DETECTION_DATA, DETECTION_IMAGES_TRAIN, DETECTION_IMAGES_VAL,
    DETECTION_IMAGES_TEST, DETECTION_MODEL, CLASSIFICATION_DATA,
    CLASSIFICATION_TRAIN, CLASSIFICATION_TEST, CLASSIFICATION_MODEL,
    WEIGHT_DATA_CSV, WEIGHT_MODEL, MODELS_DIR
)

def run_detection_notebook():
    """Run detection training"""
    print("\n" + "=" * 70)
    print("TRAINING: PINEAPPLE DETECTION (YOLOv8)")
    print("=" * 70)

    try:
        from ultralytics import YOLO
        import yaml
        import time
        import shutil

        print("\n[1/8] Setting up data configuration...")
        data_config = {
            'path': str(DETECTION_DATA / 'detection'),
            'train': str(DETECTION_IMAGES_TRAIN),
            'val': str(DETECTION_IMAGES_VAL),
            'test': str(DETECTION_IMAGES_TEST) if DETECTION_IMAGES_TEST.exists() else str(DETECTION_IMAGES_VAL),
            'nc': 1,
            'names': ['pineapple']
        }

        yaml_path = PROJECT_ROOT / 'config' / 'detection_data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f)
        print(f"[OK] Data config saved: {yaml_path}")

        print("\n[2/8] Loading YOLOv8n (Nano) model...")
        model = YOLO('yolov8n.pt')
        print(f"[OK] Model loaded")

        print("\n[3/8] Starting training (100 epochs)...")
        print("      This may take 2-4 hours on GPU, ~12 hours on CPU")
        results = model.train(
            data=str(yaml_path),
            epochs=100,
            imgsz=416,
            batch=8,  # Reduced batch size for CPU
            lr0=0.001,
            weight_decay=0.0005,
            optimizer='Adam',
            patience=20,
            save=True,
            device='cpu'  # Use CPU
        )
        print(f"[OK] Training completed")

        # Find and copy best model
        print("\n[4/8] Saving trained model...")
        import shutil
        best_models = list(PROJECT_ROOT.glob("runs/detect/train*/weights/best.pt"))
        if best_models:
            best_model = best_models[-1]
            shutil.copy(str(best_model), str(DETECTION_MODEL))
            size_mb = DETECTION_MODEL.stat().st_size / (1024 * 1024)
            print(f"[OK] Model saved: {DETECTION_MODEL.name} ({size_mb:.2f} MB)")
        else:
            print("[ERROR] Could not find trained model")
            return False

        print("\n[5/8] Running validation...")
        model = YOLO(str(DETECTION_MODEL))
        metrics = model.val(data=str(yaml_path), batch=16)
        print(f"[OK] Validation completed")
        print(f"      mAP50: {metrics.box.map50:.4f}")
        print(f"      mAP: {metrics.box.map:.4f}")

        print("\n[6/8] Benchmarking inference speed...")
        test_images = list(DETECTION_IMAGES_VAL.glob('*.jpg'))[:20]
        start = time.time()
        for img in test_images:
            model.predict(source=str(img), conf=0.25, verbose=False)
        avg_time = (time.time() - start) / len(test_images)
        print(f"[OK] Inference speed: {avg_time*1000:.2f}ms/image ({1/avg_time:.2f} fps)")

        print("\n[7/8] Exporting quantized models...")
        model.export(format='torchscript', half=True, device='cpu')
        model.export(format='onnx', opset=13, device='cpu')
        print(f"[OK] FP16 and ONNX exports completed")

        print("\n[8/8] Detection training complete!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Detection training failed:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_classification_notebook():
    """Run classification training"""
    print("\n" + "=" * 70)
    print("TRAINING: RIPENESS CLASSIFICATION (YOLOv8)")
    print("=" * 70)

    try:
        from ultralytics import YOLO
        import yaml
        import time
        import shutil

        print("\n[1/7] Setting up classification data...")
        yaml_path = PROJECT_ROOT / 'config' / 'classification_data.yaml'

        # YOLOv8 classification expects directory structure:
        # data/train/class1/, data/train/class2/
        # data/val/class1/, data/val/class2/
        data_path = CLASSIFICATION_DATA / 'molvumClassification'

        print(f"[OK] Classification data path: {data_path}")

        print("\n[2/7] Loading YOLOv8n-cls model...")
        model = YOLO('yolov8n-cls.pt')
        print(f"[OK] Model loaded")

        print("\n[3/7] Starting classification training (100 epochs)...")
        print("      This may take 1-2 hours on GPU, 6-8 hours on CPU")

        results = model.train(
            data=str(data_path),
            epochs=100,
            imgsz=224,
            batch=16,
            lr0=0.001,
            weight_decay=0.0005,
            patience=20,
            save=True,
            device='cpu'
        )
        print(f"[OK] Training completed")

        print("\n[4/7] Saving trained model...")
        best_models = list(PROJECT_ROOT.glob("runs/classify/train*/weights/best.pt"))
        if best_models:
            best_model = best_models[-1]
            shutil.copy(str(best_model), str(CLASSIFICATION_MODEL))
            size_mb = CLASSIFICATION_MODEL.stat().st_size / (1024 * 1024)
            print(f"[OK] Model saved: {CLASSIFICATION_MODEL.name} ({size_mb:.2f} MB)")
        else:
            print("[ERROR] Could not find trained model")
            return False

        print("\n[5/7] Running validation...")
        model = YOLO(str(CLASSIFICATION_MODEL))
        metrics = model.val()
        print(f"[OK] Validation completed")

        print("\n[6/7] Benchmarking inference speed...")
        test_images = list(CLASSIFICATION_TEST.glob('**/*.jpg'))[:20]
        if test_images:
            start = time.time()
            for img in test_images:
                model.predict(source=str(img), verbose=False)
            avg_time = (time.time() - start) / len(test_images)
            print(f"[OK] Inference speed: {avg_time*1000:.2f}ms/image ({1/avg_time:.2f} fps)")

        print("\n[7/7] Exporting quantized models...")
        model.export(format='torchscript', half=True, device='cpu')
        model.export(format='onnx', opset=13, device='cpu')
        print(f"[OK] Models exported for Jetson")

        print("\n[DONE] Classification training complete!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Classification training failed:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_weight_prediction_notebook():
    """Run weight prediction training"""
    print("\n" + "=" * 70)
    print("TRAINING: WEIGHT PREDICTION MODELS")
    print("=" * 70)

    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        import joblib

        print("\n[1/6] Loading weight prediction data...")
        df = pd.read_csv(str(WEIGHT_DATA_CSV))
        print(f"[OK] Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        print(f"     Columns: {', '.join(df.columns.tolist())}")

        print("\n[2/6] Preparing data...")
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"[OK] Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

        print("\n[3/6] Training ML models...")

        results = {}

        # Linear Regression
        print("     - Linear Regression")
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        y_pred = lr.predict(X_test_scaled)
        results['Linear Regression'] = {
            'model': lr,
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        # Polynomial Regression
        print("     - Polynomial Regression")
        poly_features = PolynomialFeatures(degree=2)
        X_train_poly = poly_features.fit_transform(X_train_scaled)
        X_test_poly = poly_features.transform(X_test_scaled)
        poly = LinearRegression()
        poly.fit(X_train_poly, y_train)
        y_pred = poly.predict(X_test_poly)
        results['Polynomial Regression'] = {
            'model': poly,
            'poly_features': poly_features,
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        # SVR
        print("     - Support Vector Regressor")
        svr = SVR(kernel='rbf', C=100, epsilon=0.1)
        svr.fit(X_train_scaled, y_train)
        y_pred = svr.predict(X_test_scaled)
        results['SVR'] = {
            'model': svr,
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        # Decision Tree
        print("     - Decision Tree Regressor")
        dt = DecisionTreeRegressor(max_depth=10, random_state=42)
        dt.fit(X_train_scaled, y_train)
        y_pred = dt.predict(X_test_scaled)
        results['Decision Tree'] = {
            'model': dt,
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        # KNN
        print("     - K-Nearest Neighbors")
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        results['KNN'] = {
            'model': knn,
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        print(f"[OK] All models trained")

        print("\n[4/6] Results summary:")
        for name, metrics in results.items():
            print(f"     {name:25s} - R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}")

        print("\n[5/6] Saving best model...")
        best_name = max(results, key=lambda x: results[x]['r2'])
        best_data = results[best_name]

        joblib.dump(best_data['model'], str(WEIGHT_MODEL))
        joblib.dump(scaler, str(MODELS_DIR / 'weight_scaler.pkl'))

        if 'poly_features' in best_data:
            joblib.dump(best_data['poly_features'], str(MODELS_DIR / 'weight_poly_features.pkl'))

        print(f"[OK] Best model saved: {best_name}")
        print(f"     - {WEIGHT_MODEL.name}")
        print(f"     - weight_scaler.pkl")

        print("\n[6/6] Weight prediction training complete!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Weight prediction training failed:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all training notebooks"""
    print("\n" * 2)
    print("=" * 70)
    print("FRUIT PINEAPPLE TRAINING PIPELINE")
    print("Detection - Classification - Weight Prediction")
    print("=" * 70)

    results = {
        "Detection": False,
        "Classification": False,
        "Weight Prediction": False
    }

    try:
        results["Detection"] = run_detection_notebook()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Detection training cancelled")

    try:
        results["Classification"] = run_classification_notebook()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Classification training cancelled")

    try:
        results["Weight Prediction"] = run_weight_prediction_notebook()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Weight prediction training cancelled")

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    for task, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"{status} {task:30s}")

    if all(results.values()):
        print("\n" + "=" * 70)
        print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
        print("=" * 70)
        print("\nModels saved to:")
        for f in MODELS_DIR.glob('*'):
            if f.is_file():
                size = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name:40s} ({size:6.2f} MB)")
        print("\nNext steps:")
        print("  1. Deploy models to Jetson")
        print("  2. Run inference on edge device")
        print("  3. Monitor real-time performance")
        return 0
    else:
        print("\nSome trainings failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
