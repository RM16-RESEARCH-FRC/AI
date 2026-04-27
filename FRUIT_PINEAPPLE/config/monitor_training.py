#!/usr/bin/env python3
"""
Monitor training progress and verify model outputs
"""
import json
from pathlib import Path
import time
import subprocess

PROJECT_ROOT = Path(r"c:\Users\Asus\Desktop\FARMBOT AI\FRC\FRUIT_PINEAPPLE")
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

def check_models():
    """Check if models have been trained and saved"""
    print("\n" + "="*60)
    print("MODEL STATUS CHECK")
    print("="*60)

    models_to_check = {
        "Detection Model": MODELS_DIR / "detection_model.pt",
        "Classification Model": MODELS_DIR / "classification_model.pt",
        "Weight Prediction Model": MODELS_DIR / "weight_prediction_model.pkl",
        "Weight Scaler": MODELS_DIR / "weight_scaler.pkl",
    }

    for name, path in models_to_check.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"[OK] {name}: {path.name} ({size_mb:.2f} MB)")
        else:
            print(f"[PENDING] {name}: Not yet created")

    return all(path.exists() for path in models_to_check.values())

def check_training_logs():
    """Check if training runs directory exists"""
    runs_dir = PROJECT_ROOT / "runs"
    if runs_dir.exists():
        print("\n" + "="*60)
        print("TRAINING RUNS")
        print("="*60)

        for task_dir in sorted(runs_dir.glob("*/*")):
            if task_dir.is_dir():
                print(f"  - {task_dir.relative_to(runs_dir)}")
                # Check for results
                if (task_dir / "results.csv").exists():
                    print(f"    [OK] Training results available")
    else:
        print("\n[INFO] Training hasn't started yet (runs/ directory not created)")

def check_data():
    """Verify datasets are in place"""
    print("\n" + "="*60)
    print("DATA VERIFICATION")
    print("="*60)

    datasets = {
        "Detection": DATA_DIR / "detection" / "detection",
        "Classification": DATA_DIR / "classification" / "molvumClassification",
        "Weight Data": DATA_DIR / "weight prediction data.csv"
    }

    for name, path in datasets.items():
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"[OK] {name}: {size_mb:.2f} MB")
            else:
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
                print(f"[OK] {name}: {size:.2f} MB total")
        else:
            print(f"[ERROR] {name}: Not found at {path}")

def summarize_status():
    """Print summary of training status"""
    print("\n" + "="*60)
    print("TRAINING PIPELINE STATUS")
    print("="*60)

    all_done = check_models()
    check_training_logs()
    check_data()

    print("\n" + "="*60)
    if all_done:
        print("STATUS: ALL MODELS TRAINED AND SAVED!")
        print("\nNext steps:")
        print("1. Review metrics in notebooks/outputs/")
        print("2. Export models for Jetson deployment")
        print("3. Deploy to Jetson hardware")
    else:
        print("STATUS: TRAINING IN PROGRESS")
        print("\nTo check progress again, run:")
        print("  python config/monitor_training.py")
    print("="*60 + "\n")

    return all_done

if __name__ == "__main__":
    summarize_status()
