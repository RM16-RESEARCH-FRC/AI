#!/usr/bin/env python3
"""Quick status check for training progress"""
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\Asus\Desktop\FARMBOT AI\FRC\FRUIT_PINEAPPLE")
MODELS_DIR = PROJECT_ROOT / "models"

print("=" * 60)
print("TRAINING STATUS")
print("=" * 60)

models = {
    "Detection": MODELS_DIR / "detection_model.pt",
    "Classification": MODELS_DIR / "classification_model.pt",
    "Weight Prediction": MODELS_DIR / "weight_prediction_model.pkl"
}

for name, path in models.items():
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"[DONE] {name:30s} {size_mb:8.2f} MB")
    else:
        print(f"[...] {name:30s} (training...)")

runs = list(PROJECT_ROOT.glob("runs/*/train*"))
if runs:
    latest = max(runs, key=lambda p: p.stat().st_mtime)
    print(f"\nLatest training: {latest.relative_to(PROJECT_ROOT)}")

    results = list(latest.glob("**/*.csv"))
    if results:
        print(f"Results file found: {results[0].name}")

print("=" * 60)
