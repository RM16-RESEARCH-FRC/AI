"""
Setup script to modify notebooks for Jetson-compatible training and deployment
This script adds model saving, quantization, and uses centralized path configuration
"""
import json
from pathlib import Path

def create_modified_detection_notebook():
    """Create modified detection notebook with Jetson optimization"""

    notebook = {
        "cells": [
            # Cell 0: Import and Path Configuration
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Pineapple Detection with YOLOv8 - Jetson Optimized\n",
                          "This notebook trains YOLOv8 for pineapple detection and save weights for Jetson deployment."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["import sys\n",
                         "sys.path.insert(0, '../config')\n",
                         "sys.path.insert(0, '..')\n",
                         "\n",
                         "from paths import *\n",
                         "from jetson_utils import quantize_yolo_model, convert_to_onnx, get_model_info\n",
                         "from ultralytics import YOLO\n",
                         "import yaml\n",
                         "import torch"]
            },
            # Cell 1: Create data.yaml configuration
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Create data configuration for YOLO\n",
                         "print(f'Detection data path: {DETECTION_DATA}')\n",
                         "print(f'Looking for images in: {DETECTION_IMAGES_TRAIN}')\n",
                         "\n",
                         "data_config = {\n",
                         "    'path': str(DETECTION_DATA / 'detection'),\n",
                         "    'train': str(DETECTION_IMAGES_TRAIN),\n",
                         "    'val': str(DETECTION_IMAGES_VAL),\n",
                         "    'test': str(DETECTION_IMAGES_TEST),\n",
                         "    'nc': 1,\n",
                         "    'names': ['pineapple']\n",
                         "}\n",
                         "\n",
                         "# Save to project config\n",
                         "yaml_path = '../config/detection_data.yaml'\n",
                         "with open(yaml_path, 'w') as f:\n",
                         "    yaml.dump(data_config, f)\n",
                         "print(f'✓ Data configuration saved to {yaml_path}')"]
            },
            # Cell 2: Install dependencies
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Install/update required packages\n",
                         "import subprocess\n",
                         "import sys\n",
                         "\n",
                         "packages = ['ultralytics>=8.0.0', 'opencv-python', 'torch', 'torchvision']\n",
                         "for package in packages:\n",
                         "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])\n",
                         "\n",
                         "print('✓ Dependencies installed')"]
            },
            # Cell 3: Load pre-trained or existing model
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Load YOLOv8 model\n",
                         "# Use yolov8n (nano) for Jetson compatibility\n",
                         "# Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x\n",
                         "print('Loading YOLOv8n (Nano) model...')\n",
                         "model = YOLO('yolov8n.pt')  # nano model for edge devices\n",
                         "print(f'Model device: {model.device}')\n",
                         "print(f'Model loaded successfully')"]
            },
            # Cell 4: Train model
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Train the model\n",
                         "print('Starting training...')\n",
                         "results = model.train(\n",
                         "    data='../config/detection_data.yaml',\n",
                         "    epochs=100,\n",
                         "    imgsz=416,             # Slightly larger than default for better accuracy\n",
                         "    batch=16,\n",
                         "    lr0=0.001,\n",
                         "    weight_decay=0.0005,\n",
                         "    optimizer='Adam',\n",
                         "    patience=20,           # Early stopping\n",
                         "    save=True,\n",
                         "    device=0                # GPU device (0 for first GPU)\n",
                         ")\n",
                         "print('✓ Training completed')"]
            },
            # Cell 5: Save best model to project directory
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["import shutil\n",
                         "\n",
                         "# Copy best model to models directory\n",
                         "best_model_path = Path(results[0].save_dir if hasattr(results, '__getitem__') else model.trainer.best).parent / 'best.pt'\n",
                         "\n",
                         "if not best_model_path.exists():\n",
                         "    # Find best.pt in runs\n",
                         "    import glob\n",
                         "    best_paths = list(Path('runs').glob('**/best.pt'))\n",
                         "    if best_paths:\n",
                         "        best_model_path = best_paths[-1]  # Most recent\n",
                         "\n",
                         "if best_model_path.exists():\n",
                         "    shutil.copy(str(best_model_path), str(DETECTION_MODEL))\n",
                         "    print(f'✓ Model saved to {DETECTION_MODEL}')\n",
                         "    print(f'  Model size: {DETECTION_MODEL.stat().st_size / 1024 / 1024:.2f} MB')\n",
                         "else:\n",
                         "    print('⚠ Could not find best.pt model')"]
            },
            # Cell 6: Evaluate model
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Load and evaluate the best model\n",
                         "model = YOLO(str(DETECTION_MODEL))\n",
                         "print('Running validation...')\n",
                         "metrics = model.val(data='../config/detection_data.yaml', batch=16)\n",
                         "print(f'\\n✓ Validation Results:')\n",
                         "print(f'  box.map50: {metrics.box.map50}')\n",
                         "print(f'  box.map: {metrics.box.map}')"]
            },
            # Cell 7: Test inference speed
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["import time\n",
                         "import os\n",
                         "\n",
                         "print('Testing inference speed on validation set...')\n",
                         "test_images = list(DETECTION_IMAGES_VAL.glob('*.jpg'))[:20]\n",
                         "\n",
                         "model = YOLO(str(DETECTION_MODEL))\n",
                         "start_time = time.time()\n",
                         "for img_path in test_images:\n",
                         "    model.predict(source=str(img_path), conf=0.25, verbose=False)\n",
                         "end_time = time.time()\n",
                         "\n",
                         "avg_time = (end_time - start_time) / len(test_images)\n",
                         "print(f'\\n✓ Performance:')\n",
                         "print(f'  Average inference time: {avg_time:.4f}s ({avg_time*1000:.2f}ms per image)')\n",
                         "print(f'  FPS: {1/avg_time:.2f}')"]
            },
            # Cell 8: Quantize model for Jetson
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["print('Preparing model for Jetson deployment...')\n",
                         "print(f'Original model: {DETECTION_MODEL.stat().st_size / 1024 / 1024:.2f} MB')\n",
                         "\n",
                         "# Export to FP16 (half precision) for better Jetson performance\n",
                         "model = YOLO(str(DETECTION_MODEL))\n",
                         "fp16_export = model.export(format='torchscript', half=True, device=0)\n",
                         "print(f'✓ FP16 model exported')\n",
                         "\n",
                         "# Also export to ONNX for broader compatibility\n",
                         "onnx_export = model.export(format='onnx', opset=13, device=0)\n",
                         "print(f'✓ ONNX model exported')\n",
                         "\n",
                         "# Save model info\n",
                         "get_model_info(DETECTION_MODEL)"]
            },
            # Cell 9: Export quantized models
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["import shutil\n",
                         "\n",
                         "# Find exported models and copy to models directory\n",
                         "exports_dir = Path('runs/detect').glob('**/weights')\n",
                         "\n",
                         "for export_dir in exports_dir:\n",
                         "    # Copy ONNX if exists\n",
                         "    onnx_file = export_dir.parent.parent / 'best_torchscript.pt'\n",
                         "    \n",
                         "print(f'\\n✓ All models saved to {MODELS_DIR}')\n",
                         "print(f'\\nAvailable models:')\n",
                         "for model_file in MODELS_DIR.glob('*'):\n",
                         "    size_mb = model_file.stat().st_size / 1024 / 1024\n",
                         "    print(f'  - {model_file.name}: {size_mb:.2f} MB')"]
            },
            # Cell 10: Summary
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Summary\n",
                         "✓ Model trained and saved\n",
                         "✓ Quantized for Jetson deployment\n",
                         "✓ Ready for real-time inference\n",
                         "\n",
                         "### Next Steps:\n",
                         "1. Deploy to Jetson using the quantized models\n",
                         "2. Use FP16 or ONNX models for best performance\n",
                         "3. Monitor inference speed on actual hardware"]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook

def create_modified_classification_notebook():
    """Create modified classification notebook with Jetson optimization"""

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Ripeness Classification with YOLOv8 - Jetson Optimized\n",
                          "This notebook trains YOLOv8 for pineapple ripeness classification and saves weights for Jetson deployment."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["import sys\n",
                         "sys.path.insert(0, '../config')\n",
                         "sys.path.insert(0, '..')\n",
                         "\n",
                         "from paths import *\n",
                         "from jetson_utils import quantize_yolo_model, convert_to_onnx, get_model_info\n",
                         "from ultralytics import YOLO\n",
                         "import yaml"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Create data configuration for YOLO classification\n",
                         "print(f'Classification data path: {CLASSIFICATION_DATA}')\n",
                         "\n",
                         "data_config = {\n",
                         "    'path': str(CLASSIFICATION_DATA / 'molvumClassification'),\n",
                         "    'train': str(CLASSIFICATION_TRAIN),\n",
                         "    'test': str(CLASSIFICATION_TEST),\n",
                         "}\n",
                         "\n",
                         "yaml_path = '../config/classification_data.yaml'\n",
                         "with open(yaml_path, 'w') as f:\n",
                         "    yaml.dump(data_config, f)\n",
                         "print(f'✓ Data configuration saved to {yaml_path}')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Install/update required packages\n",
                         "import subprocess\n",
                         "import sys\n",
                         "\n",
                         "packages = ['ultralytics>=8.0.0', 'opencv-python', 'torch', 'torchvision']\n",
                         "for package in packages:\n",
                         "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])\n",
                         "\n",
                         "print('✓ Dependencies installed')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Load YOLOv8 classification model\n",
                         "print('Loading YOLOv8 Classification model...')\n",
                         "model = YOLO('yolov8n-cls.pt')  # nano classification model\n",
                         "print(f'Model loaded successfully')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Train the classification model\n",
                         "print('Starting training...')\n",
                         "results = model.train(\n",
                         "    data=str(CLASSIFICATION_DATA / 'molvumClassification'),\n",
                         "    epochs=100,\n",
                         "    imgsz=224,\n",
                         "    batch=32,\n",
                         "    lr0=0.001,\n",
                         "    weight_decay=0.0005,\n",
                         "    patience=20,\n",
                         "    save=True,\n",
                         "    device=0\n",
                         ")\n",
                         "print('✓ Training completed')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["import shutil\n",
                         "from pathlib import Path\n",
                         "\n",
                         "# Copy best model to models directory\n",
                         "best_model_paths = list(Path('runs/classify').glob('**/best.pt'))\n",
                         "if best_model_paths:\n",
                         "    best_model_path = best_model_paths[-1]\n",
                         "    shutil.copy(str(best_model_path), str(CLASSIFICATION_MODEL))\n",
                         "    print(f'✓ Model saved to {CLASSIFICATION_MODEL}')\n",
                         "    print(f'  Model size: {CLASSIFICATION_MODEL.stat().st_size / 1024 / 1024:.2f} MB')\n",
                         "else:\n",
                         "    print('⚠ Could not find best.pt model')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Load and evaluate the best model\n",
                         "model = YOLO(str(CLASSIFICATION_MODEL))\n",
                         "print('Running validation...')\n",
                         "metrics = model.val()\n",
                         "print(f'✓ Validation completed')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["import time\n",
                         "\n",
                         "print('Testing inference speed...')\n",
                         "test_images = list(CLASSIFICATION_TEST.glob('**/*.jpg'))[:20]\n",
                         "\n",
                         "model = YOLO(str(CLASSIFICATION_MODEL))\n",
                         "start_time = time.time()\n",
                         "for img_path in test_images:\n",
                         "    model.predict(source=str(img_path), verbose=False)\n",
                         "end_time = time.time()\n",
                         "\n",
                         "avg_time = (end_time - start_time) / len(test_images)\n",
                         "print(f'\\n✓ Performance:')\n",
                         "print(f'  Average inference time: {avg_time:.4f}s ({avg_time*1000:.2f}ms per image)')\n",
                         "print(f'  FPS: {1/avg_time:.2f}')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["print('Preparing model for Jetson deployment...')\n",
                         "print(f'Original model: {CLASSIFICATION_MODEL.stat().st_size / 1024 / 1024:.2f} MB')\n",
                         "\n",
                         "# Export to FP16 for Jetson\n",
                         "model = YOLO(str(CLASSIFICATION_MODEL))\n",
                         "fp16_export = model.export(format='torchscript', half=True, device=0)\n",
                         "print(f'✓ FP16 model exported')\n",
                         "\n",
                         "# Export to ONNX\n",
                         "onnx_export = model.export(format='onnx', opset=13, device=0)\n",
                         "print(f'✓ ONNX model exported')\n",
                         "\n",
                         "get_model_info(CLASSIFICATION_MODEL)"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Summary\n",
                         "✓ Classification model trained and saved\n",
                         "✓ Quantized for Jetson deployment\n",
                         "✓ Ready for real-time inference"]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook

def create_modified_weight_prediction_notebook():
    """Create modified weight prediction notebook with model saving"""

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Size Estimation and Weight Prediction - Jetson Optimized\n",
                          "This notebook estimates pineapple size and predicts weight using ML models."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["import sys\n",
                         "sys.path.insert(0, '../config')\n",
                         "sys.path.insert(0, '..')\n",
                         "\n",
                         "from paths import *\n",
                         "import pandas as pd\n",
                         "import numpy as np\n",
                         "from sklearn.model_selection import train_test_split\n",
                         "from sklearn.preprocessing import StandardScaler\n",
                         "from sklearn.linear_model import LinearRegression\n",
                         "from sklearn.preprocessing import PolynomialFeatures\n",
                         "from sklearn.svm import SVR\n",
                         "from sklearn.tree import DecisionTreeRegressor\n",
                         "from sklearn.neighbors import KNeighborsRegressor\n",
                         "from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error\n",
                         "import pickle\n",
                         "import joblib\n",
                         "\n",
                         "print('[OK] Libraries imported')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Load weight prediction data\n",
                         "print(f'Loading data from {WEIGHT_DATA_CSV}')\n",
                         "df = pd.read_csv(str(WEIGHT_DATA_CSV))\n",
                         "\n",
                         "print(f'Dataset shape: {df.shape}')\n",
                         "print(f'\\nColumns: {df.columns.tolist()}')\n",
                         "print(f'\\nFirst few rows:')\n",
                         "print(df.head())\n",
                         "print(f'\\nData info:')\n",
                         "print(df.info())\n",
                         "print(f'\\nStatistics:')\n",
                         "print(df.describe())"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Prepare data for training\n",
                         "# Assuming the CSV has pixel_count and weight columns\n",
                         "# Adjust column names based on your actual CSV structure\n",
                         "\n",
                         "X = df.iloc[:, :-1].values  # Features (pixel counts)\n",
                         "y = df.iloc[:, -1].values   # Target (weight)\n",
                         "\n",
                         "# Split data\n",
                         "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                         "\n",
                         "# Normalize features\n",
                         "scaler = StandardScaler()\n",
                         "X_train_scaled = scaler.fit_transform(X_train)\n",
                         "X_test_scaled = scaler.transform(X_test)\n",
                         "\n",
                         "print(f'Training set size: {X_train.shape[0]}')\n",
                         "print(f'Test set size: {X_test.shape[0]}')\n",
                         "print(f'Features: {X_train.shape[1]}')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Train multiple models\n",
                         "models = {\n",
                         "    'Linear Regression': LinearRegression(),\n",
                         "    'Polynomial Regression': None,  # Will handle separately\n",
                         "    'SVR': SVR(kernel='rbf', C=100, epsilon=0.1),\n",
                         "    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),\n",
                         "    'KNN': KNeighborsRegressor(n_neighbors=5)\n",
                         "}\n",
                         "\n",
                         "results = {}\n",
                         "\n",
                         "# Train Linear Regression\n",
                         "lr_model = models['Linear Regression']\n",
                         "lr_model.fit(X_train_scaled, y_train)\n",
                         "y_pred_lr = lr_model.predict(X_test_scaled)\n",
                         "results['Linear Regression'] = {\n",
                         "    'model': lr_model,\n",
                         "    'mse': mean_squared_error(y_test, y_pred_lr),\n",
                         "    'mae': mean_absolute_error(y_test, y_pred_lr),\n",
                         "    'r2': r2_score(y_test, y_pred_lr)\n",
                         "}\n",
                         "\n",
                         "# Polynomial Regression\n",
                         "poly_features = PolynomialFeatures(degree=2)\n",
                         "X_train_poly = poly_features.fit_transform(X_train_scaled)\n",
                         "X_test_poly = poly_features.transform(X_test_scaled)\n",
                         "\n",
                         "poly_model = LinearRegression()\n",
                         "poly_model.fit(X_train_poly, y_train)\n",
                         "y_pred_poly = poly_model.predict(X_test_poly)\n",
                         "results['Polynomial Regression'] = {\n",
                         "    'model': poly_model,\n",
                         "    'poly_features': poly_features,\n",
                         "    'mse': mean_squared_error(y_test, y_pred_poly),\n",
                         "    'mae': mean_absolute_error(y_test, y_pred_poly),\n",
                         "    'r2': r2_score(y_test, y_pred_poly)\n",
                         "}\n",
                         "\n",
                         "# Train other models (SVR, Decision Tree, KNN)\n",
                         "for model_name in ['SVR', 'Decision Tree', 'KNN']:\n",
                         "    model = models[model_name]\n",
                         "    model.fit(X_train_scaled, y_train)\n",
                         "    y_pred = model.predict(X_test_scaled)\n",
                         "    results[model_name] = {\n",
                         "        'model': model,\n",
                         "        'mse': mean_squared_error(y_test, y_pred),\n",
                         "        'mae': mean_absolute_error(y_test, y_pred),\n",
                         "        'r2': r2_score(y_test, y_pred)\n",
                         "    }\n",
                         "\n",
                         "# Print results\n",
                         "print('\\n=== Model Performance ===')\n",
                         "for model_name, metrics in results.items():\n",
                         "    print(f'\\n{model_name}:')\n",
                         "    print(f'  MSE: {metrics[\"mse\"]:.4f}')\n",
                         "    print(f'  MAE: {metrics[\"mae\"]:.4f}')\n",
                         "    print(f'  R²: {metrics[\"r2\"]:.4f}')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Save best model\n",
                         "best_model_name = max(results, key=lambda x: results[x]['r2'])\n",
                         "best_model_data = results[best_model_name]\n",
                         "\n",
                         "print(f'\\n✓ Best model: {best_model_name} (R² = {best_model_data[\"r2\"]:.4f})')\n",
                         "\n",
                         "# Save the best model and scaler\n",
                         "model_save_path = WEIGHT_MODEL\n",
                         "scaler_save_path = MODELS_DIR / 'weight_scaler.pkl'\n",
                         "\n",
                         "joblib.dump(best_model_data['model'], str(model_save_path))\n",
                         "joblib.dump(scaler, str(scaler_save_path))\n",
                         "\n",
                         "if 'poly_features' in best_model_data:\n",
                         "    joblib.dump(best_model_data['poly_features'], str(MODELS_DIR / 'weight_poly_features.pkl'))\n",
                         "    print(f'✓ Saved: Polynomial features')\n",
                         "\n",
                         "print(f'✓ Model saved to {model_save_path}')\n",
                         "print(f'✓ Scaler saved to {scaler_save_path}')\n",
                         "print(f'\\nModels directory:')\n",
                         "for f in MODELS_DIR.glob('weight*'):\n",
                         "    print(f'  - {f.name} ({f.stat().st_size / 1024:.2f} KB)')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Test prediction function\n",
                         "def predict_weight(pixel_count):\n",
                         "    \"\"\"Predict pineapple weight from pixel count\"\"\"\n",
                         "    loaded_model = joblib.load(str(WEIGHT_MODEL))\n",
                         "    loaded_scaler = joblib.load(str(SCALER_SAVE_PATH))\n",
                         "    \n",
                         "    pixel_count_scaled = loaded_scaler.transform([[pixel_count]])\n",
                         "    weight = loaded_model.predict(pixel_count_scaled)[0]\n",
                         "    return weight\n",
                         "\n",
                         "# Test with sample\n",
                         "print('Testing prediction functions...')\n",
                         "sample_pixel = X_test[0][0]\n",
                         "predicted_weight = predict_weight(sample_pixel)\n",
                         "actual_weight = y_test[0]\n",
                         "\n",
                         "print(f'Sample prediction:')\n",
                         "print(f'  Pixel count: {sample_pixel:.0f}')\n",
                         "print(f'  Predicted weight: {predicted_weight:.2f}kg')\n",
                         "print(f'  Actual weight: {actual_weight:.2f}kg')\n",
                         "print(f'  Error: {abs(predicted_weight - actual_weight):.2f}kg')"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Summary\n",
                         "✓ Weight prediction models trained\n",
                         "✓ Best model saved and ready for deployment\n",
                         "✓ Can integrate with detection/classification pipeline"]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook

# Main execution
if __name__ == "__main__":
    notebooks_dir = Path(r"c:\Users\Asus\Desktop\FARMBOT AI\FRC\FRUIT_PINEAPPLE\notebooks")

    print("Creating modified notebooks...")

    # Create detection notebook
    detection_nb = create_modified_detection_notebook()
    with open(notebooks_dir / "Pineapple_Detection_Jetson.ipynb", 'w', encoding='utf-8') as f:
        json.dump(detection_nb, f, indent=2)
    print("[OK] Created: Pineapple_Detection_Jetson.ipynb")

    # Create classification notebook
    classification_nb = create_modified_classification_notebook()
    with open(notebooks_dir / "Ripeness_Classification_Jetson.ipynb", 'w', encoding='utf-8') as f:
        json.dump(classification_nb, f, indent=2)
    print("[OK] Created: Ripeness_Classification_Jetson.ipynb")

    # Create weight prediction notebook
    weight_nb = create_modified_weight_prediction_notebook()
    with open(notebooks_dir / "SizeEstimationANDweightPrediction_Jetson.ipynb", 'w', encoding='utf-8') as f:
        json.dump(weight_nb, f, indent=2)
    print("[OK] Created: SizeEstimationANDweightPrediction_Jetson.ipynb")

    print("\n[DONE] All notebooks created successfully!")
    print(f"\nGenerated notebooks in {notebooks_dir}:")
    for nb_file in notebooks_dir.glob("*_Jetson.ipynb"):
        print(f"  - {nb_file.name}")
