## FRC Fruit Pineapple - Directory Restructure & Jetson Optimization

This guide explains the new project structure and how to use the modified notebooks for Jetson deployment.

### Project Structure

```
FRUIT_PINEAPPLE/
├── config/                          # Configuration files
│   ├── paths.py                    # Centralized path configuration
│   ├── jetson_utils.py            # Jetson optimization utilities
│   ├── create_notebooks.py        # Script to generate modified notebooks
│   ├── detection_data.yaml        # YOLO detection data config (auto-generated)
│   └── classification_data.yaml   # YOLO classification data config (auto-generated)
│
├── data/                            # Datasets
│   ├── detection/                 # Pineapple detection dataset
│   │   └── detection/
│   │       ├── images/
│   │       │   ├── train/
│   │       │   ├── val/
│   │       │   └── test/
│   │       └── labels/
│   │           ├── train/
│   │           └── val/
│   │
│   ├── classification/            # Ripeness classification dataset
│   │   └── molvumClassification/
│   │       └── images/
│   │           ├── train/
│   │           │   ├── ripe/
│   │           │   ├── unripe/
│   │           │   └── overripe/
│   │           └── test/
│   │               ├── ripe/
│   │               ├── unripe/
│   │               └── overripe/
│   │
│   └── weight prediction data.csv  # Weight prediction dataset
│
├── notebooks/                       # Jupyter notebooks (Jetson-optimized)
│   ├── Pineapple_Detection_Jetson.ipynb         # [NEW] Detection training
│   ├── Ripeness_Classification_Jetson.ipynb     # [NEW] Classification training
│   ├── SizeEstimationANDweightPrediction_Jetson.ipynb  # [NEW] Weight prediction
│   │
│   └── (old notebooks - for reference)
│       ├── Pineapple_Detection.ipynb
│       ├── Ripeness_Classification.ipynb
│       └── SizeEstimationANDweightPrediction.ipynb
│
├── models/                          # Trained and quantized models
│   ├── detection_model.pt          # Full precision detection model
│   ├── detection_model_quantized.pt # Quantized detection model
│   ├── classification_model.pt     # Full precision classification model
│   ├── classification_model_quantized.pt # Quantized classification model
│   ├── weight_prediction_model.pkl # Weight prediction model
│   ├── weight_scaler.pkl           # Feature scaler for weight prediction
│   └── weight_poly_features.pkl    # Polynomial features (if applicable)
│
├── outputs/                         # Training outputs and results
│   ├── validation_metrics.json
│   ├── inference_benchmarks.json
│   └── sample_predictions/
│
└── README.md                        # This file

```

### Key Improvements

#### 1. **Centralized Path Configuration** (`config/paths.py`)
- All path operations now use `config/paths.py`
- Single source of truth for all file locations
- Works across all notebooks without hardcoding paths

```python
from config.paths import *

# Use predefined paths
model_path = DETECTION_MODEL
data_path = DETECTION_DATA
output_dir = OUTPUTS_DIR
```

#### 2. **Jetson Optimization Utilities** (`config/jetson_utils.py`)
Provides functions for:
- **Model quantization** (INT8, FP16)
- **ONNX conversion** for broad compatibility
- **Model info extraction** for performance analysis

Functions available:
- `quantize_yolo_model()` - INT8 quantization for YOLOv8
- `quantize_pytorch_model()` - Dynamic quantization for PyTorch
- `convert_to_onnx()` - Convert to ONNX format
- `get_model_info()` - Extract model statistics

#### 3. **Enhanced Notebooks** (`*_Jetson.ipynb`)
New notebooks with integrated features:

**Pineapple_Detection_Jetson.ipynb**
- Automatic data.yaml generation
- YOLOv8n (Nano) model for Jetson compatibility
- Automatic model saving to `/models/`
- FP16 and ONNX export for edge deployment
- Inference speed benchmarking
- Model size monitoring

**Ripeness_Classification_Jetson.ipynb**
- YOLOv8 classification (nano variant)
- Automatic model saving and quantization
- Real-time inference performance testing
- Model export in multiple formats

**SizeEstimationANDweightPrediction_Jetson.ipynb**
- Trains multiple ML models (Linear, Polynomial, SVR, Decision Tree, KNN)
- Automatic selection of best model by R² score
- Saves model and scaler for inference
- Ready for integration with detection/classification

### How to Use

#### Step 1: Prepare Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install ultralytics torch torchvision opencv-python
pip install scikit-learn pandas numpy
pip install jupyter notebook
```

#### Step 2: Run Training Notebooks
```bash
# Start Jupyter
jupyter notebook

# Open and run:
# 1. notebooks/Pineapple_Detection_Jetson.ipynb
# 2. notebooks/Ripeness_Classification_Jetson.ipynb
# 3. notebooks/SizeEstimationANDweightPrediction_Jetson.ipynb
```

#### Step 3: Use Trained Models
```python
from config.paths import *
from ultralytics import YOLO
import joblib

# Load detection model
detection_model = YOLO(str(DETECTION_MODEL))
results = detection_model.predict(source="image.jpg")

# Load classification model
classification_model = YOLO(str(CLASSIFICATION_MODEL))
classification = classification_model.predict(source="image.jpg")

# Load weight prediction model
weight_model = joblib.load(str(WEIGHT_MODEL))
weight = weight_model.predict(pixel_count)
```

### Model Sizes (Approximate)

| Model | Size | Quantized | Speed |
|-------|------|-----------|-------|
| YOLOv8n Detection | 6 MB | 2 MB (FP16) | ~20ms |
| YOLOv8n Classification | 4 MB | 1.5 MB (FP16) | ~10ms |
| Weight Prediction | <1 MB | <100 KB | <1ms |

### For Jetson Deployment

#### Export Models:
1. **FP16 (Recommended for Jetson)**
   ```python
   model = YOLO('detection_model.pt')
   model.export(format='torchscript', half=True)  # ~50% smaller
   ```

2. **ONNX (Maximum Compatibility)**
   ```python
   model.export(format='onnx', opset=13)
   # Use with ONNX Runtime or TensorRT
   ```

3. **TensorRT (Best Performance on Jetson)**
   ```python
   model.export(format='engine', device=0)  # Requires TensorRT
   ```

#### Deploy on Jetson:
```python
from ultralytics import YOLO

# Load quantized model
model = YOLO('detection_model_fp16.pt')  # FP16 version

# Real-time inference
for frame in video_stream:
    results = model(frame, conf=0.5)
    # Process results...
```

### Performance Metrics

After training, notebooks will generate:
- Precision, Recall, mAP metrics for detection/classification
- Inference speed benchmarking (ms per image)
- Model size analysis
- R² score for weight prediction

Check `outputs/` directory for detailed metrics.

### Troubleshooting

**Issue: "ModuleNotFoundError: No module named 'config'"**
- Solution: Run notebooks from repository root or adjust Python path in first cell

**Issue: CUDA out of memory**
- Solution: Reduce batch size in training notebooks
- Alternative: Use `device='cpu'` for slower but memory-efficient training

**Issue: Data paths not found**
- Solution: Verify `data/` directory structure matches as shown above
- Check that `detection_data.yaml` and `classification_data.yaml` are generated

### Next Steps

1. **Train detection model** using `Pineapple_Detection_Jetson.ipynb`
2. **Train classification model** using `Ripeness_Classification_Jetson.ipynb`
3. **Train weight prediction** using `SizeEstimationANDweightPrediction_Jetson.ipynb`
4. **Export models** in FP16 or ONNX format
5. **Deploy to Jetson** using the exported models

### References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Jetson Development Kit](https://developer.nvidia.com/jetson)
- [ONNX Runtime Jetson](https://docs.nvidia.com/tensorrt/developer-guide/)

---
Generated: 2026-04-26
