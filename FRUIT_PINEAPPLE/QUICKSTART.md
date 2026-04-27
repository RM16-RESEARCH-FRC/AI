## Quick Start Guide - FRC Fruit Pineapple

### 1. Environment Setup (One-time)

```bash
# Navigate to project directory
cd "path/to/FRUIT_PINEAPPLE"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Data Structure

```bash
# Check that all datasets are present
ls -la data/

# Should show:
# - detection/       (399 MB) - Pineapple detection images and labels
# - classification/  (2.0 GB) - Ripeness classification images
# - weight prediction data.csv  - Weight dataset
```

### 3. Run Notebooks

```bash
# Start Jupyter
jupyter notebook

# In your browser, open and run these in order:
```

#### Option A: Train All Models (Recommended)
1. **Pineapple_Detection_Jetson.ipynb**
   - Trains YOLOv8n for pineapple detection
   - Takes ~2-4 hours on GPU
   - Saves model to `models/detection_model.pt`

2. **Ripeness_Classification_Jetson.ipynb**
   - Trains YOLOv8n for ripeness classification
   - Takes ~1-2 hours on GPU
   - Saves model to `models/classification_model.pt`

3. **SizeEstimationANDweightPrediction_Jetson.ipynb**
   - Trains weight prediction models
   - Takes ~5-10 minutes
   - Saves model to `models/weight_prediction_model.pkl`

#### Option B: Use Pre-trained Models (If Available)
- If models are already in `models/` directory, skip training
- Use notebooks for inference/evaluation only

### 4. Check Outputs

```bash
# View generated metrics and models
ls -lh models/          # Show trained models
ls -lh outputs/         # Show validation results
```

### 5. Deploy Models

```bash
# Option 1: Deploy FP16 quantized models (Recommended for Jetson)
# - ~50% smaller than full precision
# - Nearly same accuracy
# - Better speed on Jetson

# Option 2: Deploy ONNX models
# - Works on any device with ONNX Runtime
# - Broad compatibility

# See PROJECT_STRUCTURE.md for deployment details
```

### 6. Use Trained Models in Python

```python
from config.paths import *
from ultralytics import YOLO
import joblib

# Detection
detection_model = YOLO(str(DETECTION_MODEL))
results = detection_model.predict(source="image.jpg")
print(results)

# Classification
classification_model = YOLO(str(CLASSIFICATION_MODEL))
class_results = classification_model.predict(source="image.jpg")
print(class_results)

# Weight Prediction
weight_model = joblib.load(str(WEIGHT_MODEL))
scaler = joblib.load(str(MODELS_DIR / "weight_scaler.pkl"))
predicted_weight = weight_model.predict(scaler.transform([[pixel_count]]))
print(f"Predicted weight: {predicted_weight[0]:.2f} kg")
```

### Troubleshooting

**Can't run notebooks - NotFoundError on data**
```bash
# Make sure you extracted the zip files:
# The script already did this, but verify:
ls -la data/detection/detection/images/
ls -la data/classification/molvumClassification/images/
```

**Not enough GPU memory**
```python
# In the notebooks, reduce batch size:
batch=8  # Instead of default
# Or use CPU (slower):
device='cpu'
```

**Import errors when running notebooks**
```python
# If config module not found, add to first cell:
import sys
sys.path.insert(0, '..')  # Go up one directory
from config.paths import *
```

**Models not saving to correct location**
```bash
# Verify models directory exists:
mkdir -p models/
# Check paths.py is correct:
python -c "from config.paths import MODELS_DIR; print(MODELS_DIR)"
```

### Performance Expectations

| Task | Time (GPU) | Model Size | Inference Speed |
|------|-----------|-----------|-----------------|
| Detection Training (100 epochs) | 2-4 hrs | 6 MB | 20ms/image |
| Classification Training (100 epochs) | 1-2 hrs | 4 MB | 10ms/image |
| Weight Prediction Training | <1 min | <1 MB | <1ms |

### Next: Deploy to Jetson

Once models are trained:

1. **Export models** (run in notebooks before deployment)
   ```python
   model.export(format='torchscript', half=True)  # FP16
   model.export(format='onnx')  # ONNX
   ```

2. **Transfer to Jetson**
   ```bash
   scp -r models/ jetson_user@jetson_ip:~/fruit-pineapple/
   ```

3. **Run inference on Jetson**
   ```python
   from ultralytics import YOLO
   model = YOLO('detection_model_fp16.pt')
   for frame in video_stream:
       results = model(frame)
   ```

### File Locations Quick Reference

| Item | Location |
|------|----------|
| Training Notebooks | `notebooks/*_Jetson.ipynb` |
| Datasets | `data/` |
| Trained Models | `models/` |
| Config Files | `config/` |
| Results/Benchmarks | `outputs/` |
| Documentation | `PROJECT_STRUCTURE.md` |

### Support

- See `PROJECT_STRUCTURE.md` for detailed architecture
- See `config/jetson_utils.py` for quantization options
- See `config/paths.py` for all path definitions
- Check notebook outputs for training metrics and warnings

Happy training! 🍍
