"""
Centralized path configuration for FRC Fruit Pineapple project
"""
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
DETECTION_DATA = DATA_DIR / "detection"
CLASSIFICATION_DATA = DATA_DIR / "classification"
WEIGHT_DATA_CSV = DATA_DIR / "weight prediction data.csv"

# Notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DETECTION_NOTEBOOK = NOTEBOOKS_DIR / "Pineapple_Detection.ipynb"
CLASSIFICATION_NOTEBOOK = NOTEBOOKS_DIR / "Ripeness_Classification.ipynb"
WEIGHT_PREDICTION_NOTEBOOK = NOTEBOOKS_DIR / "SizeEstimationANDweightPrediction.ipynb"

# Models
MODELS_DIR = PROJECT_ROOT / "models"
DETECTION_MODEL = MODELS_DIR / "detection_model.pt"
DETECTION_MODEL_QUANTIZED = MODELS_DIR / "detection_model_quantized.pt"
CLASSIFICATION_MODEL = MODELS_DIR / "classification_model.pt"
CLASSIFICATION_MODEL_QUANTIZED = MODELS_DIR / "classification_model_quantized.pt"
WEIGHT_MODEL = MODELS_DIR / "weight_prediction_model.pkl"

# Outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Detection data paths
DETECTION_IMAGES_TRAIN = DETECTION_DATA / "detection" / "images" / "train"
DETECTION_LABELS_TRAIN = DETECTION_DATA / "detection" / "labels" / "train"
DETECTION_IMAGES_VAL = DETECTION_DATA / "detection" / "images" / "val"
DETECTION_LABELS_VAL = DETECTION_DATA / "detection" / "labels" / "val"
DETECTION_IMAGES_TEST = DETECTION_DATA / "detection" / "images" / "test"

# Classification data paths
CLASSIFICATION_IMAGES = CLASSIFICATION_DATA / "molvumClassification" / "images"
CLASSIFICATION_TRAIN = CLASSIFICATION_IMAGES / "train"
CLASSIFICATION_TEST = CLASSIFICATION_IMAGES / "test"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def print_paths():
    """Print all configured paths"""
    print("=== Project Paths ===")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"DETECTION_DATA: {DETECTION_DATA}")
    print(f"CLASSIFICATION_DATA: {CLASSIFICATION_DATA}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"OUTPUTS_DIR: {OUTPUTS_DIR}")

if __name__ == "__main__":
    print_paths()
