"""
Utilities for model quantization and optimization for Jetson deployment
"""
import torch
import torch.quantization as quantization
from pathlib import Path

def quantize_yolo_model(model_path: Path, output_path: Path, device: str = "cpu"):
    """
    Quantize YOLOv8 model for Jetson deployment using INT8 quantization

    Args:
        model_path: Path to the original model
        output_path: Path to save the quantized model
        device: Device to use (cpu or cpu for quantization, gpu for inference)
    """
    try:
        from ultralytics import YOLO

        # Load the model
        model = YOLO(str(model_path))

        # Export to TorchScript with INT8 quantization
        # This creates an optimized model for edge devices
        model.export(
            format="torchscript",
            device=device,
            half=True,  # FP16 quantization as intermediate
            int8=True,  # INT8 quantization
            optimize=True,
            dynamic=False
        )

        # Save the quantized model
        model.save(str(output_path))
        print(f"✓ Model quantized and saved to {output_path}")
        return output_path

    except Exception as e:
        print(f"✗ Quantization failed: {str(e)}")
        return None

def quantize_pytorch_model(model, output_path: Path, sample_input):
    """
    Quantize a PyTorch model using dynamic quantization

    Args:
        model: PyTorch model to quantize
        output_path: Path to save the quantized model
        sample_input: Sample input tensor for tracing (optional)
    """
    try:
        # Apply dynamic quantization (good for CPU inference)
        quantized_model = quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        torch.save(quantized_model.state_dict(), str(output_path))
        print(f"✓ PyTorch model quantized and saved to {output_path}")
        return quantized_model

    except Exception as e:
        print(f"✗ PyTorch quantization failed: {str(e)}")
        return None

def convert_to_onnx(model_path: Path, output_path: Path, model_type: str = "detection"):
    """
    Convert model to ONNX format for broad compatibility

    Args:
        model_path: Path to the model
        output_path: Path to save ONNX model
        model_type: Type of model (detection, classification, etc.)
    """
    try:
        import onnx
        from ultralytics import YOLO

        if model_type in ["detection", "classification", "segmentation"]:
            model = YOLO(str(model_path))
            model.export(format="onnx", opset=13)
            onnx_path = str(model_path).replace('.pt', '.onnx')
            print(f"✓ Model converted to ONNX: {onnx_path}")
            return Path(onnx_path)
        else:
            print(f"✗ Unsupported model type for conversion: {model_type}")
            return None

    except Exception as e:
        print(f"✗ ONNX conversion failed: {str(e)}")
        return None

def get_model_info(model_path: Path):
    """
    Get information about a trained model

    Args:
        model_path: Path to the model
    """
    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        info = {
            "model_path": str(model_path),
            "model_size_mb": model_path.stat().st_size / (1024 * 1024),
            "task": model.task,
            "model": model.model.__class__.__name__
        }

        print("=== Model Information ===")
        for key, value in info.items():
            print(f"{key}: {value}")

        return info

    except Exception as e:
        print(f"✗ Failed to get model info: {str(e)}")
        return None

if __name__ == "__main__":
    print("Model optimization utilities for Jetson deployment")
    print("Import this module in your notebooks to use quantization functions")
