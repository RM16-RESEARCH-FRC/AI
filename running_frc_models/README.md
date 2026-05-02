# Jetson FRC Model Inference Setup

## Quick Start (3 Steps)

### Step 1: Test Models
```bash
chmod +x ~/running_frc_models/1_test_models.sh
~/running_frc_models/1_test_models.sh
```

### Step 2: Start Camera Server (Terminal 1)
```bash
chmod +x ~/running_frc_models/2_start_camera_server.sh
~/running_frc_models/2_start_camera_server.sh
```

### Step 3: Start Inference Pipeline (Terminal 2)
```bash
chmod +x ~/running_frc_models/3_start_inference.sh
~/running_frc_models/3_start_inference.sh
```

Then open browser: **http://10.137.170.216:8001**

---

## What Each Script Does

| Script | Purpose | Terminal |
|--------|---------|----------|
| `1_test_models.sh` | Verify all 8 ONNX models load correctly | Any |
| `2_start_camera_server.sh` | Stream USB camera to PC | Terminal 1 |
| `3_start_inference.sh` | Run fruit/leaf models + send telemetry | Terminal 2 |

---

## What's Running

### Camera Server (Port 8090)
- Streams USB camera via MJPEG
- URL: `http://jetson-ip:8090/usb.mjpg`

### Inference Pipeline
- Reads USB camera frames
- Runs fruit detection model
- Runs leaf disease model
- Generates simulated NPK sensor data
- Posts telemetry to PC dashboard every 8 seconds

### Output
Dashboard shows in real-time:
- ✓ USB camera feed
- ✓ Fruit count & ripeness confidence
- ✓ Leaf disease status & severity
- ✓ Simulated NPK sensor values (N, P, K, EC, pH, moisture, temp)

---

## Troubleshooting

**Models fail to load?**
```bash
python3 test_models.py
# Check output for missing ONNX files or runtime errors
```

**Camera not found?**
```bash
# List camera devices
ls /dev/video*

# If /dev/video0 doesn't exist, check USB connection
# Use --skip-camera flag to test without camera:
python3 jetson_inference.py --dashboard-url http://PC_IP:8001 --skip-camera
```

**Can't connect to dashboard?**
- Check PC IP in `3_start_inference.sh` is correct
- Check firewall allows port 8001
- Test: `curl http://PC_IP:8001`

---

## Files

| File | Purpose |
|------|---------|
| `test_models.py` | Test ONNX model loading |
| `jetson_inference.py` | Main inference + telemetry publisher |
| `jetson_camera_server.py` | Camera stream server |
| `detection_model.onnx` | Fruit detection |
| `leaf_detection.onnx` | Leaf disease detection |
| `pineapple_model_*.onnx` | NPK control models (5 files) |
| `weight_prediction.onnx` | Fruit weight estimation |
