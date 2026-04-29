# FARMBOT AI Dashboard

This folder contains a PC dashboard for the agricultural bot. It is designed for the final setup where the Jetson and your PC are on the same Wi-Fi/LAN, while still working today in simulation mode before the bot is fully ready.

## What This Dashboard Shows

- USB camera live feed for leaf disease and fruit detection output.
- Depth camera live feed.
- 7-in-1 sensor values: N, P, K, EC, pH, moisture, temperature, and growth stage.
- NPK Pineapple model predictions from `../NPK Pineapple/model/pineapple_model_*.onnx`.
- Leaf and fruit analysis fields: disease/health, fruit count, ripeness, and estimated weight.
- Sensor trend history and recent system events.

The PC dashboard does not need to directly own the Jetson cameras. The Jetson serves camera streams and publishes telemetry; the PC dashboard displays them.

## Folder Contents

```text
Dashboard/
  server.py                 # PC dashboard backend and telemetry receiver
  jetson_camera_server.py   # Optional Jetson MJPEG camera stream server
  jetson_publisher.py       # Jetson telemetry publisher scaffold
  requirements.txt          # PC dashboard dependencies
  static/
    index.html
    styles.css
    app.js
```

## Quick Start on Your PC

From PowerShell:

```powershell
cd "C:\Users\Asus\Desktop\FARMBOT AI\FRC\Dashboard"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python server.py --host 0.0.0.0 --port 8000
```

Open this on the PC:

```text
http://localhost:8000
```

The dashboard starts in simulation mode, so you should immediately see changing values even without the Jetson.

## How the NPK Model Is Used

`server.py` automatically tries to load:

```text
../NPK Pineapple/model/pineapple_model_delta_N.onnx
../NPK Pineapple/model/pineapple_model_delta_P.onnx
../NPK Pineapple/model/pineapple_model_delta_K.onnx
../NPK Pineapple/model/pineapple_model_irrigation_ml.onnx
../NPK Pineapple/model/pineapple_model_pH_adj.onnx
```

It imports the existing `FeatureEngineer`, `SensorReading`, and `Predictor` classes from `NPK Pineapple`. If `onnxruntime` or model files are missing, the dashboard keeps running and marks predictions as `fallback`.

## Jetson Camera Streams

On the Jetson, install the dashboard dependencies plus OpenCV:

```bash
cd /path/to/FRC/Dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install opencv-python
```

Start the camera stream server:

```bash
python jetson_camera_server.py --host 0.0.0.0 --port 8090 --usb-index 0 --depth-index 1
```

Then enter these URLs in the dashboard stream boxes:

```text
http://JETSON_IP:8090/usb.mjpg
http://JETSON_IP:8090/depth.mjpg
```

If your depth camera is not visible as OpenCV camera index `1`, test camera indices on the Jetson and change `--depth-index`.

## Jetson Telemetry Publisher

While developing, you can publish simulated Jetson telemetry to the PC:

```bash
python jetson_publisher.py \
  --dashboard-url http://PC_IP:8000 \
  --usb-stream-url http://JETSON_IP:8090/usb.mjpg \
  --depth-stream-url http://JETSON_IP:8090/depth.mjpg
```

When this publisher connects, the PC dashboard automatically disables its internal simulation and switches to Jetson telemetry.

## Telemetry API Contract

The Jetson should POST JSON to:

```text
http://PC_IP:8000/api/telemetry
```

Example payload:

```json
{
  "sensors": {
    "N": 123.4,
    "P": 41.8,
    "K": 178.2,
    "EC": 910,
    "pH": 5.58,
    "moisture": 64.2,
    "temp": 27.1,
    "growth_stage": 2
  },
  "vision": {
    "leaf_status": "Healthy",
    "leaf_confidence": 0.91,
    "fruit_count": 2,
    "ripeness": "Semi-ripe",
    "ripeness_confidence": 0.86,
    "estimated_weight_kg": 1.34
  },
  "streams": {
    "usb_cam": "http://JETSON_IP:8090/usb.mjpg",
    "depth_cam": "http://JETSON_IP:8090/depth.mjpg"
  },
  "system": {
    "jetson_name": "farmbot-jetson",
    "fps_usb": 24.5,
    "fps_depth": 18.0,
    "latency_ms": 42
  },
  "events": [
    {"level": "info", "message": "Telemetry loop healthy"}
  ]
}
```

If the Jetson does not send `predictions`, the PC dashboard computes NPK predictions using the local `NPK Pineapple` ONNX models. If the Jetson already runs the models, include a `predictions` object and the dashboard will display it.

## Connecting the Real 7-in-1 Sensor

The NPK project already contains a real sensor path:

```text
../NPK Pineapple/sensors/sensor_real.py
../NPK Pineapple/config/system_config.yaml
```

Next integration step:

1. Confirm the RS485/Modbus port on Jetson, usually `/dev/ttyUSB0`.
2. Confirm register scaling for N, P, K, EC, pH, moisture, and temperature.
3. Replace the simulated values in `jetson_publisher.py` with a call to the real sensor reader.
4. POST the resulting readings to `/api/telemetry`.

Keep the field names exactly as shown in the API contract: `N`, `P`, `K`, `EC`, `pH`, `moisture`, `temp`, `growth_stage`.

## Connecting Fruit and Leaf Models

The fruit project already has these exported assets:

```text
../FRUIT_PINEAPPLE/models/detection_model.onnx
../FRUIT_PINEAPPLE/models/weight_prediction.onnx
```

Recommended Jetson flow:

1. Read frame from USB camera.
2. Run pineapple detection model.
3. Crop detected fruit regions.
4. Run ripeness classifier when the classification model is downloaded/exported.
5. Use depth or measured fruit dimensions with `weight_prediction.onnx`.
6. Send the final `vision` object to the dashboard.

The ripeness classification ONNX model is not currently present locally according to `../ONNX_MODELS_SUMMARY.md`; download/export it before wiring true ripeness inference.

Leaf disease detection is not currently present as a separate model in the repo. The dashboard already has fields for it, so once the model is trained/exported, publish:

```json
{
  "vision": {
    "leaf_status": "Bacterial spot",
    "leaf_confidence": 0.88
  }
}
```

## Network Checklist

1. Put PC and Jetson on the same Wi-Fi/LAN.
2. Find PC IP:
   - Windows: `ipconfig`
   - Use the IPv4 address on the active Wi-Fi/Ethernet adapter.
3. Find Jetson IP:
   - Jetson: `hostname -I`
4. Start PC dashboard on `0.0.0.0:8000`.
5. From Jetson, test:

```bash
curl http://PC_IP:8000/api/state
```

6. From PC browser, test camera streams:

```text
http://JETSON_IP:8090/usb.mjpg
http://JETSON_IP:8090/depth.mjpg
```

If the dashboard opens on the PC but the Jetson cannot POST, allow Python through Windows Firewall for private networks.

## Useful Endpoints

```text
GET  /                  Dashboard UI
GET  /api/state         Current dashboard state
POST /api/telemetry     Jetson telemetry receiver
POST /api/simulation/true
POST /api/simulation/false
```

## Next Steps

1. Run the PC dashboard and verify simulation mode.
2. Run `jetson_camera_server.py` on the Jetson and confirm both stream URLs open in the PC browser.
3. Run `jetson_publisher.py` from the Jetson and confirm the dashboard switches from `SIMULATED` to `CONNECTED`.
4. Replace simulated sensor values in `jetson_publisher.py` with the real 7-in-1 sensor reader.
5. Add the fruit inference pipeline on the Jetson and publish `vision` values.
6. Add or export the missing leaf disease and ripeness models, then publish their predictions to the same telemetry endpoint.
