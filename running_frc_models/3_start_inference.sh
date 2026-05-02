#!/bin/bash
# Run inference pipeline: USB cam → models → dashboard
# Edit PC_IP below to your actual PC IP address

PC_IP="10.137.170.216"
DASHBOARD_PORT="8001"

cd ~/running_frc_models
python3 jetson_inference.py \
  --dashboard-url "http://${PC_IP}:${DASHBOARD_PORT}" \
  --usb-camera 0 \
  --interval 8.0
