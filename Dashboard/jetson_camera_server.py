from __future__ import annotations

import argparse
import time
from typing import Generator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


try:
    import cv2
except ImportError:  # pragma: no cover - optional Jetson dependency.
    cv2 = None


app = FastAPI(title="FARMBOT Jetson Camera Streams")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

USB_INDEX = 0
DEPTH_INDEX = 1
WIDTH = 960
HEIGHT = 540
JPEG_QUALITY = 78


def mjpeg_generator(camera_index: int, label: str) -> Generator[bytes, None, None]:
    if cv2 is None:
        raise HTTPException(status_code=500, detail="opencv-python is not installed")

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    if not cap.isOpened():
        raise HTTPException(status_code=404, detail=f"{label} camera index {camera_index} could not be opened")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            cv2.putText(
                frame,
                f"{label} | {time.strftime('%H:%M:%S')}",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                continue

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
    finally:
        cap.release()


@app.get("/usb.mjpg")
def usb_stream() -> StreamingResponse:
    return StreamingResponse(mjpeg_generator(USB_INDEX, "USB"), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/depth.mjpg")
def depth_stream() -> StreamingResponse:
    return StreamingResponse(mjpeg_generator(DEPTH_INDEX, "Depth"), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/health")
def health() -> dict[str, int | str]:
    return {"status": "ok", "usb_index": USB_INDEX, "depth_index": DEPTH_INDEX}


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve USB and depth camera MJPEG streams from the Jetson.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--usb-index", type=int, default=0)
    parser.add_argument("--depth-index", type=int, default=1)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    global USB_INDEX, DEPTH_INDEX, WIDTH, HEIGHT
    USB_INDEX = args.usb_index
    DEPTH_INDEX = args.depth_index
    WIDTH = args.width
    HEIGHT = args.height

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
