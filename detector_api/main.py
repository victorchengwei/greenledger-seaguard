"""YOLOv8 vessel detector as an HTTP API (deploy on Cloud Run)."""
import base64
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from ultralytics import YOLO

# Set MODEL_PATH=best.pt to use your own trained weights (put the file in this folder).
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
model = YOLO(MODEL_PATH)
# Keep only these class names ("all" keeps everything). COCO has a single generic "boat" class.
KEEP = set(os.getenv("KEEP_CLASSES", "boat").split(","))
CLASS_IDS = None if "all" in KEEP else ([i for i, n in model.names.items() if n in KEEP] or None)
app = FastAPI(title="GreenLedger vessel detector")


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_PATH}


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.25):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Not a valid image")
    res = model.predict(img, conf=conf, classes=CLASS_IDS, verbose=False)[0]
    dets = []
    for b in res.boxes:
        x1, y1, x2, y2 = [round(v, 1) for v in b.xyxy[0].tolist()]
        crop = img[max(int(y1), 0):int(y2), max(int(x1), 0):int(x2)]
        crop_b64 = ""
        if crop.size:
            scale = 320 / max(crop.shape[:2])
            if scale < 1:
                crop = cv2.resize(crop, None, fx=scale, fy=scale)
            crop_b64 = base64.b64encode(cv2.imencode(".jpg", crop)[1]).decode()
        dets.append({"class": res.names[int(b.cls[0])],
                     "conf": round(float(b.conf[0]), 3),
                     "bbox": [x1, y1, x2, y2],
                     "crop_b64": crop_b64})
    ok, buf = cv2.imencode(".jpg", res.plot())
    return {"detections": dets, "annotated_jpg_b64": base64.b64encode(buf).decode()}
