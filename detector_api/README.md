# Detector API

FastAPI service that runs the fine-tuned YOLOv8m vessel detector.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check; reports the loaded model file |
| `POST /detect?conf=0.20` | Upload one image (`file`); returns boxes, confidences, a 320 px crop per vessel (base64) and an annotated image |

## Model weights

`best.pt` (≈50 MB) is not committed to git. Download it from this repository's **Releases** page and place it in this folder.

| Variable | Meaning |
|---|---|
| `MODEL_PATH` | Weights file, e.g. `best.pt` |
| `KEEP_CLASSES` | `all` for our one-class model; `boat` to use a COCO model as a baseline |

Without `best.pt` you can still try the service with a COCO model: `MODEL_PATH=yolov8n.pt KEEP_CLASSES=boat` (Ultralytics downloads it on first use).
