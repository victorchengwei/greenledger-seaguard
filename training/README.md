# Training

The deployed detector was trained in Google Colab on a GPU.

| Setting | Value |
|---|---|
| Dataset | Roboflow project `greenledger-vessels`, version 2 (one class: `Vessel`) |
| Base weights | `yolov8m.pt` |
| Image size | 512 |
| Epochs | 200 (early-stopping patience 50) |
| Splits | 187 train / 54 validation / 27 test |

Reproduce with Ultralytics:

```python
from ultralytics import YOLO
model = YOLO("yolov8m.pt")
model.train(data="data.yaml", epochs=200, patience=50, imgsz=512)
model.val(data="data.yaml", split="test")
```

Files here:

- `metrics.json`: validation and test metrics of the deployed model
- `results.csv`: per-epoch training log
- Curves and confusion matrices: `../docs/images/`
