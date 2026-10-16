# Model card: GreenLedger SeaGuard vessel detector

## Summary

| | |
|---|---|
| Task | Object detection, one class: `Vessel` |
| Base model | YOLOv8m (Ultralytics), COCO-pretrained |
| Input size | 512 × 512 |
| Training | 200 epochs, early-stopping patience 50, Colab GPU |
| Deployed as | `best.pt` (≈50 MB) in the `greenledger-detector` Cloud Run service |
| Date | 14 October 2026 |

## Data

| Split | Images |
|---|---:|
| Train | 187 |
| Validation | 54 |
| Test (held out) | 27 (35 vessels) |

- Source: our own labelled project in Roboflow (`greenledger-vessels`). Vessel type is not labelled in v2; see below.
- Version 1 used four vessel classes that had been labelled inconsistently, which hurt the detector (F1 34.6). Version 2 merges them into one `Vessel` class; vessel **type** is then decided by Gemini on each crop.
- Preprocessing: resize to 512. No augmentation in the dataset export; YOLO's default training augmentations apply.
- 1,107 further images are uploaded but not yet labelled.

## Results

| Split | mAP50 | mAP50-95 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Validation | 78.4 | 48.0 | 89.2 | 69.6 | 78.2 |
| **Test** | **89.2** | **56.4** | **86.5** | **82.9** | **84.6** |

Full numbers: [`training/metrics.json`](../training/metrics.json). Per-epoch curves: [`training/results.csv`](../training/results.csv).

Inference speed on the Colab GPU was about 19 ms per image. The Cloud Run service runs on CPU, which is slower but adequate for a dashboard.

The validation split scores lower than the test split. With only 54 and 27 images, a few hard images move these numbers a lot. The honest reading is "about 80–90 mAP50 on similar images", not a precise figure.

## Intended use

Decision support for watch officers and VTS operators: find vessels in camera frames so they can be classified and checked against COLREG. Not for autonomous navigation or any control of a ship.

## Known limitations

- Small dataset; offshore construction and DP vessels are under-represented, so their confidence is lower.
- Day-time images dominate. Night, heavy fog and heavy rain need more data; the dashboard can add synthetic fog and rain (`sim/simulate.py`) for demonstration only.
- At the low 0.20 threshold the detector occasionally boxes the same vessel twice. We accept this to keep recall high.
- A single frame gives no heading, so COLREG encounter geometry needs AIS or radar.

## Next steps

Label the waiting images, add offshore, DP, night and fog scenes, and compare against the COCO `boat` baseline on the same test split.
