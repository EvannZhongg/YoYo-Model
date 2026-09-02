# Training v4: centerline heatmap + direction field

This experiment treats the yoyo string as a geometric centerline rather than a
thick semantic mask. Targets are a Gaussian heatmap around the morphological
skeleton plus a unit vector pointing toward that skeleton within a six-pixel
context radius. The model is a MobileNetV3-Large FPN (or a small U-Net for
smoke tests) with a 3-channel head.

```powershell
.\.venv\Scripts\python.exe -m training_v4.train --dataset-dir datasets\1Ayoyo_dataset\string_segmentation --project runs\experiments --name centerline_v4_r1 --epochs 12 --architecture mobilenet_v3_fpn --pretrained-backbone --device cuda
.\.venv\Scripts\python.exe -m training_v4.evaluate --weights runs\experiments\centerline_v4_r1\weights\best.pt --dataset-dir datasets\1Ayoyo_dataset\string_segmentation --split test --device cuda --thresholds 0.25,0.35,0.5
.\.venv\Scripts\python.exe -m training_v4.evaluate_consecutive --weights runs\experiments\centerline_v4_r1\weights\best.pt --dataset-dir datasets\1Ayoyo_consecutive --device cuda --threshold 0.35
```

Each run writes `run_manifest.json` with dataset hash, source groups,
parameters, validation metrics and checkpoint hash. Thresholds must be chosen
on validation and frozen before test or consecutive-set evaluation.
