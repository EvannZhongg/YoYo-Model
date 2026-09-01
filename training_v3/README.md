# Training v3

Training v3 builds detection, string segmentation, and orientation views from
the canonical labels. Orientation is trained from yoyo pixels. RTMPose-m
WholeBody is used at runtime.

Run every command through the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r training_v3\requirements-rtmpose.txt
.\.venv\Scripts\python.exe -m pip install rtmlib==0.0.16 --no-deps
.\.venv\Scripts\python.exe -m training_v3.download_rtmpose_models
.\.venv\Scripts\python.exe -m training_v3.prepare_dataset --source annotations\reviewed_export --output-dir datasets\1Ayoyo_dataset --clear
.\.venv\Scripts\python.exe -m training_v3.orientation_view --dataset-dir datasets\1Ayoyo_dataset --clear
```

The model downloader writes only to `models/rtmpose`. No RTMLib user-cache
path is used. The v3 orientation view records `string_geometry: false` in its
manifest so its input dependencies are mechanically auditable. The independent
`training_v3.orientation_three` and `training_v3.orientation_four` modules
select the three `trick_orientation` classes or four `presentation_orientation`
classes respectively, while sharing only ROI materialization and manifest
generation. Runtime tracking accepts either model; four-class probabilities
are projected to the coarse three-way output consumed by tracking.
