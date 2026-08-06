# Training v3

Training v3 keeps the detector and string label contracts, excludes all hand
and pose annotations from datasets, and trains orientation from yoyo pixels
without hand/string crop geometry. RTMPose-m WholeBody is runtime-only.

Run every command through the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r training_v3\requirements-rtmpose.txt
.\.venv\Scripts\python.exe -m pip install rtmlib==0.0.16 --no-deps
.\.venv\Scripts\python.exe -m training_v3.download_rtmpose_models
.\.venv\Scripts\python.exe -m training_v3.prepare_dataset --source annotations\reviewed_export --output-dir datasets\1Ayoyo_dataset --clear
.\.venv\Scripts\python.exe -m training_v3.orientation_view --dataset-dir datasets\1Ayoyo_dataset --clear
```

The model downloader writes only to `models/rtmpose`. No RTMLib user-cache
path is used. Detection and string training do not contain hand fields. The v3
orientation view records `hands_pixel: false` and `string_geometry: false` in
its manifest so the independence claim is mechanically auditable.
