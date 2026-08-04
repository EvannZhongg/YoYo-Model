---
name: create-yoyo-consecutive-blank-annotation-dataset
description: Create or incrementally extend Workbench-compatible blank yoyo/string annotation datasets from uninterrupted video frame runs, preferring runs near each video's temporal middle while preserving existing labels and SHA-bound human reviews. Use when a manual annotation queue needs consecutive frames, per-video runs, temporal clips, or different run lengths per source without VLM classification, model annotation, training, or approval.
---

# Create Yoyo Consecutive Blank Annotation Dataset

Use the repository virtual environment. The generator writes directly to
`datasets/<dataset-name>` and always selects one uninterrupted run per source.

```powershell
& PROJECT\.venv\Scripts\python.exe `
  SKILL_DIR\scripts\create_consecutive_blank_dataset.py `
  --videos VIDEO_OR_DIRECTORY `
  --dataset-name NEW_DATASET_NAME `
  --frames-per-video 30
```

Use `--videos-list` with one UTF-8 video path per line to control the source
set. Use `--total-frames N` only when counts may be allocated nearly equally
across all listed videos.

Use `--position-bias front` or `--position-bias back` when the requested run
should come from an eligible early or late part of the source. The default is
`middle`. The edge exclusion still follows `--edge-fraction`.

For different run lengths, create the first group and append each differently
sized group while the Workbench server is stopped:

```powershell
& PROJECT\.venv\Scripts\python.exe `
  SKILL_DIR\scripts\create_consecutive_blank_dataset.py `
  --videos-list FIRST_GROUP.txt --dataset-name DATASET --frames-per-video 30

& PROJECT\.venv\Scripts\python.exe `
  SKILL_DIR\scripts\create_consecutive_blank_dataset.py `
  --videos SECOND_VIDEO.mp4 --dataset-name DATASET --frames-per-video 100 --append
```

Require every command to finish with `"ok": true`. Confirm each source's
`run_start_frame` and `run_end_frame` in `sampling_manifest.json`, then verify
`consecutive_groups.json` contains the same uninterrupted frames and Workbench
sample keys.

The generator searches from the temporal middle outward. It preserves the root
`datasets/1Ayoyo_dataset` exclusion baseline and all earlier compatible blank
datasets. Provenance and exact image hashes remain unique within a run.
Perceptual similarity is allowed only among members of the same intentional
run; candidates are still checked perceptually against reference datasets and
previously completed runs. Use `--exclude-frame-window N` for wider separation
from known source frames.

Never append directly to `datasets/1Ayoyo_dataset`. Never hand-edit generated
manifests, hashes, paths, or provenance. During append, preserve every existing
image, label, and Workbench review entry byte-for-byte. Read
`references/data-contract.md` when debugging compatibility or deduplication.

After changing the generator, run:

```powershell
& PROJECT\.venv\Scripts\python.exe SKILL_DIR\scripts\self_test.py
```
