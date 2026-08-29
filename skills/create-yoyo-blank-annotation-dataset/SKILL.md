---
name: create-yoyo-blank-annotation-dataset
description: Create or append deduplicated blank agent_yoyo_string_annotation_v5 samples from yoyo videos for the Workbench manual annotation queue. Use when preparing new annotation data without model-generated geometry or model-review decisions.
---

# Create Yoyo Blank Annotation Dataset

Create samples under `datasets/<dataset-name>` with empty geometry and these
manual-queue defaults:

- `active_yoyo.visibility=uncertain` and `active_yoyo.not_visible_reason=null`
- `active_yoyo.trick_orientation=normal`
- `backup_yoyos=[]`
- `string_visibility=partial`

Use the repository virtual environment and the bundled generator. Do not call
VLM, detection, recognition, fine-tuning, or approval workflows; this skill only
creates blank annotation inputs.

## Create a Dataset

```powershell
& PROJECT\.venv\Scripts\python.exe `
  SKILL_DIR\scripts\create_blank_dataset.py `
  --videos VIDEO_OR_DIRECTORY `
  --dataset-name DATASET_NAME `
  --frames-per-video 12
```

Use `--videos-list` for a UTF-8 file containing one video path per line. Use
`--total-frames N` instead of `--frames-per-video` when an exact, source-balanced
total is required. Repeat `--exclude-dataset PATH` to add reference datasets;
use `--exclude-frame-window N` when neighboring frames from the same source
must also be excluded.

## Append Samples

Append only to a dataset previously created by this skill, and stop the
Workbench server for the operation:

```powershell
& PROJECT\.venv\Scripts\python.exe `
  SKILL_DIR\scripts\create_blank_dataset.py `
  --videos VIDEO_OR_DIRECTORY `
  --dataset-name EXISTING_DATASET_NAME `
  --frames-per-video 12 --append
```

Never append to `datasets/1Ayoyo_dataset`. Never overwrite, rename, delete, or
manually normalize an existing image, label, or Workbench review entry. Let the
generator update the dataset manifest as part of the append operation. It must
preserve existing review data and report
`review_map_unchanged=true` plus the preserved review-entry count.

## Verify

Require the command result to contain `ok: true` and confirm the output has:

```text
datasets/<dataset-name>/
|-- canonical/images/<source-group>/*.jpg
|-- canonical/labels/<source-group>/*.json
|-- manifest.json
`-- sampling_manifest.json
```

The labels must use `agent_yoyo_string_annotation_v5`, contain no boxes,
polylines, masks, or paths, and retain source/frame provenance. Select the new
dataset directly on the Workbench data-annotation page; no import or copy step
is needed.

Do not hand-edit generated provenance, hashes, paths, labels, or manifests.
Read [data-contract.md](references/data-contract.md) when checking schema,
deduplication, or append compatibility.
