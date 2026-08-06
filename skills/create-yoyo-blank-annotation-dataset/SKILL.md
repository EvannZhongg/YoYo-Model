---
name: create-yoyo-blank-annotation-dataset
description: Create or incrementally extend deduplicated blank agent_yoyo_string_annotation_v4 datasets from yoyo video frames for the Workbench data-annotation page while preserving every existing label edit and SHA-bound human verification record. Use when preparing or supplementing a manual yoyo/string annotation queue without VLM classification, agent geometry, model fine-tuning, or model-review approval.
---

# Create Yoyo Blank Annotation Dataset

Create Workbench inputs with empty geometry and the manual queue defaults
`visibility=visible`, `trick_orientation=normal`, and
`string_visibility=partial`. Do not invoke a VLM, recognition model, visual
agent, reviewer agent, fine-tuning job, or approval pipeline.

## Run The Generator

Use the repository virtual environment. The script discovers the repository
from its installed location and writes `datasets/<dataset-name>` directly.

```powershell
& PROJECT\.venv\Scripts\python.exe `
  SKILL_DIR\scripts\create_blank_dataset.py `
  --videos VIDEO_OR_DIRECTORY `
  --dataset-name NEW_DATASET_NAME `
  --frames-per-video 12
```

For a controlled source list, write one video path per line in a UTF-8 file and
use `--videos-list`. Use `--total-frames N` to allocate an exact, nearly equal
count across sources. The total must be at least the number of videos.

To add more blank samples to a dataset previously created by this skill, keep
the Workbench server stopped and pass `--append` with the same dataset name:

```powershell
& PROJECT\.venv\Scripts\python.exe `
  SKILL_DIR\scripts\create_blank_dataset.py `
  --videos VIDEO_OR_DIRECTORY `
  --dataset-name EXISTING_DATASET_NAME `
  --frames-per-video 12 --append
```

Treat a missing `--append`, stale human-review SHA, missing old sample, path
collision, or changed protected file as a hard failure. Do not bypass these
checks and do not append directly to `datasets/1Ayoyo_dataset`.

The default exclusion baseline is `datasets/1Ayoyo_dataset`. The generator also
excludes every sibling dataset whose manifest identifies it as an earlier
output of this skill. Add other baselines with repeatable `--exclude-dataset`.
Never disable the root baseline check.

## Fast Decoding

Candidate decoding is progressive: it first tries two temporally balanced
candidates per requested stratum and stops as soon as a complete diverse set
passes deduplication. `--oversample-factor` remains the hard fallback budget,
not a number of frames that must always be decoded.

The default decoded-frame cache is
`annotations/source_frame_jpeg_cache`. Entries are content-addressed by source
video SHA-256, frame index, and JPEG quality, so retries and later generation
runs can reuse decoded frames without seeking through the video again. Use
`--frame-cache PATH` to relocate it or `--no-frame-cache` for a one-off run.
The cache is disposable and is never a dataset or annotation source.

Two videos are decoded concurrently by default. Adjust with
`--decode-workers N`; reduce it to `1` on memory-constrained machines. Results
are still committed in input order and cross-video deduplication is applied
before publication.

For stricter separation around frames already present from the same source
video, pass `--exclude-frame-window N`. The default rejects the identical frame;
`N=2` also rejects two neighboring frames on either side.

## Verify The Result

Require the command to finish with `"ok": true`. Confirm the result contains:

```text
datasets/<dataset-name>/
|-- canonical/images/<source-group>/*.jpg
|-- canonical/labels/<source-group>/*.json
|-- manifest.json
`-- sampling_manifest.json
```

Open the Workbench data-annotation page and select the new dataset. It is
discoverable without a separate import or copy step.

Do not hand-edit generated provenance, hashes, paths, or the manifest. Regenerate
a failed batch under a new dataset name. Read `references/data-contract.md` when
debugging compatibility, deduplication, or provenance.

## Protect Existing Work

Before an append, validate every existing image and label against the manifest,
validate each Workbench review entry against the current label SHA-256, and
snapshot all protected hashes in memory. Publish new image/label files with
exclusive creation and update only `manifest.json` after all new pairs exist.

Never overwrite, rewrite, rename, delete, or normalize an existing image or
label. Never write the Workbench review map. On any append failure, remove only
files created by that append and restore the previous manifest bytes. Require
the result to report `review_map_unchanged=true` and the expected
`review_entry_count_preserved`.

## Validation

Run the bundled deterministic test after changing the generator:

```powershell
& PROJECT\.venv\Scripts\python.exe SKILL_DIR\scripts\self_test.py
```
