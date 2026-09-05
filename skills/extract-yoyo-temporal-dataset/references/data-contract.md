# Temporal extraction contract

The extractor intentionally mirrors the blank annotation contract used by
`create-yoyo-consecutive-blank-annotation-dataset`, while allowing multiple
groups per source video.

## Layout

```text
DATASET/
├─ canonical/images/<source_group>/<sequence_id>_frame_<index>-<hash>.jpg
├─ canonical/labels/<source_group>/<same-name>.json
├─ manifest.json
├─ sampling_manifest.json
├─ consecutive_groups.json
└─ temporal_review.json
```

`manifest.json` uses `yoyo_consecutive_annotation_dataset_v1`. Its `records`
array is one-to-one with image/label files and includes `group_id`, source
video SHA-256, frame index, and image SHA-256. `sampling_manifest.json` uses
`agent_video_sampling_v1` and records the command parameters and source
metadata. No recognition model is used.

Unless overridden, provenance checks read both `datasets/1Ayoyo_dataset` and
`datasets/1Ayoyo_consecutive`; the extractor also understands the unified
dataset's older manifests whose source hash is stored in the referenced label.

`consecutive_groups.json` uses `yoyo_consecutive_groups_v1`. Each group owns
an ordered `frames` array. A frame's `sample_key` is relative to
`canonical/labels`; `image` is relative to the dataset root. The inclusive
`selected_start_frame` and `selected_end_frame` must cover exactly the frame
indices listed in `frames`.

`temporal_review.json` uses `yoyo_temporal_review_v1` and is mutable Workbench
state. Each group stores `status`, `selected_sample_keys`, and matching
`selected_frame_indices`; confirmation requires at least three unique selected
frames, while gaps are allowed. Group status is a promotion gate and is not
copied into the aggregate dataset.

## Blank label state

Labels are `agent_yoyo_string_annotation_v5` with provenance, image size/hash,
`sequence_id`, and `group_id` filled in. Geometry is empty; `active_yoyo` is
initialized as visible with a review-needed null box, `string_visibility` is
`partial`, `string_review_status` is `unresolved`, and `backup_yoyos` is empty.
These values are queue defaults, not training truth, and must be reviewed
before training.

## Temporal loader guidance

Use `consecutive_groups.json` rather than sorting filenames. A loader should
read each group's frames in listed order, preserve the source frame indices,
and treat a group as one sample unit when splitting train/validation/test.
Never split adjacent members of one group across partitions. For a model that
learns whether to trust current visual evidence or history, expose the frame
offset (0…`frames_per_group`−1) and optionally the preceding predicted mask or
centerline as an input feature; do not convert LK output into ground truth
without an independently reviewed target.
