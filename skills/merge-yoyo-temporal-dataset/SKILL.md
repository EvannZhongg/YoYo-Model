---
name: merge-yoyo-temporal-dataset
description: Merge confirmed groups from a skill-generated temporal dataset into the aggregate datasets/1Ayoyo_temporal while preserving canonical single-frame labels, hashes, and source-group lineage. Use when a temporal annotation batch has selected at least three usable frames per confirmed group and must be promoted into the aggregate temporal dataset.
---

# Merge Yoyo Temporal Dataset

Run the repository script with the project virtual environment:

```powershell
& .\.venv\Scripts\python.exe `
  skills\merge-yoyo-temporal-dataset\scripts\merge_temporal_dataset.py `
  --source datasets\1Ayoyo_temporal_zhou_bowen `
  --target datasets\1Ayoyo_temporal `
  --review-map workbench_state\dataset_review_status.json
```

The source must contain `manifest.json`, `sampling_manifest.json`,
`consecutive_groups.json`, `temporal_review.json`, and paired canonical image
and label files. Only groups whose temporal review entry is `status=confirmed`
are considered. Each confirmed group must retain at least three selected frame
keys; non-contiguous selections such as frames 1/3/5 are valid.

The target is created if absent, or appended transactionally if it already
exists. Existing target files are never overwritten. Provenance
`(source_video_sha256, frame_index)`, image SHA-256, group ID, and label path
collisions are hard errors. The target receives the selected image/label pairs
and an updated `manifest.json`, `sampling_manifest.json`, and
`consecutive_groups.json` whose groups contain only selected frames.

`temporal_review.json` and its group-level status are intentionally not copied
to the target. When a review map is supplied, existing single-frame review
bindings for promoted frames are copied under the aggregate dataset key; group
confirmation is a promotion gate, not annotation truth. Target labels retain only the canonical
`agent_yoyo_string_annotation_v5` fields and frame-level provenance.

The script writes a backup-free staging directory and publishes with exclusive
file creation. If any validation or copy fails, the target is left unchanged.
Do not hand-edit manifests or bypass the minimum-three-frame rule.
