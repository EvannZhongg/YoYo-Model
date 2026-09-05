---
name: extract-yoyo-temporal-dataset
description: Extract fixed-size consecutive video-frame groups into a Workbench-compatible blank yoyo/string dataset with explicit temporal group metadata. Use when preparing adjacent-frame data for temporal string recognition/tracking; this skill only samples and initializes blank annotations, and does not run recognition, labeling, training, or evaluation.
---

# Extract Yoyo Temporal Dataset

Run the bundled script from the repository root with the project virtual
environment. The output must be a new directory (normally under `datasets/`)
and is never written into `datasets/1Ayoyo_dataset`.

For one source video, request an exact number of groups. Each group contains
`--frames-per-group` adjacent frames (default `5`):

```powershell
& .\.venv\Scripts\python.exe `
  skills\extract-yoyo-temporal-dataset\scripts\extract_temporal_groups.py `
  --videos "videos\2018cyml1Apre\2018 CYML 1A Pr 2nd 周博文 Zhou Bowen.mp4" `
  --output datasets\1Ayoyo_temporal_zhou_bowen `
  --groups-per-video 5 `
  --frames-per-group 5
```

For a directory, `--groups N` means *N groups total*; the script allocates
them round-robin over sorted videos (one group at a time until the requested
total is reached). Use `--groups-per-video N` when every source should receive
the same count:

```powershell
& .\.venv\Scripts\python.exe `
  skills\extract-yoyo-temporal-dataset\scripts\extract_temporal_groups.py `
  --videos videos\2018cyml1Apre `
  --output datasets\1Ayoyo_temporal_2018cyml1Apre `
  --groups 40 `
  --frames-per-group 5
```

The sampler chooses deterministic, evenly spaced, non-overlapping windows.
By default it excludes frame provenance already present in both
`datasets/1Ayoyo_dataset/manifest.json` and
`datasets/1Ayoyo_consecutive/manifest.json`; repeat `--reference-dataset` to
replace that set, or pass `--reference-dataset ""` to disable the check. It
fails rather than silently shifting a request when there are not enough valid
windows.

Every output contains `canonical/images`, `canonical/labels`,
`manifest.json`, `sampling_manifest.json`, `consecutive_groups.json`, and a
`temporal_review.json` sidecar. Groups start unresolved with all sampled frames
selected; the Workbench can retain a non-contiguous subset and requires at
least three selected frames before confirmation.
Labels use `agent_yoyo_string_annotation_v5` blank-state defaults. The group
manifest is the authoritative ordered clip index: each group has a stable
`group_id`, source hash, sequence ID, inclusive frame range, and one entry per
frame. Read [references/data-contract.md](references/data-contract.md) when
connecting a temporal loader or Workbench review flow.

After extraction, require the command to print `"ok": true`, check the group
and sample counts, and verify that every group's frame indices are consecutive.
Do not hand-edit hashes, paths, or manifests.
