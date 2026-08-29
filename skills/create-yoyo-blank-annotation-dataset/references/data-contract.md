# Blank Dataset Contract

## Layout

Workbench discovers a dataset below `datasets/` when it contains paired
`canonical/images` and `canonical/labels` trees. Each label filename mirrors its
image filename and uses the `.json` suffix.

New datasets use `manifest.json` schema `yoyo_blank_annotation_dataset_v1`.
Incremental append is allowed only when that schema is present and
`dataset_id` matches the dataset directory name. Never append to
`datasets/1Ayoyo_dataset`.

## Labels

Each label uses `agent_yoyo_string_annotation_v5` and records source-video
hash, source group, frame index, timestamp, sequence, image hash/size, and the
sampling-manifest hash.

Geometry is empty: boxes, polylines, masks, and paths must contain no data.
Initialize the manual queue with `visibility=uncertain`,
`yoyo_not_visible_reason=null`,
`trick_orientation=normal`, `string_visibility=partial`, and
`string_review_status=unresolved`. Quality history and reviews start empty;
these defaults are not reviewed training truth.

## Deduplication

Reject candidates that overlap a reference dataset or another candidate in the
same batch by:

1. source-video hash plus frame index, optionally expanded by
   `--exclude-frame-window`;
2. exact encoded-image SHA-256; or
3. perceptual hash at or below `--perceptual-hamming-threshold`.

The reference set always includes `datasets/1Ayoyo_dataset`, the target during
append, and earlier datasets created by this skill. Add more with repeated
`--exclude-dataset`.

## Provenance And Append Safety

`sampling_manifest.json` uses `agent_video_sampling_v1` and records that no
recognition model was used. `manifest.json` records inputs, reference datasets,
deduplication settings, selected samples, and rejection counts. Keep the
initial sampling manifest unchanged; append runs are recorded under
`provenance/` and listed in the dataset manifest.

During append, all existing images, labels, manifests, and Workbench review
entries are protected. The operation may create only new sample files and the
updated manifest. If it fails, the generator must leave the prior dataset and
review map unchanged.
