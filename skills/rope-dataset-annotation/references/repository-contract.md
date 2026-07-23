# YoYo Repository Contract

Use this reference when the target is the YoYo_model repository. The skill's
scripts are independent, but exported labels intentionally match this contract.

## Findings from the repository

The repository treats the rope as a thin segmentation target, not a bounding-box
target. `string_segmentation/prepare_dataset.py` accepts only labels whose
`string_review_status` is `approved` or `reviewed` and whose split is `train`,
`val`, or `test`.

For `string_visibility=visible|partial`, training uses reviewed
`string_mask_polygons_pixel` when present. Otherwise it buffers each reviewed
`string_polylines_pixel` centerline into a thin polygon. For
`string_visibility=not_visible`, it creates an empty segmentation label as an
explicit negative. It excludes `uncertain`, rejected, and pending labels.

`source_group` must never appear in more than one split. This is a video/source
identity boundary, not merely a directory name. `image_size` is `[width,height]`.
Pixel geometry is authoritative; `*_2d` is the 0-999 normalized mirror.

The current review gate requires visible/partial rope to have at least one valid
stroke or mask, and requires `not_visible` to contain no rope geometry. It does
not independently check image hashes, stale approvals, revision history,
coordinate bounds, duplicate points, split duplicate hashes, or reviewer count.
This skill adds those checks.

## Observed dataset state (2026-07-23)

The inspected `datasets/video_v1/annotations/labels` contained 249 labels:

- 79 `reviewed`, 169 `auto_labeled_needs_review`, and 1 `rejected` rope labels.
- 205 `visible`, 11 `partial`, 29 `not_visible`, and 4 `uncertain` labels.
- The 79 accepted labels comprised 50 visible/partial positives and 29 explicit
  negatives.
- Accepted geometry was heterogeneous: 22 centerline-only, 24 mask-only, 4 with
  both, and 29 negatives with neither.
- Attachment metadata was sparse: 68 accepted labels were `unknown`, 3 were
  `hand_and_yoyo_attached`, and 8 omitted the field.

Do not treat those counts as a quality guarantee. They explain why model review,
provenance, temporal reuse, and stronger gates are needed.

## Compatible top-level fields

Preserve these fields in exports:

| Field | Meaning |
| --- | --- |
| `source_image` | Absolute or resolvable source image path |
| `image_size` | `[width,height]` in pixels |
| `source_group` | Source/video identity used for split isolation |
| `video_id` | Usually the same source identity |
| `split` | `train`, `val`, or `test` |
| `string_visibility` | `visible`, `partial`, `not_visible`, or `uncertain` |
| `string_polylines_pixel` | Separate visible centerline strokes |
| `string_mask_polygons_pixel` | Optional reviewed visible rope masks |
| `string_attachment_class` | Optional relation metadata, not a segmentation gate |
| `string_review_status` | Only `approved`/`reviewed` enters training |
| `yoyo_bbox_pixel` | Visible yoyo body box, useful as an anchor |
| `hands_pixel` | Left/right hand point anchors |
| `bad_case` | Auditable difficulty flags |

The additional `image_sha256`, `quality`, and `string_path` fields are ignored by
the current trainer and remain safe compatibility metadata.

## Important semantic boundary

Keep `string_polylines_pixel` and reviewed masks limited to pixels supported by
the current frame. Put a whole-route reconstruction in `string_path`. Every path
edge declares `observed`, `temporal`, or `inferred`; temporal and inferred edges
are context for refinement and tracking, not segmentation truth.
