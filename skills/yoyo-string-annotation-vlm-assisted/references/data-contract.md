# Annotation Data Contract

## Consumer Requirements

The target is thin binary yoyo string segmentation plus yoyo geometry. Pixel
geometry is authoritative and must be stored in original image coordinates. Generated
`*_2d` mirrors use a 0-999 scale and are not the source of truth.

Required top-level fields:

| Field | Contract |
| --- | --- |
| `source_image` | Image path in the annotation project or portable export |
| `image_sha256` | Identity check for stale or duplicated source pixels |
| `image_size` | Original `[width,height]` |
| `source_video` | Original video path supplied by the sampling script |
| `source_video_sha256` | Full SHA-256 identity of the original video |
| `source_group` | Stable video/source identity for later partitioning |
| `video_id` | Exact mirror of `source_group` for host-dataset compatibility |
| `frame_index` | Zero-based frame index in `source_video` |
| `timestamp_s` | Frame timestamp derived from source FPS |
| `sequence_id` | Anchor-centered sampling sequence identity |
| `sampling_role` | `anchor` or `temporal_context` |
| `anchor_frame_index` | Anchor frame for this sampling sequence |
| `sampling_manifest_sha256` | Identity of the manifest that supplied provenance |
| `trick_orientation` | `normal`, `horizontal`, `unknown`, or `not_applicable` |
| `string_visibility` | `visible`, `partial`, `not_visible`, or `uncertain` |
| `string_polylines_pixel` | Separate visible centerline strokes only |
| `string_mask_polygons_pixel` | Optional visible rope-area polygons |
| `string_review_status` | Only `approved` or `reviewed` may enter export |

The schema has no `split` field. Downstream partitioning must keep whole
`source_group` values together.

Dataset identity, split, source-dataset, and canonical output paths belong in
the dataset manifest. Never add a `dataset_management` object to a label.

The schema also has no hand or body-pose fields. Do not store `hands_pixel`,
`hands_2d`, hand landmarks, pose landmarks, or hand attachment anchors. A
visible route ending at a hand occlusion uses an `unknown` anchor and records
the unsupported continuation in `string_path.unresolved_gaps`.

## Visible Geometry

`string_polylines_pixel` contains only visible current-frame pixels. Any
occlusion, behind-neck route, behind-hand wrap, behind-body pass, or ambiguous
crossing must create a break. A hidden connection may appear only in
`string_path` as `inferred` metadata or an unresolved gap; it must never be
rasterized as segmentation truth.

Approved positive geometry must cover every defensible visible section of the
route, not just the easiest span. Missing visible string sections, returns,
loop sides, hanging drops, or other branches are label defects even
when the annotated segment is well aligned.

For motion blur, keep one defensible physical centerline. Do not encode blur
trails as multiple parallel strings.

`yoyo_bbox_pixel` is required when the yoyo body is clearly visible and
defensibly bounded. Yoyo path anchors must be close to that bbox. If the yoyo
is outside frame, hidden, or too ambiguous to bound, the bbox remains null and
path anchors should use `unknown` rather than a guessed yoyo endpoint.

## Provenance Authority

`sample_video_frames.py` is the sole authority for video provenance. `init`
must match each image to exactly one `agent_video_sampling_v1` manifest record
by path or image SHA-256 and source group, then copy provenance. Audit rejects
missing fields, invalid hashes, source identity conflicts, or image-size drift.

## Portable Export

The exporter writes original frames under `images/<source_group>/`, labels under
`labels/<source_group>/`, terminal review overlays under
`visualizations/<source_group>/`, and `manifest.json`. Each overlay maps the
terminal reviewed geometry and status onto its source video frame at the full
source resolution by default. Exported labels keep relative `source_image` and
`visualization` paths, while `source_image_original` preserves project
provenance.

Final cleanup must preserve source images, labels, terminal review
visualizations, manifest, audit, and source provenance. Remove intermediate
grid/detail renders, candidate snippets, patch snippets, temporary tests,
self-test outputs, and acceptance outputs before handoff.
