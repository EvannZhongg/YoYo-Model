# Annotation Schema

## Schema Version

Use `agent_yoyo_string_annotation_v4`. The schema stores original-image pixel
geometry and generated 0-999 mirrors.

## Coordinate Frames

Candidate JSON may be authored from an original frame, resized render, crop, or
normalized grid. Declare `coordinate_frame` whenever coordinates are not direct
original pixels:

```json
{"type": "rendered_image", "render_size": [1280, 720]}
```

```json
{"type": "crop_resized", "origin": [500, 180], "crop_size": [300, 220], "render_size": [600, 440]}
```

The pipeline converts `*_pixel` fields into original pixels before validation
and writes. If no coordinate frame is supplied, the candidate is interpreted as
`original_pixel`.

## Complete Candidate

Every full apply supplies the complete visible state:

```json
{
  "coordinate_frame": "original_pixel",
  "visibility": "visible",
  "yoyo_bbox_pixel": [742, 430, 790, 480],
  "string_visibility": "partial",
  "string_polylines_pixel": [
    [[510, 190], [548, 250], [602, 318]],
    [[635, 340], [700, 397], [754, 440]]
  ],
  "string_mask_polygons_pixel": null,
  "hands_pixel": {"left": [510, 190], "right": null},
  "yoyo_division": "1A",
  "scene_label": "trick",
  "trick_orientation": "horizontal",
  "string_path": {
    "topology": "open",
    "reconstruction_status": "partial",
    "paths": [
      {
        "path_id": "left-hand-to-yoyo",
        "start_anchor": "left_hand",
        "end_anchor": "yoyo",
        "points_pixel": [[510,190], [548,250], [602,318], [635,340], [700,397], [754,440]],
        "edges": [
          {"from": 0, "to": 1, "evidence": "observed", "confidence": 0.96},
          {"from": 1, "to": 2, "evidence": "observed", "confidence": 0.94},
          {"from": 2, "to": 3, "evidence": "inferred", "confidence": 0.35},
          {"from": 3, "to": 4, "evidence": "observed", "confidence": 0.91},
          {"from": 4, "to": 5, "evidence": "observed", "confidence": 0.93}
        ]
      }
    ],
    "unresolved_gaps": ["string hidden behind hand between (602,318) and (635,340)"]
  },
  "bad_case": ["partial_occlusion"],
  "notes": "Visible strokes are split; hidden route is metadata only."
}
```

## Patch Candidate

Use patches for micro-adjustment. The stored label remains complete, but the
agent only writes changed fields or point operations:

```json
{
  "coordinate_frame": {"type": "rendered_image", "render_size": [1280, 720]},
  "stroke_ops": [
    {"op": "move_point", "stroke": 0, "point": 1, "value": [616, 280]},
    {"op": "delete_point", "stroke": 0, "point": 3}
  ],
  "set": {
    "string_visibility": "partial",
    "notes": "Adjusted one bend and removed a blur-trail point."
  }
}
```

The patch command can add, replace, delete, move, and insert points or strokes.
It rebuilds `string_path` from visible strokes by default; set
`rebuild_string_path=false` only when providing a precise replacement path.

## Visibility And Gaps

- `visible`: the current frame supports the important visible route.
- `partial`: visible pieces are defensible, but the complete route is not.
- `not_visible`: no string pixels can be defended; geometry must be empty.
- `uncertain`: no defensible positive or negative label.

Occluded, behind-neck, behind-hand, behind-body, and ambiguous crossing segments
must be split in visible geometry. Do not add a drawn segment to close a loop,
bridge a hand wrap, or maintain topology. Use path metadata for hidden order.

Before approval, visible geometry must cover the full defensible visible route.
Do not label only one obvious branch when another visible branch, vertical
return, loop side, hanging drop, or hand-to-yoyo section remains unlabeled.
Hidden gaps may split the route, but visible pieces on both sides of a gap must
be included when pixels support them.

For motion blur, keep one defensible physical centerline. Do not encode blur
trails as multiple parallel strings.

Use `yoyo_bbox_pixel` whenever the yoyo body is clearly visible and defensibly
bounded, including a localized motion-blurred disc. Leave it null only when the
yoyo is outside frame, fully hidden, or too blurred/ambiguous to bound. A
`string_path` endpoint with `start_anchor` or `end_anchor` equal to `yoyo`
requires a bbox and must land close to it; otherwise use `unknown`.

## Evidence And Orientation

Each path edge connects consecutive points and uses `observed`, `temporal`, or
`inferred`. `observed` requires full current-frame pixel support.

Use `trick_orientation` from nearby launch motion:

- `horizontal`: lateral launch direction is clear in a short temporal window.
- `normal`: ordinary downward or non-horizontal throw plane.
- `unknown`: draft or unresolved trick; cannot be approved.
- `not_applicable`: required for `scene_label=non_trick`.

Use `bad_case` for factual issues such as `motion_blur`, `partial_occlusion`,
`low_contrast`, `edge_clipped`, `ambiguous_string`, or `string_not_visible`.
