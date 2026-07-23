# Rope Annotation Schema

## Coordinate conventions

- Use original-image pixel coordinates for all `*_pixel` fields.
- Use `[x,y]`, with origin at the upper-left, x increasing right, y increasing
  down.
- Keep points inside `0 <= x < width` and `0 <= y < height`.
- Mirror pixel coordinates to `*_2d` on a 0-999 scale. Let the pipeline generate
  these mirrors; do not hand-maintain both forms.

## Visibility

- `visible`: the important visible rope route can be traced with current-frame
  evidence.
- `partial`: one or more rope segments are visible, but occlusion, blur, crop, or
  contrast prevents tracing the entire visible/expected route.
- `not_visible`: no rope pixels can be defended after full-frame and detail
  inspection. Retain no visible geometry.
- `uncertain`: evidence cannot support a positive or negative label. Exclude it
  from training rather than guessing.

Do not use `not_visible` merely because the rope is difficult. Use `uncertain`
for unresolved frames.

## Visible geometry

`string_polylines_pixel` is a list of centerline strokes. Add a point at each
meaningful bend, crossing, or local curvature change. Do not densely click along
a straight segment. End a stroke at an occlusion or unresolved crossing and
start a new stroke when the rope becomes visible again.

Use `string_mask_polygons_pixel` only when the visible rope boundary is actually
discernible. A centerline is usually more stable for a one-to-three-pixel rope.
Never convert an uncertain centerline into a wide mask to create false certainty.

## Whole-path reconstruction

Store ordered route hypotheses separately:

```json
{
  "string_path": {
    "topology": "open",
    "reconstruction_status": "partial",
    "paths": [
      {
        "path_id": "right-hand-to-yoyo",
        "start_anchor": "right_hand",
        "end_anchor": "yoyo",
        "points_pixel": [[810, 210], [775, 328], [742, 451]],
        "edges": [
          {"from": 0, "to": 1, "evidence": "observed", "confidence": 0.97},
          {"from": 1, "to": 2, "evidence": "inferred", "confidence": 0.42}
        ]
      }
    ],
    "unresolved_gaps": ["rope is hidden by the yoyo rim near (742,451)"]
  }
}
```

Each edge must connect consecutive points and use one evidence value:

- `observed`: visually confirmed in the current frame.
- `temporal`: propagated from an adjacent frame but not yet confirmed in the
  current frame.
- `inferred`: plausible topology used to preserve a complete route hypothesis,
  with no direct current-frame proof.

Set reconstruction status to `complete` only when the ordered route to its
declared anchors is unambiguous. `partial` is a successful result when gaps are
explicit. Complex formations can contain multiple paths or use `loop`,
`branched`, `multiple`, or `uncertain` topology.

## Anchors and relations

Record a tight visible yoyo body box and hand points when defensible. These are
review aids and temporal anchors, not permission to draw a straight rope between
objects. Use `string_attachment_class=unknown` unless the action category and
image evidence support a stronger relation.

Allowed attachment values are `hand_and_yoyo_attached`, `yoyo_detached`,
`hand_detached`, and `unknown`. Never infer a detached style from endpoint
distance alone.

## Quality metadata

The pipeline binds reviews to a SHA-256 digest of label content. Any geometry or
semantic edit changes that digest and makes earlier approvals stale. At least two
review roles are required by default. A model may perform both passes, but use a
fresh inspection objective and a different role/reviewer identity for each pass.
