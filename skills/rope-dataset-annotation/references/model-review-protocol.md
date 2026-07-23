# Model-First Review Protocol

Use this protocol for every positive label and every hard negative. Prefer model
self-review over casual human approval, but surface unresolved cases instead of
forcing a decision.

## Pass 1: annotator

Inspect the raw image and coordinate grid. For adjacent frames, inspect the
previous approved overlay and run temporal propagation before drawing from
scratch. Produce a complete candidate JSON, including visible strokes, anchors,
visibility, bad cases, and a whole-path reconstruction.

Prompt objective:

> Trace only rope pixels supported by this frame. Reuse the temporal seed as a
> hypothesis, move or remove every point that does not land on the current rope,
> split strokes at occlusions, and reconstruct the ordered route to the yoyo with
> per-edge evidence and confidence. Return unresolved gaps explicitly.

Apply the candidate with `rope_pipeline.py apply`, render the overlay and detail
views, then inspect them. Never approve directly from the JSON.

Use only `left_hand`, `right_hand`, `yoyo`, or `unknown` for endpoint anchors.
Assign anchors from endpoint proximity and leave distant endpoints unknown; do
not assign every stroke a generic hand-to-yoyo route.

## Pass 2: geometry critic

Inspect raw, grid, overlay, and detail images without relying on the annotator's
notes. Look for:

- centerline drift onto clothing, fingers, sticks, floor lines, or motion trails;
- missing visible branches or rope segments;
- a stroke incorrectly bridging an occlusion or crossing;
- a shortcut chord connecting the ends of an open V, U, arc, or partial loop;
- points that cut corners rather than following curvature;
- a yoyo box or hand anchor that pulls the route toward the wrong object;
- mask regions much wider than the visible rope.

Request changes with precise coordinates when any defect exists. Approve only
after the corrected overlay follows the rope at inspection resolution.

When both masks and centerlines exist, inspect their agreement edge by edge.
Centerlines should follow a mask-supported midline. Do not use a polygon's
farthest pair of vertices as a centerline because that can invent a closing edge
across an open formation. The strict audit rejects low centerline-to-mask pixel
support, but overlay/detail inspection remains required.

## Pass 3: semantic and temporal critic

Check visibility and topology independently. Compare adjacent frames when
available. A real rope path should move coherently; a background edge often
stays fixed or changes identity. Verify that:

- `visible`, `partial`, `not_visible`, or `uncertain` matches the evidence;
- the complete route reaches the declared yoyo/hand anchors when applicable;
- temporal edges are promoted to `observed` only if current-frame pixels confirm
  them;
- inferred edges remain outside visible training geometry;
- hard negatives were inspected at full frame and detail scale;
- attachment class is not inferred from geometry alone.

Use a different review role from the geometry pass. With two current digest-bound
approvals, the script marks the rope label approved.

Rerender after the final approval. The delivered render header and
`*_render.json` must reference the approved current digest rather than the
pre-review status.

## Temporal refinement loop

For consecutive frames:

1. Start from the nearest approved frame, not a stale unreviewed draft.
2. Run `propagate` to obtain forward/backward Lucas-Kanade point tracks.
3. Render the target overlay and detail view.
4. Move drifted points onto the current centerline, delete false segments, add
   newly visible segments, and split around new occlusions.
5. Update the whole path. Mark confirmed edges `observed`; leave unseen carried
   structure `temporal` or `inferred`.
6. Apply the corrected full candidate, which invalidates old approvals.
7. Run geometry and semantic/temporal reviews on the current digest.

If optical flow loses a point, do not connect the surviving points across the
gap. Treat the gap as unresolved until the current image supports a connection.

## Stop conditions

Export only when strict audit passes and required approvals target the current
content digest. Keep a frame pending or unresolved when any of these remain:

- two plausible ropes/cords cannot be distinguished;
- blur prevents a defensible centerline;
- a crossing topology cannot be resolved even with adjacent frames;
- the rope is outside the crop but its absence cannot be proven;
- the source image, hash, dimensions, or source group is inconsistent.
