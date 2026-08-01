# Iterative Agent Review Protocol

## Direct Annotation Pass

Inspect the original pixels and rendered coordinate aids. If using a resized
view or crop, include `coordinate_frame` in the candidate or patch. Never infer
pixel coordinates from memory of a preview size.

Trace all defensible visible strokes and build an ordered `string_path`. Hidden
segments can be represented only by `inferred` path edges and
`unresolved_gaps`; do not draw them into `string_polylines_pixel`.

Before applying or approving, trace the full visible route from hands through
mounts, returns, loop sides, hanging drops, and the visible yoyo when present.
Reject half-route labels where one obvious visible branch is missing even if
the labeled branch is accurate.

## Fine-Adjustment Loop

Repeat until approved or unresolved:

1. Compare raw, grid, overlay, detail, and render metadata.
2. Check the coordinate frame if overlay drift is large on the original frame.
3. Inspect every consecutive edge, not only each control point.
4. Move drifted points onto the string centerline.
5. Add points where a straight segment cuts a bend.
6. Delete background, motion-trail, clothing, finger, floor, or blur-ghost edges.
7. Split strokes at unsupported gaps, hand/body/neck occlusions, wraps behind a hand, and ambiguous crossings.
8. Add every other visible route segment supported by pixels.
9. Check `yoyo_bbox_pixel` when the yoyo is visible and every yoyo path anchor against that bbox.
10. Update anchors, visibility, path ordering, gaps, `bad_case`, and notes.
11. Verify `trick_orientation` from launch motion in nearby frames.
12. Apply a compact patch for small edits, or a full candidate for rewrites.
13. Render again.

There is no fixed revision limit. Each edit changes the content digest and
invalidates prior approvals.

## Motion Blur And Duplicate Lines

A blur trail is not multiple strings. Annotate one centerline only when the
physical string location is defensible. If several ghosted lines could be the
string and no single centerline can be justified, mark the record unresolved.

## Independent Critics

The geometry critic checks pixel alignment, curvature, gaps, masks, shortcut
edges, coordinate-frame correctness, and hidden-section leakage into visible
geometry. The semantic critic checks visibility, negative status, attachment
claims, topology, ordered route, `bad_case`, `scene_label`,
`trick_orientation`, and neighbor coherence.

The geometry critic must explicitly reject:

- half-route labels with another visible rope branch left unlabeled
- offset labels consistently displaced from the rope centerline
- shortcut labels whose long edges cross background, body, or hand pixels

The semantic critic must explicitly check that visible yoyo bodies have
defensible bboxes, invisible or ambiguous yoyos do not receive guessed bboxes,
and yoyo path anchors agree with the bbox.

Two distinct roles must approve the same content digest:
`geometry-critic` and `semantic-critic`.

## Stop And Abandon

Use `request_changes` when a concrete correction remains possible. Use
`unresolved` when blur, occlusion, behind-neck/behind-hand routing, multiple
plausible strings, an indeterminate crossing, or repeated failed adjustment
prevents defensible truth. Use `reject` for invalid source data or frames
outside the domain.

Unresolved and rejected records remain auditable but are excluded from export.
Never relabel an ambiguous case as `not_visible` just to reach a terminal class.

## Final Cleanup Review

Before delivery, run cleanup in dry-run mode and inspect candidates. Then run
confirmed cleanup only after strict audit passes. Final directories should not
contain intermediate grid/detail review renders, ad hoc candidate JSON, patch
snippets, self-test output, acceptance output, temp files, or caches unrelated
to final source frames, labels, manifests, and audit evidence. Preserve the
terminal review overlays in `visualizations/`; they are label-linked dataset
artifacts that map final annotation data onto the original video frames.
