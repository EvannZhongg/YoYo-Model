---
name: rope-dataset-annotation
description: Create, refine, self-review, audit, and export high-quality rope/string image annotations, especially thin yoyo strings in still images or consecutive video frames. Use for rope centerlines, segmentation masks, ordered hand-to-yoyo path reconstruction, visibility negatives, temporal propagation from adjacent frames, coordinate-overlay review, multi-pass model QA, or generation of review-gated train/val/test labels.
---

# Rope Dataset Annotation

Build evidence-preserving rope labels with model-first iterative review. Keep all
code and protocol resources inside this skill; write only generated datasets and
review artifacts to the user-selected output directory.

## Load the contract

Read the references needed for the task before labeling:

- Read `references/annotation-schema.md` for every annotation task.
- Read `references/model-review-protocol.md` before reviewing, propagating, or
  approving labels.
- Read `references/repository-contract.md` when working in YoYo_model or exporting
  labels for its string segmentation trainers.

Use `scripts/rope_pipeline.py` for deterministic state changes. Do not hand-edit
approval metadata, normalized coordinate mirrors, content digests, or revision
history.

## Initialize labels

Create draft label files. Preserve source identity with `--source-group`; never
split frames from the same video/source across train, val, and test.

```bash
python "$SKILL_DIR/scripts/rope_pipeline.py" init \
  --images INPUT_IMAGES --output ANNOTATION_PROJECT \
  --split train --source-group VIDEO_ID --min-approvals 2
```

Use one project per coherent dataset. Run `init` separately for each source group
and assigned split. Let the script bind each label to the source image hash and
dimensions.

## Annotate with the model

For each draft:

1. Inspect the source image at full-frame scale.
2. Render a coordinate grid when exact points are difficult to estimate.
3. Trace every defensible visible rope segment as a separate centerline stroke.
4. Reconstruct the ordered full route in `string_path`, including the route to
   the yoyo when applicable. Mark every edge `observed`, `temporal`, or `inferred`.
5. Record visibility, yoyo box, hand anchors, topology, unresolved gaps, bad-case
   flags, and concise factual notes.
6. Write a complete candidate JSON and apply it as a new revision.

```bash
python "$SKILL_DIR/scripts/rope_pipeline.py" apply \
  --label LABEL.json --candidate CANDIDATE.json \
  --actor model-annotator --role model-annotator --model MODEL_ID \
  --message "initial current-frame trace"

python "$SKILL_DIR/scripts/rope_pipeline.py" render \
  --label LABEL.json --output REVIEW_DIR
```

Inspect the generated `*_grid.jpg`, `*_overlay.jpg`, and `*_detail.jpg`. Use a
local image-viewing tool, not JSON alone. Revise and rerender until the visible
centerline sits on rope pixels and path evidence is honest.

Observed `string_path` edges duplicate visible geometry and are intentionally not
drawn a second time. Review renders show visible centerlines in cyan, temporal
edges in orange, and inferred edges as dashed magenta. This keeps separate open
strokes from appearing like a closed formation.

Keep current-frame segmentation truth separate from reconstruction hypotheses:
put only reviewed visible geometry in `string_polylines_pixel` or
`string_mask_polygons_pixel`; retain hidden or merely plausible route sections in
`string_path` as `temporal` or `inferred` edges.

When a reviewed label has mask geometry but no centerline, derive a mask-supported
midline by pairing the polygon's two boundary arcs. Never join the farthest
polygon vertices directly: on
an open V, U, or curved formation that creates a shortcut chord across pixels
where no rope exists.

```bash
python "$SKILL_DIR/scripts/rope_pipeline.py" derive-centerlines \
  --label LABEL.json --actor model-mask-centerline-deriver --model MODEL_ID
```

Render and review the result. Strict audit compares centerline samples with mask
support and rejects unsupported shortcut edges when both representations exist.

## Refine consecutive frames

Start from the nearest approved frame when frames are consecutive. Propagate its
points with forward/backward optical flow, then make small current-frame edits.

```bash
python "$SKILL_DIR/scripts/rope_pipeline.py" propagate \
  --previous-label PREVIOUS.json --target-label CURRENT.json \
  --actor model-temporal-propagator --model MODEL_ID

python "$SKILL_DIR/scripts/rope_pipeline.py" render \
  --label CURRENT.json --output REVIEW_DIR
```

Treat propagation as a seed, never approval. Inspect the target raw image and
overlay. Move drifted points, remove false segments, add newly visible segments,
and split strokes at new occlusions. Promote a temporal edge to `observed` only
after current-frame confirmation. Reapply the corrected complete candidate so
the revision and review digest update.

If OpenCV is unavailable, skip propagation and use the previous overlay as visual
context. Do not silently assume unchanged coordinates.

## Run model self-review

Perform at least two digest-bound review passes with distinct objectives and
roles. The same model may perform both passes, but re-open the raw, overlay, grid,
and detail images and use separate reviewer identities.

First run a geometry critic. Request changes when the line drifts, bridges an
occlusion, misses a visible segment, cuts a bend, or includes background pixels.
Treat every consecutive edge independently. An apparent triangle or loop is not
evidence for its closing edge; approve only edges whose full length follows
current-frame rope pixels.

```bash
python "$SKILL_DIR/scripts/rope_pipeline.py" review \
  --label LABEL.json --decision approve \
  --reviewer model-geometry-pass --role geometry-critic \
  --model MODEL_ID --notes "Precise evidence-based findings"
```

Then run a semantic or temporal critic. Check visibility, negative status,
topology, anchors, inferred edges, and agreement with adjacent frames.

```bash
python "$SKILL_DIR/scripts/rope_pipeline.py" review \
  --label LABEL.json --decision approve \
  --reviewer model-semantic-pass --role semantic-critic \
  --model MODEL_ID --notes "Independent visibility and topology findings"
```

Use `request_changes`, `reject`, or `unresolved` when appropriate. Any candidate
edit changes the content digest and makes prior approvals stale. Repeat render,
critique, revision, and approval until the current digest passes.

After the final approval, render once more into the delivered review directory.
The `*_render.json` sidecar must contain the current content digest, revision,
and `approved` status; do not deliver a stale pending-status preview.

## Audit and export

Run strict audit over the whole collection. Resolve every error. Review warnings
as targeted quality prompts; a warning may remain only when it accurately
describes an explicit limitation such as a partial reconstruction.

```bash
python "$SKILL_DIR/scripts/rope_pipeline.py" audit \
  --labels ANNOTATION_PROJECT/labels \
  --output ANNOTATION_PROJECT/audit.json --strict
```

Use `--require-approved` when auditing a final batch that must contain no pending
or rejected items. Omit it for a working project that intentionally retains
`uncertain` frames. Export only approved, current, non-leaking labels:

```bash
python "$SKILL_DIR/scripts/rope_pipeline.py" export \
  --labels ANNOTATION_PROJECT/labels --output REVIEWED_EXPORT
```

Inspect `manifest.json`. Report exported and excluded counts by split and
visibility. Require reviewed positive rope examples and reviewed hard negatives
in every evaluation split before claiming the dataset is suitable for model
comparison. Keep `uncertain` frames in the annotation project for future review;
do not force them into either class.

## Verify the tooling

Run the bundled end-to-end test after changing the skill or its script:

```bash
python "$SKILL_DIR/scripts/self_test.py"
```

For a repeatable suite of at least ten real scenarios, provide a JSON object with
`cases: [{id, scenario, legacy_label}, ...]`, then run:

```bash
python "$SKILL_DIR/scripts/acceptance_suite.py" prepare \
  --cases CASES.json --output ACCEPTANCE_PROJECT

# Inspect raw_contact_sheet.jpg, overlay_contact_sheet.jpg, and detail renders.
# Use acceptance_suite.py confirm for visually verified seeds or defer for an
# unresolved case, then record decisions with rope_pipeline.py review.

python "$SKILL_DIR/scripts/acceptance_suite.py" report \
  --output ACCEPTANCE_PROJECT
```

Read `references/acceptance-results.md` for the checked YoYo_model 13-scenario
baseline and its limitations.
