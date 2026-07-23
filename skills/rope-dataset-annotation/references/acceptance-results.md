# Acceptance Results

## Scope

Executed on 2026-07-23 against 13 real frames from 10 source groups in the
YoYo_model video dataset. The suite covered train, val, and test sources. Legacy
geometry was imported only as a low-confidence temporal seed. The model then
inspected raw contact sheets, overlays, and original-coordinate detail views.

Every accepted case received two current-content-digest approvals with distinct
geometry and semantic review roles. Unresolved cases were removed from current
training geometry while their former candidate and inferred path remained in
revision history.

## Results

| ID | Scenario | Result |
| --- | --- | --- |
| 01 | Clear visible rope with yoyo and hands | Accepted |
| 02 | Complex three-stroke formation with occlusion gaps | Accepted |
| 03 | Motion-blurred loop with omitted branches | Safely unresolved |
| 04 | Long partial rope with hidden sections | Accepted as partial |
| 05 | Yoyo/rope clipped at the image edge | Accepted as partial |
| 06 | Multiple branches with uncertain yoyo anchor | Safely unresolved |
| 07 | Motion blur with overlapping yoyo-like candidates | Safely unresolved |
| 08 | Hands occluded with ambiguous topology | Accepted as partial |
| 09 | Yoyo visible but rope not defensibly visible | Accepted hard negative |
| 10 | Non-trick frame with no yoyo or rope | Accepted negative |
| 11 | Dense mask components and background distractors | Accepted after removing noisy mask |
| 12 | Clear temporal-pair previous frame | Accepted |
| 13 | Temporal current frame separated by a large sampling gap | Safely unresolved |

Summary:

- 9/13 (69.2%) became fully approved training labels.
- 4/13 (30.8%) were explicitly unresolved instead of receiving invented paths.
- 13/13 reached a safe terminal classification; none remained pending or failed.
- Final collection audit: 0 errors, 0 warnings, and 0 split-leakage findings.
- Approved set: 7 positive/partial rope labels and 2 reviewed negatives.

## Temporal finding

The real sampled temporal pair was about two seconds apart. Forward/backward
Lucas-Kanade flow rejected all 12 attempted measurements (`tracked_fraction=0`).
The pipeline correctly refused to connect surviving points, but acceptance
testing exposed two fallback bugs that were fixed:

1. Zero-track propagation now changes positive visibility to `uncertain` and
   adds `temporal_propagation_failed`.
2. Sparse point flow never copies a previous-frame mask, and clearing geometry
   clears both modern multi-stroke fields and legacy single-stroke mirrors.

Short-gap synthetic temporal testing still passes: translated centerline/path
points propagate as `temporal`, remain unapproved, and require current-frame
model confirmation.

## Interpretation

This is a workflow acceptance test, not a pixel-accuracy benchmark against a new
independent ground-truth set. It demonstrates that the skill handles a majority
of representative video scenes as approved labels and safely contains the hard
remainder. Measure production accuracy with a larger double-annotated holdout and
centerline/mask agreement metrics before claiming a numerical accuracy level.
