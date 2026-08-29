# Weak-VLM Triage Contract

## Authority Boundary

Weak-VLM output is routing evidence, not annotation truth. Only the
deterministic triage script may promote fields into a draft label. A visual
agent remains authoritative for current-frame geometry and terminal review.

Allowed observations:

- `domain_status`: `in_domain`, `out_of_domain`, `invalid_source`, or `uncertain`
- `scene_label`: `trick`, `non_trick`, or `unknown`
- `scene_is_obvious`: boolean
- `obvious_yoyo_presence`: `present`, `absent`, or `uncertain`
- `coarse_string_evidence`: `obvious`, `possible`, `none_obvious`, or `uncertain`
- `frame_usability`: `usable`, `severely_degraded`, or `uncertain`
- `priority_suggestion`: `quick_verify`, `clear_candidate`, `standard`,
  `hard_case`, or `uncertain`
- `obvious_bad_cases`: `motion_blur`, `low_contrast`, `edge_clipped`, or `severe_occlusion`
- scalar confidence values in `[0,1]`

Prohibited output includes coordinates, boxes, points, masks, polygons,
polylines, centerlines, final string visibility, trick orientation,
attachment, path ordering, topology, gaps, review decisions, and approvals.
The script discards prohibited keys and records a warning.

`coarse_string_evidence=none_obvious` never means `string_visibility=not_visible`.
It only indicates that the weak model cannot see an obvious string at its input
resolution.

## Promotion Rules

The deterministic script may promote:

- `scene_label` when domain and scene confidence both meet
  `promotion_confidence` and `scene_is_obvious=true`
- `motion_blur`, `low_contrast`, and `edge_clipped` when bad-case confidence
  meets `promotion_confidence`

It never promotes `severe_occlusion`; that value only routes the frame to the
hard-case queue. It never promotes any geometry or terminal status.

Promotion is allowed only before visual geometry or reviews exist. Applying a
promotion creates a normal audited revision through `annotation_pipeline.py`.
Repeated processing of the same result is idempotent.

## Queue Semantics

`quick_verify`:

- High-confidence invalid or out-of-domain candidate.
- Visual agent inspects the raw frame once and confirms reject or returns the
  record to full annotation.

`clear_candidate`:

- Obvious usable in-domain frame with a clearly present yoyo.
- Process early as likely productive annotation work.

`standard`:

- No obvious terminal routing signal.
- Perform the full direct visual pass.

`hard_case`:

- Severe degradation, obvious severe occlusion, or low overall confidence.
- Allocate detailed raw/crop/context inspection and allow `unresolved` when no
  defensible truth remains.

The VLM supplies a priority suggestion. The deterministic script accepts it
only when confidence and compatible domain/usability evidence satisfy the
queue gate. The handoff order is `quick_verify`, `clear_candidate`, `standard`,
then `hard_case`. Each record includes `skip_decisions`, remaining visual tasks,
and an override rule. The visual agent must not repeat skipped coarse decisions
without contradictory pixel evidence.

## Evidence Files

Keep these files through final audit and handoff:

- `triage/triage_manifest.json`
- `triage/results/<source_group>/*.json`
- `triage/agent_handoff.json`

They record the prompt version, model, normalized assessment, discarded-field
warnings, applied promotions, queue assignment, and required visual work. They
contain no API key.
