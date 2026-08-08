---
name: yoyo-string-annotation-vlm-assisted
description: Create, review, validate, and export high-quality yoyo string annotations from sampled video frames, with VLM-assisted triage and auditable quality gates. Use when building or maintaining yoyo string segmentation datasets from videos.
---

# VLM-Assisted Yoyo String Annotation

Run weak-VLM triage once, promote only bounded coarse fields, and give the
visual agent a sorted handoff containing all remaining work. Never let the weak
VLM create geometry or approve a label.

Write `agent_yoyo_string_annotation_v5` labels. Never add hand coordinates,
hand pose, body pose, `hands_pixel`, `hands_2d`, or hand path anchors. Path
anchors are limited to `yoyo` and `unknown`; preserve visible string geometry
through hand occlusions as split strokes with unresolved-gap metadata.

## Load References

- Read `references/data-contract.md` before initialization or export.
- Read `references/annotation-schema.md` before writing agent candidates or patches.
- Read `references/sampling-protocol.md` before sampling.
- Read `references/triage-contract.md` before running VLM triage or consuming its handoff.
- Read `references/review-protocol.md` before geometry refinement or approval.

Use `scripts/annotation_pipeline.py` for every label state change. Do not
hand-edit labels, digests, revision history, approvals, exports, or cleanup
reports.

## Upgrade V4 Labels

Upgrade a label tree in one pass. The command changes only fields whose v4
values are invalid in v5, removes deprecated non-task fields recursively, and
never creates migration history or other top-level annotation fields.

```powershell
& VENV_PYTHON "$SKILL_DIR\scripts\annotation_pipeline.py" upgrade-v5 `
  --labels DATASET\canonical\labels `
  --review-map PROJECT\workbench_state\dataset_review_status.json `
  --dataset-key DATASET_NAME --dry-run
```

Remove `--dry-run` only after every label and review-map key validates. The
write pass rebinds existing `yoyo_dataset_review_v3` entries to the migrated
file revisions without changing reviewer decisions.

## Interpreter And Configuration

Use a project virtual-environment interpreter that contains OpenCV, OpenAI,
PyYAML, python-dotenv, Pillow, and NumPy. Never use the global Python
interpreter. Confirm the interpreter and OpenCV before a run:

```powershell
& PROJECT\.venv\Scripts\python.exe -c "import sys,cv2; print(sys.executable); print(cv2.__version__)"
```

By default, every VLM command loads `.env` from this
skill folder, including when `--config` points elsewhere. Pass `--env-file` only
for an intentional override. Never write an API key into a label, triage result,
manifest, log, or command argument.

## Preserve Unicode Paths

Pass Chinese and other Unicode paths directly to every bundled script. Do not
rename media, change the working directory to avoid Unicode, or manually encode
paths.

Never call `cv2.imread` or `cv2.imwrite`. Use
`scripts/unicode_media.py` when a custom operation needs OpenCV arrays. The
bundled image workflow reads bytes through Python and decodes or encodes them
with OpenCV.

## Sample And Initialize

Sample anchors and nearby context without recognition:

```powershell
& VENV_PYTHON "$SKILL_DIR\scripts\sample_video_frames.py" `
  --videos INPUT_VIDEOS --output PROJECT `
  --frames-per-video 12 --oversample-factor 5 `
  --neighbor-offsets=-2,-1,1,2 --separate-context `
  --hash-cache DATASET_ROOT\source_video_sha256_cache.json
```

For an explicit source set, write one source path per line in a UTF-8 file and
use `--videos-list`. Use `--total-anchors` to allocate an exact source-balanced
anchor count.

Initialize only anchors:

```powershell
& VENV_PYTHON "$SKILL_DIR\scripts\annotation_pipeline.py" init `
  --images PROJECT\images --output PROJECT --min-approvals 2
```

## Run Weak-VLM Triage Once

Run triage before any visual-agent geometry is applied:

```powershell
& VENV_PYTHON "$SKILL_DIR\scripts\vlm_triage.py" run `
  --labels PROJECT\labels `
  --output PROJECT\triage `
  --config "$SKILL_DIR\config.yaml"
```

The script caches results by prompt version and model, validates the response,
discards prohibited fields, applies only safe high-confidence fields, and
writes:

- `triage/triage_manifest.json`: run status and queue counts
- `triage/results/<source_group>/*.json`: normalized evidence and promotions
- `triage/agent_handoff.json`: sorted work for the visual agent

Do not call the VLM again for a cached record. Use `--force` only when the
source, model, prompt contract, or triage result is known to be invalid.

The weak VLM may supply only:

- coarse domain and scene suggestions
- obvious frame-level yoyo presence and string evidence for routing
- obvious `motion_blur`, `low_contrast`, and `edge_clipped` tags
- task priority
- short factual notes and normalized JSON

It must never supply or promote string visibility, trick orientation,
yoyo bbox, coordinates, masks, centerlines, paths, topology, review decisions,
or approval status.

## Consume The Agent Handoff

Read `triage/agent_handoff.json` once and process records in its existing order.
Do not repeat coarse classification listed in `skip_decisions` unless original
pixels clearly contradict it.

For `quick_verify`, inspect the raw frame once. Confirm `reject` only when the
source is invalid or outside the yoyo domain; otherwise send it through full
visual annotation.

For `clear_candidate`, `standard`, and `hard_case`, perform the listed
`required_visual_agent_tasks` as one continuous visual pass:

1. Inspect raw pixels plus relevant temporal context.
2. Trace every defensible visible string section and split all unsupported gaps.
3. Set final string visibility, yoyo bbox, yoyo/unknown path anchors, ordered path, hidden gaps, and orientation.
4. Apply one complete candidate for the first geometry revision.
5. Render, inspect every edge, and use compact patches until approved or unresolved.

Keep promoted scene, bad-case, and notes fields unless direct evidence conflicts.
Correct them through the normal candidate or patch when needed; never edit the
label directly.

## Apply, Render, And Review

Use full `apply` for the first visual geometry and `apply-patch` for localized
refinement. Always declare `coordinate_frame` for resized images or crops.

```powershell
& VENV_PYTHON "$SKILL_DIR\scripts\annotation_pipeline.py" apply `
  --label LABEL.json --candidate CANDIDATE.json `
  --actor agent-annotator --role model-annotator --model AGENT_ID

& VENV_PYTHON "$SKILL_DIR\scripts\annotation_pipeline.py" render `
  --label LABEL.json --output REVIEW_DIR
```

Require independent `geometry-critic` and `semantic-critic` approvals on the
same digest. The weak VLM is not eligible for either role. Use
`request_changes` when a concrete correction remains and `unresolved` when no
defensible visible truth remains.

## Audit, Export, And Cleanup

Run strict audit and export only approved labels:

```powershell
& VENV_PYTHON "$SKILL_DIR\scripts\annotation_pipeline.py" audit `
  --labels PROJECT\labels --output PROJECT\audit.json --require-approved --strict

& VENV_PYTHON "$SKILL_DIR\scripts\annotation_pipeline.py" export `
  --labels PROJECT\labels --output REVIEWED_EXPORT
```

The export recreates one terminal review overlay per approved label under
`visualizations/<source_group>/`. Each image maps the reviewed geometry and
status onto the original video frame at full source resolution, and each
exported label links to it through `visualization`. Treat these overlays as
final dataset artifacts, not temporary review files.

Preview cleanup, then confirm it only after audit succeeds. Cleanup may remove
iterative grid/detail renders and temporary review directories, but it must
preserve `images/`, `labels/`, and `visualizations/` in a portable export.
Preserve the triage manifest, normalized results, and agent handoff as model
provenance evidence.
