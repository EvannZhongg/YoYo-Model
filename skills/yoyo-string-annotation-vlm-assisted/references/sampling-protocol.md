# Dispersed Video Sampling Protocol

## Purpose

Build annotation candidates that span sources, time, scene appearance, and
adjacent motion without using learned recognition. Sampling does not identify
the yoyo or string; direct agent inspection creates labels.

## Deterministic Selection

`sample_video_frames.py`:

1. Hashes each source video and preserves it as a stable source group. It may
   reuse a full digest only when the cache entry also matches file metadata and
   a dispersed byte fingerprint.
2. Generates candidate indices across nearly the full duration of every video,
   then uses OpenCV random seeks to decode only those candidates and requested
   context frames. The imageio fallback decodes the full video and is not
   suitable for large production sources.
3. Selects anchors from temporal strata using low-resolution color, intensity,
   and edge appearance distance only.
4. Extracts configured neighbor offsets for continuity review.

Use `--total-anchors N` when the final collection has an exact size. The
sampler allocates anchors evenly across all discovered sources, with at most
one anchor of difference between sources. Use a shared `--hash-cache` across
runs and invoke the script through the repository virtual environment so
OpenCV is present.

For an anchors-only annotation project that still keeps temporal evidence, use
`--separate-context`. Anchor files go to `images/` and neighbor files go to
`context/`; provenance for both remains in the same sampling manifest.

The descriptor cannot identify a yoyo, hand, string, or pose. The manifest
records `recognition_model_used=false`, parameters, source hashes, timestamps,
and anchor/context roles.

## Agent Coverage Pass

Inspect the normal, large, and center-crop anchor contact sheets plus raw
frames. Use large views to retain positives, hard negatives, occlusions, motion
blur, low contrast, edge clipping, crossings, behind-hand sections, behind-neck
sections, behind-body sections, half-route risks, and background-edge
distractors as reviewed cases using visibility, `bad_case`, and factual notes.

Use neighbor frames for continuity checks and trick orientation. Do not classify
horizontal play from a single static horizontal string segment.

Do not create evaluation partitions in this skill. Pass source groups,
visibility states, review metadata, and factual bad-case notes to the downstream
partitioner so it can build leakage-free train/validation/test collections.
