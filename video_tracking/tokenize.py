"""Reserved design notes for future event-level motion tokenization.

The former implementation exported a fixed-width per-frame vector. It was not
consumed by any recognition model and imposed an unstable feature contract, so
tracking no longer imports or writes those artifacts.

Future implementation requirements:

1. Read the source-resolution ``frames.jsonl`` records and align them with
   score-annotation evidence intervals by video fingerprint and timestamps.
2. Keep modality-specific inputs separate: yoyo motion, multi-component string
   geometry, orientation probabilities, pose, confidence, visibility, and
   review quality flags should each have an explicit presence mask.
3. Normalize geometry in a person/yoyo reference frame; do not use absolute
   pixel coordinates as the event identity.
4. Preserve variable-length intervals and uncertain boundaries. An annotated
   score event is not necessarily a complete trick, so support one-to-many and
   many-to-one alignment between score events and trick segments.
5. Produce learned event embeddings with a temporal encoder. Use score-family
   and score-delta supervision first, then action names, positive pairs, or
   supervised contrastive loss when those annotations exist.
6. Split by source video/player/competition before training or retrieval
   evaluation to prevent near-duplicate event leakage.
7. Version the event dataset and embedding schema independently from the
   perception-model checkpoints. Do not reintroduce a fixed division one-hot
   feature unless a downstream model explicitly needs group conditioning.

This module intentionally contains no executable tokenizer API yet.
"""
