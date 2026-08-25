---
name: rebuild-leakage-safe-dataset
description: Rebuild or expand a manifest-driven machine-learning dataset while preserving source-group splits, preventing image leakage, and transactionally retaining canonical labels and review mappings. Use when adding annotations to an existing train/val/test dataset, freezing evaluation content, or auditing split lineage before training.
---

# Rebuild Leakage-Safe Dataset

Use the bundled `scripts/rebuild_leakage_safe.py` as a repository-independent
guard around the host dataset builder. The builder is supplied as an external
command; this skill does not import or modify host training code.

## Policies

- `append-isolated` (default): keep every existing source group and image hash
  in its original split. Assign only new complete source groups to splits near
  the target ratios; validation and test may therefore expand.
- `strict-eval`: keep evaluation content frozen for metric comparability. New
  source groups and images are train-only.

Both policies require the rebuilt manifest to have no missing or moved old
groups/images, no source-group or image-hash overlap, consistent record and
assignment splits, zero declared leakage counts, and final ratios within the
configured deviation limit.

## Manifest Contract

The manifest is one JSON object containing:

- `split_policy.source_groups.train|val|test`: unique source-group IDs.
- `records`: one image record with `source_group`, `split`, lowercase
  SHA-256 `image_sha256`, `trick_orientation`, boolean `yoyo_positive`, boolean
  `string_positive`, and `string_visibility`.
- Optional `split_policy.target_sample_ratios` (default `0.70/0.15/0.15`).

Source groups are atomic. Assignment should prioritize final sample-count
ratios, then the task-label distributions (yoyo presence, string presence,
orientation, and string visibility). A supported label feature must occur in
every split once it has at least three supporting source groups; increase the
threshold with `--min-support-groups` when appropriate, but never below three.

Manifests and plans are generated artifacts, not hand-edited files. Keep the
baseline snapshot, discovery output, and lineage reports outside the active
dataset directory so a clearing rebuild cannot remove them.

## Workbench Protection

Use `protected-run` whenever the active dataset contains editable
`canonical/labels` or a human review map; use plain `run` only for datasets
without that state. A protected rebuild must:

- validate existing review entries against their current label revisions;
- snapshot the complete active dataset and review map outside the output;
- include the protected canonical overlay in builder inputs;
- preserve the JSON values of existing canonical labels and remove unsupported
  non-task fields from rebuilt labels;
- rebind review entries by image SHA to the rebuilt label paths/revisions;
- restore the original dataset if building or validation fails.

Keep the snapshots after success while they are referenced by manifests or
provenance. Prevent concurrent label or review writes during the transaction.

## Workflow

1. Build a candidate manifest from the active canonical overlay plus all
   eligible annotation exports. Write it outside the active dataset.
2. For `append-isolated`, run the `plan` subcommand with baseline and candidate
   manifests. It restores baseline assignments and allocates only new atomic
   groups. Inspect `new_groups_by_split` and `label_balance`; any lineage,
   ratio, or coverage failure is a hard gate.
3. Run `protected-run` for workbench datasets, otherwise `run`, passing the
   host builder after `--`. Use the generated plan with
   `--freeze-splits-from`; for `strict-eval`, use the baseline assignment input
   and keep new groups train-only. A protected builder must include the
   `{protected_canonical}` source token.
4. When a rebuild already exists, run `verify` against the retained baseline
   manifest before training or evaluation.

Use a dry run before a protected execution and keep its backup/review paths
unchanged for the actual run. Treat a nonzero exit, `ok=false`, or
`rolled_back=true` as a failed rebuild and do not train from it.

## Release Gates

Before training, require a successful report with preserved old assignments and
images. For protected runs also require the protected-label count to equal the
baseline sample count, all reviewed entries to be rebound, no residual
non-task fields, and the dataset backup retained. In `append-isolated`, require
`label_balance.coverage_gaps=[]`.

If `evaluation_expanded=true`, version the evaluation protocol and do not label
new metrics as directly identical to metrics from the smaller evaluation set.
