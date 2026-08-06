---
name: rebuild-leakage-safe-dataset
description: Rebuild or expand a manifest-driven machine-learning dataset without source-group or image leakage while transactionally preserving workbench-edited canonical labels and SHA-bound human review mappings. Use when adding annotations to an existing train/val/test dataset, keeping existing split membership stable, protecting manual edits and verification state, optionally freezing evaluation content for metric comparability, auditing a rebuilt manifest, or creating split-lineage evidence before training.
---

# Rebuild Leakage-Safe Dataset

Keep this skill independent from the host repository. Use only the bundled
standard-library script and pass the host dataset builder as an external
command. Never import or patch the host training modules from this skill.

## Choose The Policy

- Use `append-isolated` by default. This does not freeze evaluation-set
  contents. Keep every existing source group and image hash in its original
  split, then allocate each new complete source group to `train`, `val`, or
  `test` according to the target ratios. Record evaluation expansion.
- Use `strict-eval` only when metrics must remain directly comparable to the
  old evaluation protocol. It additionally requires all new groups to be
  train-only, so no new image enters `val` or `test`.

Both policies reject missing or moved old samples, source-group overlap,
duplicate image hashes, split disagreement between records and assignments,
nonzero declared leakage counts, and excessive final ratio deviation.

## Protect Workbench State

Use `protected-run` for every active dataset that exposes editable
`canonical/labels` or has a human review map. Never use plain `run` for that
dataset. The protected action:

- validates every existing review SHA before changing the dataset;
- atomically moves the complete active dataset to a unique backup;
- requires `{protected_canonical}` in the builder command so manual labels are
  included as a source;
- normalizes canonical labels to the current task contract and rejects any other existing canonical JSON change except `dataset_management`;
- rebinds review entries by image SHA to the rebuilt label paths and hashes;
- restores the original dataset automatically if build or validation fails.

Keep the dataset backup and review-map snapshot after success. Do not delete a
backup while the active manifest or canonical provenance references it. A
manifest-only snapshot is not sufficient workbench protection.

## Manifest Contract

Require one JSON object containing:

- `split_policy.source_groups.train|val|test`: arrays of unique source IDs.
- `records`: one record per image with `source_group`, `split`, and lowercase
  SHA-256 `image_sha256`.
- Optional `split_policy.target_sample_ratios`; defaults to `0.70/0.15/0.15`.

Do not hand-edit either manifest. Store the baseline snapshot outside the
dataset output directory so a clearing rebuild cannot delete it.

## Preflight

Use the project virtual environment. Keep discovery output and lineage files
outside the active dataset directory.

Stop the workbench server, or otherwise guarantee that no label save or human
review update can occur during the protected rebuild. The transaction protects
files on disk but does not coordinate with a concurrently open editor.

First create a source list containing the active canonical overlay and every
eligible annotation export. This ensures the candidate contains manual edits
as well as new annotations. The discovery build may use a fresh split because
it never replaces the active dataset:

```powershell
$annotationSources = @(
  Get-ChildItem -LiteralPath 'annotations' -Directory |
    Where-Object {
      $_.Name -ne 'score_annotations' -and
      (Test-Path -LiteralPath (Join-Path $_.FullName 'labels'))
    } |
    Select-Object -ExpandProperty FullName
)
$discoveryArgs = @(
  '-m', 'cli.dataset.prepare_training',
  '--output-dir', 'tmp\incremental-discovery',
  '--clear', '--resplit', '--source', 'datasets\1Ayoyo_dataset\canonical'
)
foreach ($source in $annotationSources) { $discoveryArgs += @('--source', $source) }
& '.\.venv\Scripts\python.exe' @discoveryArgs
if ($LASTEXITCODE -ne 0) { throw 'protected discovery failed' }
```

Create a stable incremental plan. The plan ignores candidate assignments for
old groups, restores their baseline membership, and optimizes only new atomic
groups for the final sample-count ratios:

```powershell
& '.\.venv\Scripts\python.exe' `
  'skills\rebuild-leakage-safe-dataset\scripts\rebuild_leakage_safe.py' plan `
  --baseline 'datasets\1Ayoyo_dataset\manifest.json' `
  --candidate 'tmp\incremental-discovery\manifest.json' `
  --output 'annotations\lineage\incremental-plan.json' `
  --report 'annotations\lineage\plan-report.json'
```

Treat a failed ratio or lineage check as a hard gate. Inspect
`new_groups_by_split` before rebuilding.

## Protected Rebuild

Create unique backup paths once and reuse them for dry-run and execution. Pass
the builder after `--`; the wrapper uses `subprocess` without a shell. Put the
`{protected_canonical}` token first among builder sources, then pass all
annotation exports. Give the completed plan through `--freeze-splits-from`.

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$datasetBackup = "annotations\lineage\backups\yoyo-dataset-$stamp"
$reviewSnapshot = "annotations\lineage\backups\dataset-review-status-$stamp.json"
$protectedBuilder = @(
  '.\.venv\Scripts\python.exe', '-m', 'cli.dataset.prepare_training',
  '--output-dir', 'datasets\1Ayoyo_dataset', '--clear',
  '--freeze-splits-from', 'annotations\lineage\incremental-plan.json',
  '--source', '{protected_canonical}'
)
foreach ($source in $annotationSources) { $protectedBuilder += @('--source', $source) }
$guardArgs = @(
  'skills\rebuild-leakage-safe-dataset\scripts\rebuild_leakage_safe.py',
  'protected-run', '--manifest', 'datasets\1Ayoyo_dataset\manifest.json',
  '--backup-dir', $datasetBackup,
  '--review-map', 'workbench_state\dataset_review_status.json',
  '--review-snapshot-out', $reviewSnapshot,
  '--review-dataset-key', '1Ayoyo_dataset',
  '--report', 'annotations\lineage\rebuild-report.json',
  '--mode', 'append-isolated', '--allow-command-without-baseline'
)
& '.\.venv\Scripts\python.exe' @guardArgs '--dry-run' '--' @protectedBuilder
if ($LASTEXITCODE -ne 0) { throw 'protected rebuild dry-run failed' }
```

Inspect every resolved path and source printed by dry-run. Confirm that the
backup and review snapshot do not exist, then execute:

```powershell
& '.\.venv\Scripts\python.exe' @guardArgs '--' @protectedBuilder
if ($LASTEXITCODE -ne 0) { throw 'protected rebuild failed or was rolled back' }
```

Do not change `$stamp`, `$datasetBackup`, `$reviewSnapshot`, `$guardArgs`, or
`$protectedBuilder` between calls. Do not use `--resplit`.

Treat nonzero exit as a hard gate. Confirm `protected_label_count` matches the
baseline sample count, `review_entry_count_rebound` matches the preflight
review count, `non_task_field_residual_count=0`, `dataset_backup_retained=true`,
and `ok=true` before training.
If `rolled_back=true`, diagnose the rejected candidate and start again with
new backup paths; never bypass the label-content check.

For `strict-eval`, skip `plan`, use `{baseline_manifest}` as the builder's
assignment input, and omit `--allow-command-without-baseline`. Keep
`{protected_canonical}` as a builder source.

## Verify An Existing Rebuild

Run verification without rebuilding when two manifests already exist:

```powershell
& '.\.venv\Scripts\python.exe' `
  'skills\rebuild-leakage-safe-dataset\scripts\rebuild_leakage_safe.py' verify `
  --baseline (Join-Path $datasetBackup 'manifest.json') `
  --rebuilt 'datasets\1Ayoyo_dataset\manifest.json' `
  --mode append-isolated `
  --report 'annotations\lineage\verify-report.json'
```

Treat a nonzero exit code as a hard gate: do not train, evaluate, or promote a
model from that rebuild. If `evaluation_expanded=true`, version the evaluation
protocol and avoid presenting the new metrics as directly identical to the
old protocol even though the old samples stayed fixed.

Run `scripts/self_test.py` after changing the skill.
