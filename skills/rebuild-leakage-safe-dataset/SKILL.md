---
name: rebuild-leakage-safe-dataset
description: Rebuild or expand a manifest-driven machine-learning dataset without source-group or image leakage. Use when adding annotations to an existing train/val/test dataset, keeping existing split membership stable while distributing isolated incremental groups across splits, optionally freezing evaluation content for strict metric comparability, auditing a rebuilt JSON manifest, or creating split-lineage evidence before training.
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

First ask the external builder to create a temporary candidate manifest from
all currently approved annotations. This discovery build may use a fresh split
because it never replaces the active dataset:

```powershell
& '.\.venv\Scripts\python.exe' 'prepare_training_v2.py' `
  --output-dir 'tmp\incremental-discovery' --clear --resplit
```

Create a stable incremental plan. The plan ignores candidate assignments for
old groups, restores their baseline membership, and optimizes only new atomic
groups for the final sample-count ratios:

```powershell
& '.\.venv\Scripts\python.exe' `
  'skills\rebuild-leakage-safe-dataset\scripts\rebuild_leakage_safe.py' plan `
  --baseline 'datasets\yoyo_dataset\manifest.json' `
  --candidate 'tmp\incremental-discovery\manifest.json' `
  --output 'annotations\lineage\incremental-plan.json' `
  --report 'annotations\lineage\plan-report.json'
```

Treat a failed ratio or lineage check as a hard gate. Inspect
`new_groups_by_split` before rebuilding.

## Rebuild

Pass the builder after `--`; the wrapper uses `subprocess` without a shell.
Give the completed plan to any builder that accepts a source-group assignment
manifest. This project's adapter calls that input `--freeze-splits-from`; all
new groups are already assigned in the plan, so the builder does not force
them into `train`.

```powershell
& '.\.venv\Scripts\python.exe' `
  'skills\rebuild-leakage-safe-dataset\scripts\rebuild_leakage_safe.py' run `
  --manifest 'datasets\yoyo_dataset\manifest.json' `
  --snapshot-out 'annotations\lineage\dataset-before.json' `
  --report 'annotations\lineage\rebuild-report.json' `
  --mode append-isolated --allow-command-without-baseline --dry-run -- `
  '.\.venv\Scripts\python.exe' 'prepare_training_v2.py' --clear `
  --freeze-splits-from 'annotations\lineage\incremental-plan.json'
```

Inspect the resolved paths and command printed by dry-run. Then repeat without
`--dry-run`. Do not use a builder's fresh-resplit option in this active rebuild.

For `strict-eval`, skip `plan`, use the literal `{baseline_manifest}` token as
the builder's assignment input, and omit `--allow-command-without-baseline`.

## Verify An Existing Rebuild

Run verification without rebuilding when two manifests already exist:

```powershell
& '.\.venv\Scripts\python.exe' `
  'skills\rebuild-leakage-safe-dataset\scripts\rebuild_leakage_safe.py' verify `
  --baseline 'annotations\lineage\dataset-before.json' `
  --rebuilt 'datasets\yoyo_dataset\manifest.json' `
  --mode append-isolated `
  --report 'annotations\lineage\verify-report.json'
```

Treat a nonzero exit code as a hard gate: do not train, evaluate, or promote a
model from that rebuild. If `evaluation_expanded=true`, version the evaluation
protocol and avoid presenting the new metrics as directly identical to the
old protocol even though the old samples stayed fixed.

Run `scripts/self_test.py` after changing the skill.
