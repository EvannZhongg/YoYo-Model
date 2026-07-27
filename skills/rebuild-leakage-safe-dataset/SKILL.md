---
name: rebuild-leakage-safe-dataset
description: Rebuild or expand a manifest-driven machine-learning dataset without source-group or image leakage. Use when adding annotations to an existing train/val/test dataset, preserving old assignments while allowing isolated new groups to be distributed reasonably, strictly freezing evaluation content when required, auditing a rebuilt JSON manifest, or creating split-lineage evidence before training.
---

# Rebuild Leakage-Safe Dataset

Keep this skill independent from the host repository. Use only the bundled
standard-library script and pass the host dataset builder as an external
command. Never import or patch the host training modules from this skill.

## Choose The Policy

- Use `append-isolated` by default. Existing source groups and image hashes
  must remain in their original split. New complete source groups may enter
  `train`, `val`, or `test`; the report records any evaluation expansion.
- Use `strict-eval` when metrics must remain directly comparable. It adds the
  requirement that no new image may enter `val` or `test`; new groups must be
  train-only.

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

Use the project virtual environment. Pass the builder after `--`; the wrapper
uses `subprocess` without a shell. Include the literal `{baseline_manifest}`
token wherever the builder accepts its split-lineage input.

```powershell
& '.\.venv\Scripts\python.exe' `
  'skills\rebuild-leakage-safe-dataset\scripts\rebuild_leakage_safe.py' run `
  --manifest 'datasets\yoyo_dataset\manifest.json' `
  --snapshot-out 'annotations\lineage\dataset-before.json' `
  --report 'annotations\lineage\rebuild-report.json' `
  --mode append-isolated --dry-run -- `
  '.\.venv\Scripts\python.exe' 'prepare_training_v2.py' --clear `
  --freeze-splits-from '{baseline_manifest}'
```

Inspect the resolved paths and command printed by dry-run. Then repeat without
`--dry-run`. Do not use a builder's fresh-resplit option in the same command.

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
