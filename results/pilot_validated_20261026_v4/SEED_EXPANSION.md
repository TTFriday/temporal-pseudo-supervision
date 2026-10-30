# Seed expansion record (2026-10-26)

## Reason

The frozen 2-seed pilot (`pilot_validated_20261025_v3`, seeds 17/42) returned
`stop_for_review`: the performance gate failed (M4 minus baseline: +0.32 / -1.87
pp) while the quality gate passed. The paired difference was dominated by
training stochasticity, so the human review step pre-designed by the protocol
was used to expand the seed count instead of leaving the gate at n=2.

## Amendment

- Added training seeds: 73, 101, 202, 303 (total 6 with 17, 42).
- Seeds were chosen and this record written before any new-seed run started.
- Nothing else changes: same config (`configs/validated_pilot.json`), same
  source hashes, same plant split, same label budget, same pseudo-label code,
  same gate definition. The gate is not redefined after seeing results.
- The train/val plant split is seed-independent; these six seeds measure
  optimisation/training stability, not split-level generalisation. The
  confirmation split remains a separate requirement (`confirmation_split_still_required`).

## Analysis plan

- Run the four new seeds in this directory (same pipeline, `--mode pilot`).
- Combine this directory's runs with the v3 runs (config/source/protocol hashes
  verified identical) and call the unmodified `training.pipeline.gate_summary`
  with all six seeds.
- Report per-seed paired differences, mean, and SD. Any significance-style
  number is reference-only, per the frozen protocol.
