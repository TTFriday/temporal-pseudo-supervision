# Random-query control at 10% — results (2026-10-29)

Plan fixed before the run in `splits/RANDOM_QUERY_CONTROL_PREREGISTRATION.md`.
Descriptive comparisons over seeds 17/42/73; no significance claims.

## Integrity

- Control runs recorded `source_sha256`, `manifest_sha256`, `split_sha256` and
  `audit_sha256` identical to the reused 10% column (verified in
  `random_control.json`); only `budget_ratio` (0.10), methods and query
  selector differ.
- 6 students, 2000 updates each, B=166 labels, V=72, at most 2 AMP-skipped
  updates per training segment, no early stopping.
- The random selector is one fixed draw (seed 42) shared by the three training
  seeds; the plant-allocation rule keeps per-plant label counts identical to
  time coverage. Query overlap with M3 per seed: 51 shared, 115 M3-only,
  115 random-only (the 40 common seed frames plus 11 coincidences).

## AP (%) per seed

| method | 17 | 42 | 73 | mean |
|---|---:|---:|---:|---:|
| M1 (time coverage, no propagation) | 69.78 | 69.82 | 68.61 | 69.41 |
| R1 (random, no propagation) | 68.80 | 68.71 | 68.82 | 68.77 |
| M3 (time coverage + basic propagation) | 69.85 | 70.74 | 70.78 | 70.46 |
| R2 (random + basic propagation) | 68.57 | 69.63 | 69.86 | 69.35 |

## Primary comparisons

| comparison | per seed (pp) | mean | SD | seeds positive | pre-registered reading |
|---|---|---:|---:|---:|---|
| M3−R2 | +1.29, +1.11, +0.92 | **+1.10** | 0.19 | 3/3 | temporal selection adds value on top of propagation |
| M1−R1 | +0.98, +1.12, −0.20 | +0.63 | 0.73 | 2/3 | no clear separation |

## Secondary comparisons

| comparison | per seed (pp) | mean | SD | seeds positive |
|---|---|---:|---:|---:|
| M3−M1 | +0.07, +0.92, +2.17 | +1.05 | 1.06 | 3/3 |
| R2−R1 | −0.23, +0.93, +1.05 | +0.58 | 0.71 | 2/3 |

## Interpretation

Temporal coverage matters specifically when propagation is used: with basic
SAM2 propagation, time-coverage labels beat random labels by 1.10 pp in all
three seeds, and the propagation gain is larger under time coverage (+1.05)
than under random selection (+0.58). Without propagation the selection
difference is within noise (+0.63, one negative seed), so the manuscript should
describe the contribution as the combination of temporally spread labels and
propagation, not as a standalone benefit of the selector.

The control is one random draw; the draw is not unusually weak (R2 is within
0.06 pp of M1), but selection variance is not characterised by this run and
belongs in the limitations. No method, seed or reading rule changed after the
results. The official test plants were not accessed.

Files: `random_control.json` (machine-readable), `control_manifest.json`,
`results_final.json`, `pilot_validated_20261025_v3/` + `_v4/` (reused column),
`analysis/random_control.py`.
