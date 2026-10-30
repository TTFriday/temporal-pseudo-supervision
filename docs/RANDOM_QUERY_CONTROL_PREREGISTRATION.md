# Random-query control pre-registration (2026-10-29)

Written before any control run. Purpose: separate the effect of *propagation*
from the effect of *which frames were labelled*, so the manuscript does not
overstate the role of the time-coverage selector.

## Design

| method | selector | pseudo supervision | comparable to |
|---|---|---|---|
| R1 | random (seed 42, single fixed draw) | none | M1 |
| R2 | random (same draw) | basic SAM2 propagation | M3 |

- Budget 10% (166 labels, the same 40 common seed frames as the reused
  column), config `configs/validated_pilot.json`, training seeds 17/42/73.
- Same audit, manifest, plant split and hyperparameters as the reused 10%
  column. The selector's plant-balancing rule keeps per-plant counts identical
  to time coverage, so only the frame choice changes.
- The control methods are added to the pipeline's method table at runtime by
  `controls/run_random_query.py`; files under `training/`, `selection/`,
  `evaluation/` and `propagation/` are not modified, so the recorded source
  hashes stay identical to the 10% column. The runner hash, method definitions
  and the single-draw choice are written to `control_manifest.json` beside the
  results.
- The random draw is fixed across training seeds (`select_query_keys` default
  seed 42), matching the deterministic selection used by M1/M3; only training
  randomness varies, as in the main table.

## Readings fixed in advance (descriptive, no significance claims)

- M3−R2: if the mean is within ±0.5 pp, the propagation benefit is
  selection-agnostic; if M3 is more than 1.0 pp above R2 with the same sign in
  all three seeds, temporal selection adds value on top of propagation;
  otherwise report no clear separation.
- M1−R1: same rule for the supervision-only comparison.
- R2−R1: propagation effect under random selection, for comparison with
  M3−M1 (+1.05 pp over these seeds).
- No thresholds are used as a pass/fail gate and no method, seed or analysis
  rule will be changed after seeing the results.

## Output

`results/random_query_control`, analysis written
to `random_control.json` and `RANDOM_QUERY_CONTROL.md` by
`analysis/random_control.py`.
