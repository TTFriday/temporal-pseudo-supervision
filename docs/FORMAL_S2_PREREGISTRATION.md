# Formal S2 pre-registration — core budget sweep (2026-10-28)

Written before any formal-budget run. Scope chosen by the user after the
confirmation split passed: core S2 only (no random-query control, no
cross-genotype extension yet).

## Runs

| group | config | methods | seeds | new students |
|---|---|---|---|---:|
| budget 5% | `configs/formal_budget05.json` | M1–M4 | 17, 42, 73 | 12 |
| budget 20% | `configs/formal_budget20.json` | M1–M4 | 17, 42, 73 | 12 |
| full-label reference | `configs/formal_full100.json` | M1 (human-only) | 17, 42, 73 | 3 |

Outputs: `results/pilot_budget05`,
`pilot_budget20`, `pilot_full100`.

## Reused evidence (not rerun)

- 10% column: M1–M6 x seeds 17/42/73/101/202/303 from
  `pilot_validated_20261025_v3` and `pilot_validated_20261026_v4`. Those runs
  share identical source, manifest, split and protocol except `budget_ratio`,
  verified by provenance hashes; this makes them the 10% row of the same table.
- The 10% M5/M6 runs serve as the 2x2 mechanism ablation.

## Fixed factors

- Pilot development split (40 train / 12 val plants), same audit, same
  hyperparameters (the only config difference is `budget_ratio`).
- No source-code changes; no model or loss changes.
- Test plants are not read. The frozen endpoint evaluation remains a separate
  protocol that requires a later freeze.

## Analysis plan

- Per budget: paired M3−M1 (primary), M4−M3, M4−M1, M2−M1 over the three
  seeds; mean, SD, per-seed values. Descriptive only; no significance claims.
- Budget curve of M3−M1 over 5/10/20%, with the 10% column taken from the
  reused runs.
- Full-label reference (100%) is reported as context for the label-efficiency
  claim, not as a theoretical upper bound.
- No method, threshold or seed changes after seeing the formal results.
