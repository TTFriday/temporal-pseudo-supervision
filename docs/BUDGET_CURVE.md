# Formal S2 budget sweep — results (analysis frozen 2026-10-28)

Analysis follows `FORMAL_S2_PREREGISTRATION.md` (copied beside this file).
Descriptive paired comparisons over seeds 17/42/73; no significance claims, no
method or threshold changes after the runs.

## Provenance

All five columns share identical `source_sha256`, `manifest_sha256`,
`split_sha256` and `audit_sha256` (checked in `formal_budget.json`); only
`budget_ratio` differs:

| column | directory | budget_ratio | labels |
|---|---|---:|---:|
| 5% | `pilot_budget05` | 0.05 | 83 |
| 10% | `pilot_validated_20261025_v3` + `_v4` (reused) | 0.10 | 166 |
| 20% | `pilot_budget20` | 0.20 | 332 |
| 100% (reference) | `pilot_full100` | 1.00 | 1659 |

All runs: 2000 updates, V=72 validation labels, at most 2 AMP-skipped updates
per training segment (limit 10), no early stopping triggered.

## AP by budget (mean over seeds 17/42/73, %)

| budget | M1 | M2 | M3 | M4 | 100% M1 |
|---|---:|---:|---:|---:|---:|
| 5% (83) | 67.44 | 68.22 | 68.69 | **69.33** | 71.19 |
| 10% (166) | 69.41 | 68.43 | **70.46** | 69.97 | 71.19 |
| 20% (332) | 70.63 | 68.94 | **70.73** | 70.52 | 71.19 |
| 100% (1659) | 71.19 | — | — | — | 71.19 |

## Paired comparisons (pp, seeds 17/42/73)

| comparison | 5% | 10% | 20% |
|---|---:|---:|---:|
| M3−M1 (primary) | **+1.25** (3/3) | **+1.05** (3/3) | +0.11 (1/3) |
| M4−M1 | +1.89 (3/3) | +0.56 (2/3) | −0.11 (1/3) |
| M4−M3 | +0.64 (3/3) | −0.49 (2/3) | −0.22 (2/3) |
| M2−M1 | +0.78 (2/3) | −0.97 (1/3) | −1.68 (0/3) |

Per-seed M3−M1: 5% [+0.91, +1.62, +1.22]; 10% [+0.07, +0.92, +2.17];
20% [−0.15, −0.29, +0.76]. The 10% column over all six reused seeds gives
+1.19 (6/6).

## Budget curve and label efficiency

M3−M1 shrinks monotonically as labels grow: +1.25 → +1.05 → +0.11 pp. The
benefit of temporal propagation is largest in the scarce-label regime and is
exhausted by 20% labels.

Descriptive budget-equivalence (same AP level reached with fewer labels):

- 83 labels + basic propagation (M3, 68.69) is 0.72 pp below 166 human-only
  labels (M1, 69.41); with event-aware selection (M4, 69.33) the gap is 0.08 pp.
- 166 labels + basic propagation (70.46) is 0.17 pp below 332 human-only labels
  (70.63).
- 332 labels + basic propagation (70.73) is 0.46 pp below the full 1659-label
  reference (71.19).

Human-only M1: 67.44 → 69.41 → 70.63 → 71.19 across 5/10/20/100%.

## Mechanism dissociation (unchanged from pilot/confirmation)

Reliable propagation (M4) raises pseudo-label precision from 0.845 (M3) to
0.957–0.958 at slightly lower coverage (0.58–0.60 vs 0.62) at both 5% and 20%,
yet M4−M3 is +0.64 (5%, 3/3), −0.49 (10%, 2/3) and −0.22 (20%, 2/3): better
label quality does not translate into a stable AP gain. Uncertainty-driven
querying (M2) degrades with more labels (+0.78 → −0.97 → −1.68).

## Caveats

- Three seeds per new budget; the 10% column has six. Comparisons are paired
  and descriptive.
- M5/M6 mechanism ablation exists only at 10% (reused v3/v4 runs).
- 100% M1 is context for the label-efficiency claim, not a theoretical upper
  bound for the low-label methods.
- Test plants were never read; the frozen endpoint evaluation stays a separate
  protocol.

Files: `formal_budget.json` (machine-readable), `FORMAL_S2_PREREGISTRATION.md`,
`pilot_budget05/`, `pilot_budget20/`, `pilot_full100/`.
