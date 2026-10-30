# Confirmation protocol — pre-registration (2026-10-27)

Written before any confirmation-split training run. This amends the primary
hypothesis only; it does not change training code, hyperparameters, the data
audit, the label budget, or the gate thresholds.

## Context

The frozen six-seed pilot (same 40 train / 12 val plants; seeds 17, 42, 73,
101, 202, 303) rejected the original primary hypothesis (M4, event-aware
reliable propagation, beating the pseudo-label baseline by at least 1 pp) and
produced a stable alternative:

- M3 (basic temporal propagation) − M1 (human-only): +1.19 pp mean, positive on
  6/6 seeds (reference paired t p=0.012, reported as reference only).
- M4 − M3: −0.54 pp mean, positive on 2/6 seeds, despite raising pseudo-label
  precision from 0.845 to about 0.95 at unchanged coverage.

## Amended primary hypothesis

M3 (SAM2 basic temporal propagation + positive-only partial supervision) beats
M1 (human-only, same 10% label budget) on validation AP.

Decision rule (unchanged thresholds, corrected comparison):

- PASS iff mean(M3 − M1) >= +1.0 pp across the six seeds AND every seed
  difference is > 0.
- Otherwise the confirmation returns `confirmation_stop_for_review`.

Secondary, non-gating outcomes: M5 − M1 and M4 − M1, the M4 − M3 dissociation,
the M4 pseudo-label quality gate (precision_iou_075 >= 0.85 and
coverage_iou_075 >= 0.30), and M2/M6 as controls.

## Confirmation split

- Plants are disjoint from the pilot split by construction: the pilot train and
  val plants are removed from the pool first, then
  `run_pilot.subsample_plants` (stratified by genotype) selects 40 train and 12
  val plants with `selection_seed = 2026`.
- Resulting coverage: 21 genotypes in train, 12 in val; same sizes and 10%
  frame budget as the pilot.
- Machine-readable split: `splits/confirmation_split.json` (sha-pinned source
  files inside).
- Test plants remain untouched; no test RGB, masks, or metrics are read.

## Pre-registered run

- Config: `configs/confirmation.json` (identical to the pilot except the split
  reference).
- Seeds: 17, 42, 73, 101, 202, 303 (same as the pilot, pre-registered here).
- Methods: M1–M6 (full table for the paper; primary comparison M3 vs M1).
- Analysis: `analysis/confirmation_gate.py`, which consumes the run's
  `results_final.json` and writes `confirmation_gate.json`.

## Commitment

If the primary rule fails on the confirmation split, the hypothesis is not
tuned again on these plants. The outcome is reported as-is and the paper
narrative is adjusted accordingly instead of re-searching hyperparameters or
selection rules on the confirmation data.
