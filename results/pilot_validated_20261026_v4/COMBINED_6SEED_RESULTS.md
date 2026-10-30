# Combined 6-seed pilot result (2026-10-27)

Batches combined (config/source/protocol/manifest/split hashes verified identical):

- `pilot_validated_20261025_v3`: seeds 17, 42
- `pilot_validated_20261026_v4`: seeds 73, 101, 202, 303
- Combined gate artifact: `combined_6seed_gate.json` (gate computed with the
  unmodified `training.pipeline.gate_summary`).

## Gate

- status: `stop_for_review`
- performance gate: fail (M4 − M3 mean −0.54 pp, positive in 2/6 seeds)
- quality gate: pass (M4 precision_iou_075 0.946–0.958, coverage 0.586–0.594)

## Per-seed best validation AP (%)

| seed | M1 | M2 | M3 | M4 | M5 | M6 | M4−M3 | M3−M1 |
|-----:|----:|----:|----:|----:|----:|----:|------:|------:|
| 17 | 69.78 | 67.60 | 69.85 | 70.17 | 69.82 | 69.99 | +0.32 | +0.07 |
| 42 | 69.82 | 68.58 | 70.74 | 68.87 | 69.85 | 69.29 | −1.87 | +0.92 |
| 73 | 68.61 | 69.12 | 70.78 | 70.85 | 70.40 | 70.68 | +0.08 | +2.17 |
| 101 | 69.02 | 67.90 | 70.09 | 69.76 | 70.25 | 69.03 | −0.34 | +1.07 |
| 202 | 69.24 | 68.23 | 70.22 | 69.51 | 69.70 | 69.04 | −0.71 | +0.97 |
| 303 | 69.09 | 68.53 | 71.00 | 70.26 | 70.47 | 70.04 | −0.74 | +1.91 |

## Paired differences (percentage points, 6 seeds)

| comparison | mean | SD | min | max | wins | reference paired t / Wilcoxon p |
|---|---:|---:|---:|---:|---:|---|
| M4 − M3 | −0.54 | 0.78 | −1.87 | +0.32 | 2/6 | 0.147 / 0.156 |
| M3 − M1 | +1.19 | 0.76 | +0.07 | +2.17 | 6/6 | 0.012 / 0.031 |
| M5 − M1 | +0.82 | 0.75 | +0.02 | +1.79 | 6/6 | 0.043 / 0.031 |
| M4 − M1 | +0.64 | 1.06 | −0.95 | +2.24 | 5/6 | 0.197 / 0.219 |
| M2 − M1 | −0.93 | 0.89 | −2.18 | +0.51 | 1/6 | — |
| M6 − M1 | +0.42 | 0.95 | −0.54 | +2.07 | 4/6 | — |

p-values are reference-only; the frozen protocol explicitly does not treat the
AP comparison as a significance test.

## Conclusion

1. Temporal propagation is a stable positive: basic propagation (M3) beats
   human-only in 6/6 seeds (+1.19 pp mean); reliable propagation (M5) also wins
   6/6 but by less (+0.82 pp).
2. The proposed refinement (M4, event-aware + reliable propagation) does not
   improve AP over M3 (2/6 wins, mean −0.54 pp) even though it raises pseudo-label
   precision from 0.845 to 0.95 at similar coverage. Precision of pseudo labels is
   not the AP bottleneck in this setting.
3. Uncertainty-based spatial self-training (M2) is consistently harmful
   (−0.93 pp, 1/6 wins); temporal structure is what makes pseudo labels work.
4. Quality gate passes on all six seeds; the failure is purely the M4-vs-M3
   performance hypothesis.
