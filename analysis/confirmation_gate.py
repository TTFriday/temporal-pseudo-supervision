"""Pre-registered confirmation gate for the low-label temporal segmentation pilot.

Primary comparison (amended 2026-10-27, before the confirmation run):
M3 (basic temporal propagation) minus M1 (human-only) must have a mean of at
least +1.0 percentage points over the six pre-registered seeds and be positive
on every seed. Thresholds are unchanged from the frozen pilot gate; only the
primary method pair changed, following the six-seed pilot evidence.

Usage:
    python analysis/confirmation_gate.py --results <run>/results_final.json
"""

import argparse
import json
import statistics
from pathlib import Path

SEEDS = [17, 42, 73, 101, 202, 303]
METHODS = ["M1", "M2", "M3", "M4", "M5", "M6"]
PRIMARY = ("M3", "M1")
MEAN_THRESHOLD_PP = 1.0


def load_runs(path):
    data = json.loads(Path(path).read_text())
    runs = {(r["method"], r["seed"]): r for r in data["runs"]}
    missing = [(m, s) for m in METHODS for s in SEEDS if (m, s) not in runs]
    if missing:
        raise SystemExit(f"incomplete runs, missing {len(missing)}: {missing[:5]}...")
    return runs


def ap(runs, method, seed):
    return 100.0 * runs[(method, seed)]["student"]["best_validation_AP"]


def paired(runs, a, b):
    diffs = [ap(runs, a, s) - ap(runs, b, s) for s in SEEDS]
    return {
        "mean_pp": statistics.mean(diffs),
        "sd_pp": statistics.stdev(diffs),
        "min_pp": min(diffs),
        "max_pp": max(diffs),
        "positive_seeds": sum(d > 0 for d in diffs),
        "per_seed_pp": dict(zip(map(str, SEEDS), map(lambda x: round(x, 3), diffs))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    runs = load_runs(args.results)

    primary = paired(runs, *PRIMARY)
    passed = primary["mean_pp"] >= MEAN_THRESHOLD_PP and primary["positive_seeds"] == len(SEEDS)

    secondary = {
        f"{a}-{b}": paired(runs, a, b)
        for a, b in [("M5", "M1"), ("M4", "M1"), ("M4", "M3"), ("M2", "M1"), ("M6", "M1")]
    }
    quality = {
        str(s): {
            "precision_iou_075": runs[("M4", s)]["diagnostic"]["precision_iou_075"],
            "coverage_iou_075": runs[("M4", s)]["diagnostic"]["coverage_iou_075"],
        }
        for s in SEEDS
    }
    quality_pass = all(
        q["precision_iou_075"] >= 0.85 and q["coverage_iou_075"] >= 0.30
        for q in quality.values()
    )

    gate = {
        "status": "confirmation_pass" if passed else "confirmation_stop_for_review",
        "primary_comparison": f"{PRIMARY[0]}-{PRIMARY[1]}",
        "primary_rule": f"mean >= {MEAN_THRESHOLD_PP} pp and all seeds > 0",
        "primary": primary,
        "primary_pass": passed,
        "secondary_not_gating": secondary,
        "m4_quality_not_gating": {"pass": quality_pass, "per_seed": quality},
        "reference_only_not_significance": True,
        "seeds": SEEDS,
        "test_set_untouched": True,
    }

    out = Path(args.results).parent / "confirmation_gate.json"
    out.write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
