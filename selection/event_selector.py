"""Pure CPU prediction evidence and balanced, prefix-stable query selection.

No images, annotations, models, or training code are loaded here. Prediction
masks must already share the original image coordinates (including BOTH inverse
flips). Boxes, when supplied, are xywh metadata; all matching uses masks instead.
"""

from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from fractions import Fraction
from numbers import Integral, Real
import random

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata


__all__ = ["frame_key", "select_query_keys", "compute_signals",
           "mask_iou_matrix", "match_instances"]


def frame_key(frame):
    """Return the immutable (plant_id, replicate_id, filename) identity."""
    return frame["plant_id"], frame["replicate_id"], frame["filename"]


def mask_iou_matrix(listA, listB):
    """Return float64 mask IoUs for two lists of instance dictionaries.

    Masks must be boolean HxW arrays of the same shape. Empty masks have zero
    IoU, including against another empty mask. Empty lists retain (N, M) shape.
    Pixel counts do not use uint8 arithmetic, so large masks cannot overflow.
    """
    masks_a = [np.asarray(instance["mask"]) for instance in listA]
    masks_b = [np.asarray(instance["mask"]) for instance in listB]
    shape = None
    for mask in masks_a + masks_b:
        if mask.ndim != 2 or mask.dtype != np.bool_:
            raise ValueError("instance masks must be boolean HxW arrays")
        if shape is not None and mask.shape != shape:
            raise ValueError("instance masks must have the same shape")
        shape = mask.shape
    ious = np.zeros((len(masks_a), len(masks_b)), dtype=np.float64)
    areas_a = [np.count_nonzero(mask) for mask in masks_a]
    areas_b = [np.count_nonzero(mask) for mask in masks_b]
    for i, mask_a in enumerate(masks_a):
        for j, mask_b in enumerate(masks_b):
            intersection = np.count_nonzero(mask_a & mask_b)
            union = areas_a[i] + areas_b[j] - intersection
            if union:
                ious[i, j] = intersection / union
    return ious


def _validate_threshold(value, name):
    if not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} threshold must be finite and in [0, 1]")


def match_instances(listA, listB, threshold):
    """Return (index_a, index_b, mask_iou) triples, each index used at most once.

    Hungarian assignment maximizes the number of positive-IoU matches meeting
    the inclusive threshold, then their total IoU. Subthreshold edges cannot
    steal valid matches. Results are ordered by index_a.
    """
    _validate_threshold(threshold, "IoU")
    ious = mask_iou_matrix(listA, listB)
    if not ious.size:
        return []
    valid = (ious >= threshold) & (ious > 0)
    # One additional valid match outweighs any change in the total IoU.
    reward = np.where(valid, min(ious.shape) + 1 + ious, 0.0)
    rows, columns = linear_sum_assignment(-reward)
    return [(int(i), int(j), float(ious[i, j])) for i, j in zip(rows, columns)
            if valid[i, j]]


def _disagreement(first, second, matches):
    if not first and not second:
        return 0.0
    return float(np.clip(1 - sum(iou for _, _, iou in matches)
                         / max(len(first), len(second), 1), 0, 1))


def compute_signals(original, augmented, propagated=None,
                    conf_threshold=0.8, iou_threshold=0.5):
    """Compute U, D, E, n_detected, has_propagation and fallback_reason.

    Inputs are lists of {mask: bool[H,W], score: float, bbox: xywh} instances.
    ``augmented`` contains exactly two lists ALREADY inverse transformed by the
    caller. U is the mean of 1 - sum(matched IoU)/max(instance counts) for those
    two views. D uses the same bounded disagreement against propagation. Two
    empty lists have disagreement zero; one empty list has disagreement one.

    E is the fraction of original detections that are high confidence, matched
    to high-confidence detections in BOTH views, and unmatched to propagation.
    Confidence and IoU thresholds are inclusive. All three signals are in [0,1].
    ``propagated=None`` means unavailable evidence (D=E=0, no_propagation), while
    ``[]`` is a valid empty cache and can provide positive event evidence.
    No original detections takes precedence as the no_detection fallback.
    """
    _validate_threshold(conf_threshold, "confidence")
    _validate_threshold(iou_threshold, "IoU")
    if len(augmented) != 2:
        raise ValueError("augmented must contain exactly two inverse-transformed views")
    for instances in [original, *augmented]:
        for instance in instances:
            _validate_threshold(instance["score"], "instance confidence")

    view_matches = [match_instances(original, view, iou_threshold) for view in augmented]
    instability = float(np.mean([_disagreement(original, view, matches)
                                 for view, matches in zip(augmented, view_matches)]))
    stable = {i for i, instance in enumerate(original)
              if instance["score"] >= conf_threshold}
    for view, matches in zip(augmented, view_matches):
        stable.intersection_update(i for i, j, _ in matches
                                   if view[j]["score"] >= conf_threshold)

    has_propagation = propagated is not None
    divergence, events = 0.0, 0.0
    if has_propagation:
        propagation_matches = match_instances(original, propagated, iou_threshold)
        divergence = _disagreement(original, propagated, propagation_matches)
        covered = {i for i, _, _ in propagation_matches}
        events = len(stable - covered) / max(len(original), 1)
    reason = "no_detection" if not original else (
        None if has_propagation else "no_propagation")
    return {"U": instability, "D": divergence, "E": events,
            "n_detected": len(original), "has_propagation": has_propagation,
            "fallback_reason": reason}


def _timestamp(frame):
    """Parse actual time, treating naive ISO datetimes as UTC, never as indices."""
    value = frame.get("timestamp")
    try:
        if isinstance(value, Real) and not isinstance(value, bool):
            seconds = float(value)
        else:
            if isinstance(value, str):
                value = datetime.fromisoformat(value.removesuffix("Z") + "+00:00"
                                              if value.endswith("Z") else value)
            if not isinstance(value, datetime):
                raise ValueError("expected ISO datetime or numeric seconds")
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            seconds = value.timestamp()
        if not np.isfinite(seconds):
            raise ValueError("nonfinite time")
        return seconds
    except (ValueError, TypeError, OverflowError, OSError) as exc:
        raise ValueError(f"invalid timestamp for {frame_key(frame)!r}: {value!r}") from exc


def _signal_reason(record, strategy):
    if not isinstance(record, Mapping):
        return "missing_signals"
    n = record.get("n_detected")
    if not isinstance(n, Real) or not np.isfinite(n) or n < 0:
        return "missing_signals"
    if n == 0:
        return "no_detection"
    if strategy == "event_aware" and not record.get("has_propagation", False):
        return "no_propagation"
    components = ("U",) if strategy == "uncertainty" else ("U", "D", "E")
    for component in components:
        value = record.get(component)
        if not isinstance(value, Real) or not np.isfinite(value):
            return "missing_signals"
    return None


def _fixed_ranks(by_plant, signals, strategy):
    """Rank the full supplied evidence universe, NOT current unqueried frames."""
    ranks, scores, reasons = {}, {}, {}
    components = ("U",) if strategy == "uncertainty" else ("U", "D", "E")
    for keys in by_plant.values():
        for key in keys:
            reasons[key] = _signal_reason(signals.get(key), strategy)
        valid = [key for key in keys if reasons[key] is None]
        for key in valid:
            ranks[key] = {}
        for component in components:
            values = [signals[key][component] for key in valid]
            normalized = ((rankdata(values, method="average") - 1) / (len(valid) - 1)
                          if len(valid) > 1 else np.full(len(valid), 0.5))
            for key, value in zip(valid, normalized):
                ranks[key][component] = float(value)
        for key in valid:
            scores[key] = sum(ranks[key].values()) / len(components)
    return ranks, scores, reasons


def select_query_keys(frames, already_selected, target_budget, strategy,
                      signals=None, eta=0.2, seed=42):
    """Return (ordered_NEW_keys, audit_records) without mutating any inputs.

    ``frames`` is the fixed RGB-only metadata universe, including existing
    queries: plant_id, replicate_id, filename, timestamp (ISO datetime or numeric
    seconds); rgb_path is optional and never opened. ``already_selected`` is an
    iterable of keys. ``target_budget`` is an integer TOTAL including them, not
    a batch size. Exhaustion caps the budget; a met budget returns empty lists.

    Strategies: time_coverage, uncertainty, event_aware, random. A plant means
    (plant_id, replicate_id). Every ONE query goes to the available plant with
    the lowest selected/total ratio, with lexicographic plant ties. Thus plant
    sequences and counts are identical across strategies for the same inputs.

    Supply signals as {frame_key: compute_signals_result} over the FULL FIXED
    nonseed candidate universe: omit original seed entries and RETAIN entries
    for candidates acquired in previous calls. Average-tie ranks are normalized
    per plant/component by (rank-1)/(N-1), or 0.5 for a singleton. Uncertainty
    uses rank(U); event_aware uses mean(rank(U), rank(D), rank(E)). Ranks never
    shrink with already_selected. No persistent state or label data is needed.

    Scored selection re-evaluates score - eta * similarity after every query,
    using the same eta for both methods. Similarity is 1 - nearest elapsed time
    / full plant time span, clipped to [0,1], against ALL existing queries,
    including those just acquired. No existing query means zero penalty.
    Candidates with missing/invalid evidence, no detections, or (event_aware)
    no propagation remain eligible with time utility min(1, 2 * nearest elapsed
    time / full plant span): 1 without queries, 0 for a zero span with queries.
    Mixed pools compare these utilities with valid candidates' penalized scores,
    even for a singleton or equal valid ranks. Pure coverage applies when all
    candidates lack evidence, or all have valid but equal scores. Each decision
    logs the CHOSEN candidate's fallback_reason and effective_strategy.

    Coverage maximizes distance to the nearest same-plant query. Without one,
    choose the earliest time; ties use time then key. Utility ties use coverage,
    then time/key. Random uses a local, seeded permutation of the fixed keys.
    Fixed frames/signals/seed yield nested prefixes across budgets AND resumed
    calls. Audit records contain one entry per new key, requested/effective
    strategy, fallback, fixed ranks, score/penalty, time distance and counts.
    """
    strategies = {"time_coverage", "uncertainty", "event_aware", "random"}
    if strategy not in strategies:
        raise ValueError(f"unknown strategy: {strategy!r}")
    if isinstance(target_budget, bool) or not isinstance(target_budget, Integral) or target_budget < 0:
        raise ValueError("target_budget must be a nonnegative integer total")
    if not isinstance(eta, Real) or not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")

    lookup = {}
    for frame in frames:
        key = frame_key(frame)
        if key in lookup:
            raise ValueError(f"duplicate frame key: {key!r}")
        lookup[key] = frame
    selected = {tuple(key) for key in (() if already_selected is None else already_selected)}
    if not selected.issubset(lookup):
        raise ValueError("already_selected contains keys outside the fixed frames universe")
    n_new = min(target_budget, len(lookup)) - len(selected)
    if n_new <= 0:
        return [], []

    by_plant = defaultdict(list)
    for key in sorted(lookup):
        by_plant[key[:2]].append(key)
    times = {key: _timestamp(frame) for key, frame in lookup.items()}
    available, counts, spans, nearest = {}, {}, {}, {}
    for plant, keys in by_plant.items():
        existing = [key for key in keys if key in selected]
        available[plant] = set(keys) - selected
        counts[plant] = len(existing)
        spans[plant] = max(times[k] for k in keys) - min(times[k] for k in keys)
        for key in keys:
            nearest[key] = min((abs(times[key] - times[other]) for other in existing),
                               default=float("inf"))

    scored = strategy in {"uncertainty", "event_aware"}
    ranks, scores, reasons = {}, {}, {}
    if scored:
        ranks, scores, reasons = _fixed_ranks(by_plant, signals or {}, strategy)
    random_order = {}
    if strategy == "random":
        permutation = sorted(lookup)
        random.Random(seed).shuffle(permutation)
        random_order = {key: index for index, key in enumerate(permutation)}

    def coverage_order(key):
        return -nearest[key], times[key], key

    def coverage_utility(key):
        if not np.isfinite(nearest[key]):
            return 1.0
        span = spans[key[:2]]
        return min(1.0, 2 * nearest[key] / span) if span else 0.0

    def penalty(key):
        if not scored or not np.isfinite(nearest[key]):
            return 0.0
        span = spans[key[:2]]
        similarity = max(0.0, 1 - nearest[key] / span) if span else 1.0
        return float(eta * similarity)

    def utility(key):
        return coverage_utility(key) if reasons[key] else scores[key] - penalty(key)

    new_keys, audit = [], []
    for _ in range(n_new):
        plant = min((p for p in by_plant if available[p]),
                    key=lambda p: (Fraction(counts[p], len(by_plant[p])), p))
        candidates = available[plant]
        reason = None
        effective = strategy
        if strategy == "random":
            key = min(candidates, key=random_order.__getitem__)
        elif strategy == "time_coverage":
            key = min(candidates, key=coverage_order)
        else:
            valid = [k for k in candidates if reasons[k] is None]
            # Equal ranks are uninformative only without a mixed alternative.
            scores_equal = (len(valid) == len(candidates)
                            and len({scores[k] for k in valid}) == 1)
            if not valid or scores_equal:
                key = min(candidates, key=coverage_order)
                reason = reasons[key] or "scores_equal"
            else:
                key = min(candidates, key=lambda k: (-utility(k), *coverage_order(k)))
                reason = reasons[key]
            if reason:
                effective = "time_coverage"

        before = counts[plant]
        base = scores.get(key)
        applied_penalty = penalty(key) if effective in {"uncertainty", "event_aware"} else 0.0
        audit.append({
            "frame_key": key, "strategy": strategy, "effective_strategy": effective,
            "fallback_reason": reason, "ranks": dict(ranks.get(key, {})),
            "base_score": base, "diversity_penalty": applied_penalty,
            "adjusted_score": base - applied_penalty if base is not None else None,
            "nearest_time_seconds": nearest[key] if np.isfinite(nearest[key]) else None,
            "selected_before": before, "selected_after": before + 1,
            "plant_total": len(by_plant[plant]),
            "ratio_before": before / len(by_plant[plant]),
            "ratio_after": (before + 1) / len(by_plant[plant]),
            "selected_total_after": len(selected) + len(new_keys) + 1,
        })
        new_keys.append(key)
        available[plant].remove(key)
        counts[plant] += 1
        # Refresh EVERY remaining candidate, rather than walking a stale sort.
        for other in available[plant]:
            nearest[other] = min(nearest[other], abs(times[other] - times[key]))
    return new_keys, audit
