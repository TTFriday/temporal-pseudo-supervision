"""Synthetic, CPU-only regressions for query selection and prediction evidence."""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
import random
import sys

import numpy as np
import pytest

# Support running this file directly with pytest from outside the repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selection.event_selector import (  # noqa: E402
    compute_signals,
    frame_key,
    mask_iou_matrix,
    match_instances,
    select_query_keys,
)


STRATEGIES = ("time_coverage", "uncertainty", "event_aware", "random")


def instance(mask=None, score=0.95):
    if mask is None:
        mask = np.zeros((8, 11), dtype=bool)
        mask[1:4, 2:6] = True
        mask[4, 2] = True  # Deliberately asymmetric under either flip.
    ys, xs = np.nonzero(mask)
    bbox = [int(xs.min()), int(ys.min()), int(np.ptp(xs) + 1),
            int(np.ptp(ys) + 1)] if len(xs) else [0, 0, 0, 0]
    return {"mask": mask.copy(), "score": score, "bbox": bbox}


def frames_at(hours, plant="A", replicate="r1"):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [
        {"plant_id": plant, "replicate_id": replicate,
         "filename": f"{i:03d}.png", "rgb_path": f"unused/{plant}/{i}.png",
         "timestamp": (start + timedelta(hours=hour)).isoformat()}
        for i, hour in enumerate(hours)
    ]


def signal(u, d=None, e=None, n_detected=1, has_propagation=True):
    return {"U": u, "D": u if d is None else d, "E": u if e is None else e,
            "n_detected": n_detected, "has_propagation": has_propagation,
            "fallback_reason": None}


def test_frame_key_keeps_replicates_distinct():
    first = frames_at([0])[0]
    second = dict(first, replicate_id="r2")
    assert frame_key(first) == ("A", "r1", "000.png")
    assert frame_key(first) != frame_key(second)


def test_mask_iou_ignores_xywh_boxes_and_handles_large_masks():
    a = instance(np.ones((32, 32), dtype=bool))
    b = deepcopy(a)
    b["bbox"] = [999, 999, 1, 1]
    assert mask_iou_matrix([a], [b]).tolist() == [[1.0]]
    assert mask_iou_matrix([], [a]).shape == (0, 1)
    assert mask_iou_matrix([a], []).shape == (1, 0)
    empty = instance(np.zeros((32, 32), dtype=bool))
    assert mask_iou_matrix([empty], [empty]).tolist() == [[0.0]]
    assert match_instances([empty], [empty], 0) == []


def test_duplicate_detections_cannot_share_one_match():
    a = instance()
    assert len(match_instances([a, a], [a], 0.5)) == 1
    scores = compute_signals([a, a], [[a], [a]], [a])
    assert scores["U"] == pytest.approx(0.5)
    assert scores["D"] == pytest.approx(0.5)
    assert 0 <= scores["D"] <= 1
    stable_duplicates = compute_signals([a, a], [[a, a], [a, a]], [a])
    assert stable_duplicates["U"] == 0
    assert stable_duplicates["D"] == stable_duplicates["E"] == 0.5


def test_hungarian_matching_is_not_row_greedy():
    # Greedy row 0 would take column 0 and leave row 1 with no valid match.
    first = np.zeros((1, 10), dtype=bool)
    first[0, :6] = True
    second = np.ones((1, 10), dtype=bool)
    right = np.zeros((1, 10), dtype=bool)
    right[0, :3] = True
    matches = match_instances([instance(first), instance(second)],
                              [instance(second), instance(right)], 0.5)
    assert {(i, j) for i, j, _ in matches} == {(0, 1), (1, 0)}
    assert sorted(iou for _, _, iou in matches) == [0.5, 1.0]


def test_threshold_invalid_edges_do_not_displace_valid_matches(monkeypatch):
    import selection.event_selector as module

    monkeypatch.setattr(module, "mask_iou_matrix",
                        lambda a, b: np.array([[0.99, 0.51], [0.50, 0.49]]))
    assert {(i, j) for i, j, _ in match_instances([{}, {}], [{}, {}], 0.5)} == {
        (0, 1), (1, 0)}


def test_both_inverse_transformed_augmentations_are_stable():
    original = instance()
    augmented = []
    for axis in (1, 0):
        flipped_prediction = np.flip(original["mask"], axis=axis)
        inverse = np.flip(flipped_prediction, axis=axis)
        augmented.append([instance(inverse)])
    scores = compute_signals([original], augmented, [original])
    assert scores == {"U": 0.0, "D": 0.0, "E": 0.0, "n_detected": 1,
                      "has_propagation": True, "fallback_reason": None}
    wrong = compute_signals([original], [[original],
                            [instance(np.flip(original["mask"], axis=0))]], [])
    assert wrong["U"] > 0
    assert wrong["E"] == 0


def test_missing_propagation_differs_from_valid_empty_cache():
    a = instance()
    absent = compute_signals([a], [[a], [a]])
    empty = compute_signals([a], [[a], [a]], [])
    assert absent["has_propagation"] is False
    assert absent["fallback_reason"] == "no_propagation"
    assert (absent["D"], absent["E"]) == (0, 0)
    assert empty["has_propagation"] is True
    assert empty["fallback_reason"] is None
    assert (empty["D"], empty["E"]) == (1, 1)


@pytest.mark.parametrize("weak_part", ["original", "horizontal", "vertical", "missing"])
def test_events_require_high_confidence_stability_in_both_views(weak_part):
    a, h, v = instance(), instance(), instance()
    if weak_part == "original":
        a["score"] = 0.79
    elif weak_part == "horizontal":
        h["score"] = 0.79
    elif weak_part == "vertical":
        v["score"] = 0.79
    scores = compute_signals([a], [[h], [] if weak_part == "missing" else [v]], [])
    assert scores["E"] == 0


def test_event_is_fraction_of_stable_high_confidence_uncovered_detections():
    a, low = instance(score=0.8), instance(score=0.1)
    low["mask"] = np.roll(low["mask"], 5, axis=1)
    predictions = [a, low]
    assert compute_signals(predictions, [predictions, predictions], [])["E"] == 0.5
    assert compute_signals(predictions, [predictions, predictions], [a])["E"] == 0


@pytest.mark.parametrize("propagated", [None, [], [instance()]])
def test_no_detection_is_explicit_and_divergence_bounded(propagated):
    scores = compute_signals([], [[], []], propagated)
    assert scores["fallback_reason"] == "no_detection"
    assert scores["n_detected"] == 0
    assert scores["E"] == 0
    assert scores["D"] == (1 if propagated else 0)


def test_augmentation_extra_instances_and_reordered_predictions():
    a = instance()
    b = instance(np.roll(a["mask"], 5, axis=1))
    scores = compute_signals([a], [[a, b], [a, b]], [a, b])
    assert scores["U"] == scores["D"] == 0.5
    assert compute_signals([a, b], [[b, a], [b, a]], [b, a])["U"] == 0


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_ratios_are_updated_after_every_query_across_replicates(strategy):
    frames = (frames_at(range(5), "A", "r1") + frames_at(range(9), "A", "r2")
              + frames_at(range(13), "B", "r1") + frames_at([0], "C", "r1"))
    seeds = {frame_key(frames[2]), frame_key(frames[8])}
    signals = {frame_key(f): signal((i * 13 % 29) / 29)
               for i, f in enumerate(frames) if frame_key(f) not in seeds}
    selected, audit = select_query_keys(frames, seeds, 22, strategy, signals)
    totals = Counter(frame_key(f)[:2] for f in frames)
    counts = Counter(k[:2] for k in seeds)
    assert len(selected) == len(audit) == 20
    assert not set(selected) & seeds
    assert len(set(selected)) == len(selected)
    for key, record in zip(selected, audit):
        available = [p for p in totals if counts[p] < totals[p]]
        expected = min(available, key=lambda p: (Fraction(counts[p], totals[p]), p))
        assert key[:2] == expected
        assert record["frame_key"] == key
        assert record["selected_before"] == counts[expected]
        assert record["ratio_before"] == counts[expected] / totals[expected]
        counts[expected] += 1
        assert record["selected_after"] == counts[expected]
    baseline, _ = select_query_keys(frames, seeds, 22, "time_coverage")
    assert [k[:2] for k in selected] == [k[:2] for k in baseline]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("partial_evidence", [False, True])
def test_five_and_twenty_percent_budgets_are_nested_even_incrementally(strategy, partial_evidence):
    frames = frames_at(range(100), "A") + frames_at(range(100), "B")
    seeds = {frame_key(frames[50]), frame_key(frames[150])}
    signals = {frame_key(f): signal((i * 13 % 31) / 31, (i * 7 % 23) / 23,
                                  (i * 11 % 19) / 19)
               for i, f in enumerate(frames) if frame_key(f) not in seeds}
    if partial_evidence:
        for i, key in enumerate(list(signals)):
            if i % 4 == 0:
                signals.pop(key)
            elif i % 4 == 1:
                signals[key]["n_detected"] = 0
            elif i % 4 == 2:
                signals[key]["has_propagation"] = False
    small, small_audit = select_query_keys(frames, seeds, 10, strategy, signals, seed=8)
    large, large_audit = select_query_keys(frames, seeds, 40, strategy, signals, seed=8)
    continuation, continuation_audit = select_query_keys(frames, seeds | set(small), 40,
                                                         strategy, signals, seed=8)
    assert large[:len(small)] == small
    assert small + continuation == large
    assert small_audit + continuation_audit == large_audit
    assert len(large) + len(seeds) == 40
    coverage, _ = select_query_keys(frames, seeds, 40, "time_coverage")
    assert [key[:2] for key in large] == [key[:2] for key in coverage]
    selected = set(seeds)
    one_at_a_time, incremental_audit = [], []
    for budget in range(len(seeds) + 1, 41):
        new, records = select_query_keys(frames, selected, budget, strategy, signals, seed=8)
        one_at_a_time.extend(new)
        incremental_audit.extend(records)
        selected.update(new)
    assert one_at_a_time == large
    assert incremental_audit == large_audit


def test_coverage_uses_real_elapsed_time_and_existing_queries():
    frames = frames_at([0, 1, 2, 3, 100])
    seeds = {frame_key(frames[0])}
    keys, _ = select_query_keys(frames, seeds, 3, "time_coverage")
    assert keys == [frame_key(frames[4]), frame_key(frames[3])]
    reordered = list(reversed(frames))
    assert select_query_keys(reordered, seeds, 3, "time_coverage")[0] == keys


def test_timestamps_are_parsed_not_sorted_lexically():
    frames = frames_at([0, 1, 2])
    frames[2]["timestamp"] = "2024-12-31T22:00:00-04:00"  # Jan 1, 02:00 UTC.
    keys, _ = select_query_keys(frames, {frame_key(frames[0])}, 2, "time_coverage")
    assert keys == [frame_key(frames[2])]
    unseeded, _ = select_query_keys(frames, set(), 1, "time_coverage")
    assert unseeded == [frame_key(frames[0])]


@pytest.mark.parametrize("strategy", ["uncertainty", "event_aware"])
@pytest.mark.parametrize("reason", ["missing_signals", "no_detection", "scores_equal"])
def test_no_evidence_falls_back_to_coverage_with_audit(strategy, reason):
    frames = frames_at([0, 1, 2, 3, 100])
    seeds = {frame_key(frames[0])}
    signals = None if reason == "missing_signals" else {
        frame_key(f): signal(0.5, n_detected=0 if reason == "no_detection" else 1)
        for f in frames[1:]}
    expected, _ = select_query_keys(frames, seeds, 3, "time_coverage")
    actual, audit = select_query_keys(frames, seeds, 3, strategy, signals)
    assert actual == expected
    assert all(r["fallback_reason"] == reason for r in audit)
    assert all(r["effective_strategy"] == "time_coverage" for r in audit)


def test_absent_propagation_falls_back_only_for_event_strategy():
    frames = frames_at([0, 1, 100])
    seeds = {frame_key(frames[0])}
    signals = {frame_key(frames[1]): signal(1, has_propagation=False),
               frame_key(frames[2]): signal(0, has_propagation=False)}
    event, audit = select_query_keys(frames, seeds, 2, "event_aware", signals)
    assert event == [frame_key(frames[2])]
    assert audit[0]["fallback_reason"] == "no_propagation"
    uncertainty, audit = select_query_keys(frames, seeds, 2, "uncertainty", signals)
    assert uncertainty == [frame_key(frames[1])]
    assert audit[0]["fallback_reason"] is None
    for record in signals.values():
        record["has_propagation"] = True  # An empty cache is still evidence.
    event, audit = select_query_keys(frames, seeds, 2, "event_aware", signals)
    assert event == uncertainty
    assert audit[0]["fallback_reason"] is None


@pytest.mark.parametrize("strategy", ["uncertainty", "event_aware"])
def test_partial_missing_evidence_remains_eligible_and_logged(strategy):
    frames = frames_at([0, 1, 100])
    signals = {frame_key(frames[1]): signal(1)}
    keys, audit = select_query_keys(frames, {frame_key(frames[0])}, 2,
                                    strategy, signals)
    assert keys == [frame_key(frames[2])]
    assert audit[0]["fallback_reason"] == "missing_signals"
    assert audit[0]["effective_strategy"] == "time_coverage"


def test_limited_seed_propagation_window_preserves_event_evidence():
    # One seed covers only 48h of a 240h sequence. After two coverage queries,
    # the event at 24h must beat the many frames without SAM2 evidence.
    hours = list(range(0, 241, 12))
    frames = frames_at(hours)
    keys = [frame_key(f) for f in frames]
    seeds = {keys[0]}
    signals = {key: signal(0, d=int(hour == 24), e=int(hour == 24),
                           has_propagation=hour <= 48)
               for key, hour in zip(keys[1:], hours[1:])}
    selected, audit = select_query_keys(frames, seeds, 4, "event_aware", signals)
    assert selected == [keys[-1], keys[10], keys[2]]  # 240h, 120h, event at 24h.
    event = audit[-1]
    assert event["ranks"] == {"U": 0.5, "D": 1.0, "E": 1.0}
    assert event["adjusted_score"] == pytest.approx(5 / 6 - 0.2 * 0.9)
    assert event["adjusted_score"] > 2 * 60 / 240  # Best remaining coverage utility.
    assert [r["fallback_reason"] for r in audit] == [
        "no_propagation", "no_propagation", None]
    assert [r["effective_strategy"] for r in audit] == [
        "time_coverage", "time_coverage", "event_aware"]
    coverage, _ = select_query_keys(frames, seeds, 4, "time_coverage")
    missing, missing_audit = select_query_keys(frames, seeds, 4, "event_aware")
    assert missing == coverage == [keys[-1], keys[10], keys[5]]
    assert all(r["fallback_reason"] == "missing_signals" for r in missing_audit)
    assert all(r["effective_strategy"] == "time_coverage" for r in missing_audit)


@pytest.mark.parametrize("strategy,reason", [
    ("uncertainty", "missing_signals"), ("event_aware", "missing_signals"),
    ("uncertainty", "no_detection"), ("event_aware", "no_detection"),
    ("event_aware", "no_propagation"),
])
@pytest.mark.parametrize("missing_hour", [10, 50])
def test_mixed_utilities_compare_each_candidate_and_log_its_reason(
        strategy, reason, missing_hour):
    frames = frames_at([0, missing_hour, 20, 30, 100])
    keys = [frame_key(f) for f in frames]
    signals = {keys[2]: signal(1), keys[3]: signal(0)}
    if reason != "missing_signals":
        signals[keys[1]] = signal(1, n_detected=0 if reason == "no_detection" else 1,
                                 has_propagation=reason != "no_propagation")
    selected, audit = select_query_keys(frames, {keys[0], keys[4]}, 3,
                                       strategy, signals)
    # Evidence utility is 1 - 0.2 * 0.8 = 0.84; fallback is 0.2 or 1.0.
    fallback_wins = missing_hour == 50
    assert selected == [keys[1] if fallback_wins else keys[2]]
    assert audit[0]["fallback_reason"] == (reason if fallback_wins else None)
    assert audit[0]["effective_strategy"] == ("time_coverage" if fallback_wins else strategy)
    assert audit[0]["diversity_penalty"] == pytest.approx(0 if fallback_wins else 0.16)
    if not fallback_wins:
        assert audit[0]["adjusted_score"] == pytest.approx(0.84)


def test_all_invalid_candidates_log_the_chosen_reason_not_plant_precedence():
    frames = frames_at([0, 1, 50, 100])
    keys = [frame_key(f) for f in frames]
    signals = {keys[2]: signal(0, n_detected=0),
               keys[3]: signal(0, has_propagation=False)}
    selected, audit = select_query_keys(frames, {keys[0]}, 4, "event_aware", signals)
    assert selected == [keys[3], keys[2], keys[1]]
    assert [r["fallback_reason"] for r in audit] == [
        "no_propagation", "no_detection", "missing_signals"]
    assert all(r["effective_strategy"] == "time_coverage" for r in audit)


@pytest.mark.parametrize("strategy", ["uncertainty", "event_aware"])
def test_equal_valid_ranks_still_compete_with_missing_coverage(strategy):
    frames = frames_at([0, 25, 30, 40, 100])
    keys = [frame_key(f) for f in frames]
    signals = {keys[2]: signal(0.5), keys[3]: signal(0.5)}
    selected, audit = select_query_keys(frames, {keys[0], keys[4]}, 3, strategy, signals)
    # Mixed utilities distinguish the missing 25h frame (0.5) from both valid
    # frames (0.36, 0.38), despite equal ranks. Pure coverage would choose 40h.
    assert selected == [keys[1]]
    assert audit[0]["fallback_reason"] == "missing_signals"
    assert audit[0]["effective_strategy"] == "time_coverage"


@pytest.mark.parametrize("strategy", ["uncertainty", "event_aware"])
@pytest.mark.parametrize("hours", [[0, 1, 2], [0, 0, 0]])
@pytest.mark.parametrize("missing_index", [0, 1])
def test_unseeded_mixed_utilities_are_one_and_tie_by_time_then_key(
        strategy, hours, missing_index):
    frames = frames_at(hours)
    keys = [frame_key(f) for f in frames]
    signals = {keys[1 - missing_index]: signal(1), keys[2]: signal(0)}
    selected, audit = select_query_keys(list(reversed(frames)), set(), 1, strategy, signals)
    # The missing and highest-ranked candidates both have utility 1.0.
    assert selected == [keys[0]]
    assert audit[0]["fallback_reason"] == ("missing_signals" if missing_index == 0 else None)
    assert audit[0]["effective_strategy"] == ("time_coverage" if missing_index == 0 else strategy)
    assert audit[0]["diversity_penalty"] == 0


@pytest.mark.parametrize("strategy", ["uncertainty", "event_aware"])
def test_zero_span_mixed_coverage_is_zero_with_an_existing_query(strategy):
    frames = frames_at([0, 0, 0, 0])
    keys = [frame_key(f) for f in frames]
    signals = {keys[2]: signal(1), keys[3]: signal(0)}
    selected, audit = select_query_keys(frames, {keys[0]}, 2, strategy, signals)
    assert selected == [keys[2]]  # Evidence utility 0.8 beats missing utility 0.
    assert audit[0]["adjusted_score"] == pytest.approx(0.8)
    assert audit[0]["fallback_reason"] is None
    assert audit[0]["effective_strategy"] == strategy


@pytest.mark.parametrize("missing_hour,evidence_hour", [(80, 100), (100, 80)])
def test_mixed_coverage_utility_is_capped_and_ties_use_nearest_distance(
        missing_hour, evidence_hour):
    frames = frames_at([0, missing_hour, evidence_hour, 10])
    keys = [frame_key(f) for f in frames]
    signals = {keys[2]: signal(1), keys[3]: signal(0)}
    selected, audit = select_query_keys(frames, {keys[0]}, 2, "event_aware", signals, eta=0)
    # Both utilities are 1.0, not an uncapped 1.6/2.0 for the missing frame.
    assert selected == [keys[1] if missing_hour == 100 else keys[2]]
    assert audit[0]["fallback_reason"] == ("missing_signals" if missing_hour == 100 else None)


def test_rank_normalization_is_per_component_per_plant_with_average_ties():
    frames = frames_at([0, 1, 2, 3]) + frames_at([0, 1, 2, 3], "B")
    u, d, e = [0, 1, 1, 1000], [3, 1, 2, 0], [7, 7, 7, 7]
    signals = {frame_key(f): signal(u[i % 4] * (100 if i >= 4 else 1),
                                  d[i % 4], e[i % 4]) for i, f in enumerate(frames)}
    _, audit = select_query_keys(frames, set(), len(frames), "event_aware", signals, eta=0)
    expected_u = [0, 0.5, 0.5, 1]
    expected_d = [1, 1 / 3, 2 / 3, 0]
    for record in audit:
        i = int(record["frame_key"][2][:3])
        assert record["ranks"] == pytest.approx({"U": expected_u[i],
                                                 "D": expected_d[i], "E": 0.5})
        assert record["base_score"] == pytest.approx(
            (expected_u[i] + expected_d[i] + 0.5) / 3)


@pytest.mark.parametrize("strategy", ["uncertainty", "event_aware"])
def test_resumed_ranks_retain_queried_candidates_but_exclude_unscored_seeds(strategy):
    frames = frames_at([0, 1, 2, 3, 4])
    seeds = {frame_key(frames[0])}
    signals = {frame_key(f): signal(v)
               for f, v in zip(frames[1:], [0, 0.01, 0.02, 0.99])}
    first, first_audit = select_query_keys(frames, seeds, 2, strategy, signals, eta=0)
    assert first == [frame_key(frames[4])]
    assert first_audit[0]["ranks"]["U"] == 1
    second, second_audit = select_query_keys(frames, seeds | set(first), 3,
                                            strategy, signals, eta=0)
    assert second == [frame_key(frames[3])]
    # Re-ranking only the remaining candidates would incorrectly give 1.0.
    assert second_audit[0]["ranks"]["U"] == pytest.approx(2 / 3)
    assert second_audit[0]["fallback_reason"] is None


@pytest.mark.parametrize("strategy", ["uncertainty", "event_aware"])
def test_nonfinite_evidence_does_not_force_valid_winner_to_fallback(strategy):
    frames = frames_at([0, 1, 100])
    signals = {frame_key(frames[1]): signal(float("nan")),
               frame_key(frames[2]): signal(0)}
    keys, audit = select_query_keys(frames, {frame_key(frames[0])}, 2, strategy, signals)
    assert keys == [frame_key(frames[2])]
    # The chosen singleton's valid score (0.5) beats missing utility (0.02).
    # Do not attribute the other candidate's nonfinite evidence to this one.
    assert audit[0]["fallback_reason"] is None
    assert audit[0]["effective_strategy"] == strategy
    assert audit[0]["adjusted_score"] == 0.5


@pytest.mark.parametrize("strategy", ["uncertainty", "event_aware"])
def test_diversity_is_truly_greedy_with_the_same_eta(strategy):
    # Rank gap between the top two is 0.1, smaller than the diversity penalty.
    # After selecting hour 0, hour 100 must overtake hour 1.
    frames = frames_at([0, 1, 100, 10, 20, 30, 40, 50, 60, 70, 80])
    values = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    signals = {frame_key(f): signal(v) for f, v in zip(frames, values)}
    plain, _ = select_query_keys(frames, set(), 2, strategy, signals, eta=0)
    greedy, audit = select_query_keys(frames, set(), 2, strategy, signals, eta=0.2)
    assert plain == [frame_key(frames[0]), frame_key(frames[1])]
    assert greedy == [frame_key(frames[0]), frame_key(frames[2])]
    assert audit[1]["base_score"] == pytest.approx(0.8)
    assert audit[1]["diversity_penalty"] == 0
    continuation, _ = select_query_keys(frames, {greedy[0]}, 2, strategy, signals)
    assert continuation == greedy[1:]


def test_random_is_reproducible_order_invariant_and_does_not_touch_global_rng():
    frames = frames_at(range(20))
    state = random.getstate()
    first, _ = select_query_keys(frames, set(), 10, "random", seed=17)
    repeated, _ = select_query_keys(list(reversed(frames)), set(), 10, "random", seed=17)
    different, _ = select_query_keys(frames, set(), 10, "random", seed=18)
    assert first == repeated
    assert first != different
    assert random.getstate() == state


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_budget_is_total_and_caps_at_available_frames(strategy):
    frames = frames_at([0, 1, 2])
    seeds = {frame_key(frames[0])}
    for budget in (0, 1):
        assert select_query_keys(frames, seeds, budget, strategy) == ([], [])
    keys, audit = select_query_keys(frames, seeds, 99, strategy)
    assert len(keys) == len(audit) == 2
    assert set(keys) | seeds == {frame_key(f) for f in frames}
    assert select_query_keys([], set(), 3, strategy) == ([], [])


def test_inputs_are_not_mutated_or_label_files_opened():
    frames = frames_at([0, 1, 2])
    signals = {frame_key(f): signal(i) for i, f in enumerate(frames)}
    seeds = {frame_key(frames[0])}
    before = deepcopy((frames, signals, seeds))
    select_query_keys(frames, seeds, 3, "event_aware", signals)
    assert (frames, signals, seeds) == before
    a = instance()
    mask = a["mask"].copy()
    compute_signals([a], [[a], [a]], [])
    np.testing.assert_array_equal(a["mask"], mask)


@pytest.mark.parametrize("bad_timestamp", [None, "not-a-time", float("nan")])
def test_invalid_timestamps_do_not_silently_become_frame_indices(bad_timestamp):
    frames = frames_at([0, 1])
    frames[0]["timestamp"] = bad_timestamp
    with pytest.raises(ValueError, match="timestamp"):
        select_query_keys(frames, set(), 1, "time_coverage")


@pytest.mark.parametrize("kwargs", [{"strategy": "unknown"}, {"target_budget": -1},
                                     {"target_budget": 0.5}, {"eta": -1},
                                     {"eta": float("nan")}])
def test_invalid_selection_arguments(kwargs):
    args = dict(frames=frames_at([0, 1]), already_selected=set(),
                target_budget=1, strategy="time_coverage")
    args.update(kwargs)
    with pytest.raises(ValueError):
        select_query_keys(**args)


def test_duplicate_or_unknown_keys_are_rejected():
    frames = frames_at([0, 1])
    with pytest.raises(ValueError, match="duplicate"):
        select_query_keys(frames + [frames[0]], set(), 1, "random")
    with pytest.raises(ValueError, match="already_selected"):
        select_query_keys(frames, {("unknown", "r1", "000.png")}, 2, "random")


def test_signal_inputs_require_two_views_and_consistent_boolean_masks():
    a = instance()
    with pytest.raises(ValueError, match="two"):
        compute_signals([a], [[a]])
    with pytest.raises(ValueError, match="shape"):
        mask_iou_matrix([a], [instance(np.ones((1, 1), dtype=bool))])
    with pytest.raises(ValueError, match="boolean"):
        mask_iou_matrix([dict(a, mask=a["mask"].astype(float))], [a])
    with pytest.raises(ValueError, match="threshold"):
        match_instances([a], [a], 1.1)
