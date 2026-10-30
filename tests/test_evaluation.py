"""CPU-only regression tests for leaf instance evaluation (no model training)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from evaluation.coco_eval import (
    PHENOLEAF_PALETTE,
    LeafCOCOEvaluator,
    LeafMaskEvaluator,
    extract_instances_from_mask,
    model_predict_to_instances,
)


METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100")
OFFICIAL_METRICS = (
    Path(__file__).resolve().parents[1]
    / "data/phenoleaf_ts/PhenoLeaf-TS-main/phenoleaf_ts/metrics/segmentation.py"
)


def make_image(tmp_path, shape, name="rgb.png"):
    path = tmp_path / name
    Image.fromarray(np.zeros((*shape, 3), dtype=np.uint8)).save(path)
    return path


def instance(mask, **extra):
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {"mask": mask, "bbox": mask_utils.toBbox(rle).tolist(), **extra}


@pytest.mark.parametrize("shape", [(43, 79), (79, 43), (525, 531)])
def test_gt_as_prediction_original_nonsquare_shape(tmp_path, shape, monkeypatch):
    rgb_path = make_image(tmp_path, shape)
    colours = np.zeros((*shape, 3), dtype=np.uint8)
    colours[3:11, 5:16] = PHENOLEAF_PALETTE[0]
    colours[20:25, 25:33] = PHENOLEAF_PALETTE[1]
    mask_path = tmp_path / "mask.png"
    Image.fromarray(colours).save(mask_path)
    before = mask_path.read_bytes()
    gt = extract_instances_from_mask(mask_path)
    model_output = {
        "boxes": torch.tensor([
            [g["bbox"][0], g["bbox"][1],
             g["bbox"][0] + g["bbox"][2], g["bbox"][1] + g["bbox"][3]]
            for g in gt
        ], dtype=torch.float32),
        "scores": torch.tensor([0.99, 0.98]),
        "labels": torch.ones(2, dtype=torch.int64),
        "masks": torch.tensor(np.stack([g["mask"] for g in gt])[:, None]).float(),
    }
    predictions = model_predict_to_instances(model_output, (shape[1], shape[0]))
    evaluator = LeafCOCOEvaluator()  # The default size must not override image size.
    evaluator.add_sample(rgb_path, mask_path, predictions)

    def forbid_file_io(*args, **kwargs):
        raise AssertionError("Evaluation must not open temporary files")

    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", forbid_file_io)
        results = evaluator.evaluate()
        exported = evaluator.export_results()

    for key in ("AP", "AP50", "AP75", "APs", "AR10", "AR100"):
        assert results[key] == pytest.approx(1.0)
    assert results["APm"] is None
    assert results["APl"] is None
    assert {k: results[k] for k in ("n_images", "n_gt", "n_pred")} == {
        "n_images": 1, "n_gt": 2, "n_pred": 2,
    }
    assert exported["ground_truth"]["images"][0]["width"] == shape[1]
    assert exported["ground_truth"]["images"][0]["height"] == shape[0]
    assert mask_path.read_bytes() == before
    for annotation, source in zip(evaluator.ground_truths, gt):
        np.testing.assert_array_equal(mask_utils.decode(annotation["segmentation"]), source["mask"])


def test_empty_predictions_are_valid_coco_results(tmp_path):
    shape = (27, 41)
    mask = np.zeros(shape, dtype=bool)
    mask[1:9, 2:10] = True
    evaluator = LeafCOCOEvaluator()
    evaluator.add_sample(make_image(tmp_path, shape), None, [], [instance(mask)])
    results = evaluator.evaluate()
    for key in ("AP", "AP50", "AP75", "APs", "AR1", "AR10", "AR100"):
        assert results[key] == 0.0
    assert results["APm"] is None and results["APl"] is None
    assert results["n_images"] == results["n_gt"] == 1
    assert results["n_pred"] == 0
    assert evaluator.coco_dt.dataset["annotations"] == []
    assert evaluator.coco_eval is not None
    assert set(METRICS).issubset(results)


@pytest.mark.parametrize("with_predictions", [False, True])
def test_all_area_bins_are_fractions_or_zero_for_empty_predictions(tmp_path, with_predictions):
    shape = (150, 250)
    masks = [np.zeros(shape, dtype=bool) for _ in range(3)]
    masks[0][140:142, 5:7] = True
    masks[1][4:44, 10:50] = True
    masks[2][10:110, 100:210] = True
    gt = [instance(mask) for mask in masks]
    evaluator = LeafCOCOEvaluator()
    evaluator.add_sample(make_image(tmp_path, shape), None, gt if with_predictions else [], gt)
    results = evaluator.evaluate()
    for key in METRICS:
        assert 0.0 <= results[key] <= 1.0
        expected = (1.0 / 3 if key == "AR1" else 1.0) if with_predictions else 0.0
        assert results[key] == pytest.approx(expected)


@pytest.mark.parametrize("with_image,with_prediction", [(False, False), (True, False), (True, True)])
def test_no_ground_truth_is_explicitly_undefined(tmp_path, with_image, with_prediction):
    evaluator = LeafCOCOEvaluator()
    if with_image:
        mask = np.ones((7, 11), dtype=bool)
        evaluator.add_sample(
            make_image(tmp_path, mask.shape), None,
            [instance(mask)] if with_prediction else [], [],
        )
    results = evaluator.evaluate()
    assert all(results[key] is None for key in METRICS)
    assert "undefined" in results["error"].lower()
    assert results["n_gt"] == 0
    assert results["n_images"] == int(with_image)
    assert results["n_pred"] == int(with_prediction)
    json.dumps(evaluator.export_results(), allow_nan=False)


def test_mask_area_not_detection_box_controls_small_bin(tmp_path, monkeypatch):
    shape = (128, 160)
    gt_mask = np.zeros(shape, dtype=bool)
    gt_mask[10:12, 10:12] = True
    false_positive = np.zeros(shape, dtype=bool)
    false_positive[90:92, 90:92] = True
    huge_bbox = [0, 0, shape[1], shape[0]]
    evaluator = LeafCOCOEvaluator()
    evaluator.add_sample(
        make_image(tmp_path, shape), None,
        [instance(false_positive, bbox=huge_bbox, score=0.99),
         instance(gt_mask, bbox=huge_bbox, score=0.9)],
        [instance(gt_mask, bbox=huge_bbox, area=shape[0] * shape[1])],
    )
    original_load_res = COCO.loadRes

    def check_mask_only_load_res(coco, predictions):
        assert predictions and all("bbox" not in p for p in predictions)
        return original_load_res(coco, predictions)

    monkeypatch.setattr(COCO, "loadRes", check_mask_only_load_res)
    results = evaluator.evaluate()
    # A high-score tiny false positive must count in APs, not be ignored as large.
    assert results["APs"] == pytest.approx(0.5)
    assert results["AP"] == pytest.approx(0.5)
    assert evaluator.ground_truths[0]["area"] == 4
    assert all(a["area"] == 4 for a in evaluator.coco_dt.dataset["annotations"])
    assert all(p["bbox"] == huge_bbox and "area" not in p for p in evaluator.predictions)


def test_background_image_false_positive_is_not_dropped(tmp_path):
    mask = np.ones((8, 13), dtype=bool)
    evaluator = LeafCOCOEvaluator()
    evaluator.add_sample(make_image(tmp_path, mask.shape, "positive.png"), None,
                         [instance(mask, score=0.8)], [instance(mask)])
    evaluator.add_sample(make_image(tmp_path, mask.shape, "background.png"), None,
                         [instance(mask, score=0.99)], [])
    results = evaluator.evaluate()
    assert results["AP"] == pytest.approx(0.5)
    assert results["n_images"] == 2 and results["n_gt"] == 1 and results["n_pred"] == 2


def test_export_is_optional_serializable_and_reproducible(tmp_path):
    mask = np.ones((9, 17), dtype=bool)
    evaluator = LeafCOCOEvaluator()
    evaluator.add_sample(make_image(tmp_path, mask.shape), None,
                         [instance(mask)], [instance(mask)])
    first = evaluator.evaluate()
    assert evaluator.evaluate() == first
    exported = evaluator.export_results()
    json.dumps(exported, allow_nan=False)
    assert "bbox" not in exported["predictions"][0]
    assert exported["results"] == first
    assert exported["per_image_results"][0]["image_id"] == 1
    assert exported["per_image_results"][0]["n_gt"] == 1
    records = exported["per_image_results"][0]["evaluations"]
    assert len(records) == 4  # all / small / medium / large COCO area ranges
    assert len(records[0]["dtMatches"]) == 10
    assert len(exported["parameters"]["iouThrs"]) == 10
    exported["ground_truth"]["annotations"].clear()
    assert len(evaluator.ground_truths) == 1
    path = tmp_path / "evaluation.json"
    evaluator.export_results(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == evaluator.export_results()
    replay_gt = COCO()
    replay_gt.dataset = saved["ground_truth"]
    replay_gt.createIndex()
    replay = COCOeval(replay_gt, replay_gt.loadRes(saved["predictions"]), "segm")
    replay.evaluate()
    replay.accumulate()
    replay.summarize()
    assert replay.stats[0] == first["AP"]
    evaluator.reset()
    assert evaluator.results is None
    assert evaluator.per_image_results == []
    assert evaluator.coco_gt is evaluator.coco_dt is evaluator.coco_eval is None
    assert evaluator.predictions == evaluator.ground_truths == evaluator.images_info == []


def test_added_sample_invalidates_previous_results(tmp_path):
    mask = np.ones((8, 13), dtype=bool)
    path = make_image(tmp_path, mask.shape)
    evaluator = LeafCOCOEvaluator()
    evaluator.add_sample(path, None, [instance(mask)], [instance(mask)])
    evaluator.evaluate()
    evaluator.add_sample(path, None, [], [instance(mask)])
    assert evaluator.results is None
    assert evaluator.per_image_results == []
    assert evaluator.export_results()["results"] is None
    assert evaluator.evaluate()["n_images"] == 2


def test_palette_retains_every_nonempty_instance_exactly(tmp_path):
    colours = np.zeros((8, 12, 3), dtype=np.uint8)
    colours[1, 2] = PHENOLEAF_PALETTE[0]  # One-pixel leaf must not be removed.
    colours[3, 4] = colours[6, 9] = PHENOLEAF_PALETTE[-1]
    path = tmp_path / "palette.png"
    Image.fromarray(colours).save(path)
    instances = extract_instances_from_mask(path)
    assert [i["leaf_id"] for i in instances] == [1, len(PHENOLEAF_PALETTE)]
    assert [i["area"] for i in instances] == [1, 2]
    assert instances[0]["bbox"] == [2, 1, 1, 1]
    assert instances[1]["bbox"] == [4, 3, 6, 4]
    restored = np.zeros_like(colours)
    for item in instances:
        restored[item["mask"]] = item["colour"]
    np.testing.assert_array_equal(restored, colours)


@pytest.mark.parametrize("unknown", [(255, 255, 255), (243, 64, 14), (1, 2, 3)])
def test_unknown_colours_fail_strictly(tmp_path, unknown):
    colours = np.zeros((4, 7, 3), dtype=np.uint8)
    colours[1, 1] = PHENOLEAF_PALETTE[0]
    colours[2, 3] = unknown
    path = tmp_path / "unknown.png"
    Image.fromarray(colours).save(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="[Uu]nknown.*colo[u]?r"):
        extract_instances_from_mask(path)
    assert path.read_bytes() == before


def test_custom_palette_and_unreadable_mask(tmp_path):
    colours = np.zeros((3, 4, 3), dtype=np.uint8)
    colours[1, 2] = (1, 2, 3)
    path = tmp_path / "custom.png"
    Image.fromarray(colours).save(path)
    assert extract_instances_from_mask(path, [(1, 2, 3)])[0]["area"] == 1
    with pytest.raises(FileNotFoundError):
        extract_instances_from_mask(tmp_path / "missing.png")


@pytest.mark.parametrize("count", [0, 1])
def test_model_output_must_already_have_original_shape(count):
    predictions = {
        "boxes": torch.zeros((count, 4)),
        "scores": torch.zeros(count),
        "masks": torch.zeros((count, 1, 13, 8)),
    }
    with pytest.raises(AssertionError, match="original"):
        model_predict_to_instances(predictions, (13, 8))


def test_model_empty_predictions_and_threshold():
    predictions = {
        "boxes": torch.tensor([[1.0, 2.0, 4.0, 6.0], [2.0, 3.0, 5.0, 7.0]]),
        "scores": torch.tensor([0.49, 0.5], requires_grad=True),
        "masks": torch.full((2, 1, 8, 13), 0.5, requires_grad=True),
    }
    converted = model_predict_to_instances(predictions, (13, 8))
    assert len(converted) == 1 and converted[0]["score"] == 0.5
    assert converted[0]["bbox"] == [2, 3, 3, 4]
    assert not converted[0]["mask"].any()
    empty = {key: value[:0] for key, value in predictions.items()}
    assert model_predict_to_instances(empty, (13, 8)) == []


@pytest.mark.parametrize("bad_gt", [False, True])
def test_add_sample_rejects_wrong_mask_shape_without_partial_state(tmp_path, bad_gt):
    evaluator = LeafCOCOEvaluator()
    good = instance(np.ones((8, 13), dtype=bool))
    bad = instance(np.ones((13, 8), dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        evaluator.add_sample(make_image(tmp_path, (8, 13)), None,
                             [good] if bad_gt else [bad], [bad] if bad_gt else [good])
    assert evaluator.images_info == evaluator.ground_truths == evaluator.predictions == []
    assert evaluator.image_id_counter == evaluator.ann_id_counter == 0


def sbd_examples():
    gt = np.array([[0, 1, 1, 1, 2, 2], [0, 1, 1, 1, 2, 2]])
    split = np.array([[0, 4, 4, 8, 9, 9], [0, 4, 4, 8, 9, 9]])
    empty = np.zeros_like(gt)
    return [(gt, gt), (split, gt), (gt, split), (gt > 0, gt),
            (empty, gt), (gt, empty), (empty, empty)]


@pytest.mark.parametrize("pred,gt", sbd_examples())
def test_sbd_matches_official_reference(pred, gt):
    if not OFFICIAL_METRICS.is_file():
        pytest.skip("Official PhenoLeaf-TS reference is not installed")
    spec = importlib.util.spec_from_file_location("official_segmentation", OFFICIAL_METRICS)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    pred_instances = [pred == label for label in np.unique(pred) if label > 0]
    gt_instances = [gt == label for label in np.unique(gt) if label > 0]
    expected = official.compute_symmetric_best_dice(pred_instances, gt_instances)
    evaluator = LeafMaskEvaluator()
    assert evaluator.compute_sbd(pred, gt) == pytest.approx(expected, abs=1e-12)
    assert evaluator.compute_sbd(pred, gt, max_dist=0) == pytest.approx(expected, abs=1e-12)


def test_sbd_uses_minimum_not_average_and_keeps_instances():
    pred, gt = sbd_examples()[1]
    evaluator = LeafMaskEvaluator()
    # Directional best-Dice means: (0.8 + 0.5 + 1) / 3 versus (0.8 + 1) / 2.
    assert evaluator.compute_sbd(pred, gt) == pytest.approx(2.3 / 3)
    assert evaluator.compute_sbd(pred, gt) != pytest.approx((2.3 / 3 + 0.9) / 2)
    assert evaluator.compute_boundary_distance(pred, gt) == 0.0
    assert evaluator.compute_boundary_distance(pred, np.zeros_like(gt), max_dist=7) == 7.0


def test_sbd_rejects_different_shapes():
    with pytest.raises(ValueError, match="shape"):
        LeafMaskEvaluator().compute_sbd(np.zeros((8, 13)), np.zeros((13, 8)))
