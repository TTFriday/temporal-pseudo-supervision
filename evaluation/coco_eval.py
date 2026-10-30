"""
COCO Evaluation Wrapper for Leaf Instance Segmentation
======================================================
Uses pycocotools to compute COCO mask AP@[0.50:0.95] and related metrics.
Also provides Symmetric Best Dice (SBD) and a separate boundary distance.
"""

import copy
import json

import numpy as np
from PIL import Image

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils


def _encode_mask(mask, image_shape):
    """Encode an original-resolution binary mask without spatial alteration."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.shape != image_shape:
        raise ValueError(f"Mask shape {mask.shape} must match original image shape {image_shape}")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Instance masks must be binary (0/1 or bool)")
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _json_compatible(value):
    """Copy COCO's numpy-backed matching records into JSON-native values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


class LeafCOCOEvaluator:
    """In-memory COCO mask evaluation, with optional reproducible export.

    ``coco_gt``, ``coco_dt``, ``coco_eval``, ``results`` and
    ``per_image_results`` are available after evaluate(). Per-image results
    contain COCO matching/ignore arrays for each area bin, not image AP scores;
    these and the exported masks allow later paired-plant reaggregation.
    """

    METRIC_NAMES = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100")

    def __init__(self, image_size=(531, 525)):
        """
        Args:
            image_size: legacy (width, height) default, retained for compatibility.
                Each sample uses its actual RGB image size, never this fallback.
        """
        self.image_size = image_size
        self.reset()

    def _clear_results(self):
        self.coco_gt = None
        self.coco_dt = None
        self.coco_eval = None
        self.results = None
        self.per_image_results = []

    def reset(self):
        """Clear accumulated samples and cached evaluation/export results."""
        self.predictions = []
        self.ground_truths = []
        self.image_id_counter = 0
        self.ann_id_counter = 0
        self.images_info = []
        self._clear_results()

    def add_sample(self, rgb_path, mask_path, model_predictions=None, ground_truth_instances=None):
        """Add a sample; all masks must already match the original RGB shape.

        Args:
            rgb_path: path to RGB image.
            mask_path: path to the exact-palette GT mask, unused if GT is supplied.
            model_predictions: list of dicts with binary 'mask' (H,W), optional
                'score' and 'bbox' [x,y,w,h]. None/[] means no detections.
            ground_truth_instances: list of dicts with binary 'mask' (H,W).
                If None, extracted from mask_path; [] is a background-only image.
                GT area/bbox are derived from masks, not caller-supplied boxes.

        Returns:
            The assigned image ID. Invalid samples leave existing state unchanged.
        """
        with Image.open(rgb_path) as img:
            w, h = img.size
        img_id = self.image_id_counter + 1
        ann_id = self.ann_id_counter
        if ground_truth_instances is None:
            ground_truth_instances = extract_instances_from_mask(mask_path)
            # Even an all-background mask must have the original image dimensions.
            with Image.open(mask_path) as mask_image:
                if mask_image.size != (w, h):
                    raise ValueError(f"GT mask shape must match original image shape {(h, w)}")

        ground_truths = []
        predictions = []
        for inst in ground_truth_instances:
            rle = _encode_mask(inst["mask"], (h, w))
            ann_id += 1
            ground_truths.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "bbox": mask_utils.toBbox(rle).tolist(),
                "area": int(mask_utils.area(rle)),
                "iscrowd": 0,
            })

        for pred in model_predictions if model_predictions is not None else []:
            rle = _encode_mask(pred["mask"], (h, w))
            ann_id += 1
            predictions.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                # Retain detection boxes for existing callers, but NEVER load
                # these into COCO's segmentation result API (see below).
                "bbox": [float(v) for v in pred.get("bbox", [0, 0, w, h])],
                "score": float(pred.get("score", 1.0)),
            })

        self.images_info.append({
            "id": img_id, "file_name": str(rgb_path), "width": w, "height": h,
        })
        self.ground_truths.extend(ground_truths)
        self.predictions.extend(predictions)
        self.image_id_counter = img_id
        self.ann_id_counter = ann_id
        self._clear_results()
        return img_id

    def _ground_truth_dataset(self):
        return copy.deepcopy({
            "info": {"description": "Leaf Evaluation"},
            "licenses": [],
            "categories": [{"id": 1, "name": "leaf", "supercategory": "plant"}],
            "images": self.images_info,
            "annotations": self.ground_truths,
        })

    def _segmentation_predictions(self):
        # loadRes prefers its bbox branch if bbox exists, assigning box area even
        # for segm evaluation. It also mutates records: always pass fresh copies.
        return copy.deepcopy([
            {key: pred[key] for key in ("image_id", "category_id", "segmentation", "score")}
            for pred in self.predictions
        ])

    def evaluate(self):
        """Run standard COCO segmentation evaluation without any file writes.

        Returns:
            AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100 (fractions in
            [0,1]), plus n_images, n_gt, n_pred. Undefined metrics (including
            area bins with no GT) are None, not COCO's -1 sentinel. No GT adds
            an explicit 'error'; empty predictions with GT yield AP/AR = 0.
        """
        self.coco_gt = COCO()
        self.coco_gt.dataset = self._ground_truth_dataset()
        self.coco_gt.createIndex()
        predictions = self._segmentation_predictions()
        if predictions:
            self.coco_dt = self.coco_gt.loadRes(predictions)
        else:
            # pycocotools.loadRes([]) indexes anns[0], so construct an indexed,
            # valid empty detection dataset with the same images/categories.
            self.coco_dt = COCO()
            self.coco_dt.dataset = self._ground_truth_dataset()
            self.coco_dt.dataset["annotations"] = []
            self.coco_dt.createIndex()

        self.coco_eval = COCOeval(self.coco_gt, self.coco_dt, "segm")
        results = {key: None for key in self.METRIC_NAMES}
        if self.images_info:
            self.coco_eval.evaluate()
            self.coco_eval.accumulate()
            self.coco_eval.summarize()
            results.update({
                key: float(value) if value >= 0 else None
                for key, value in zip(self.METRIC_NAMES, self.coco_eval.stats)
            })
        results.update({
            "n_images": len(self.images_info),
            "n_gt": len(self.ground_truths),
            "n_pred": len(self.predictions),
        })
        if not self.ground_truths:
            results["error"] = "No ground truth instances; COCO metrics are undefined."

        by_image = {
            image["id"]: {
                "image_id": image["id"], "file_name": image["file_name"],
                "n_gt": len(self.coco_gt.imgToAnns.get(image["id"], [])),
                "n_pred": len(self.coco_dt.imgToAnns.get(image["id"], [])),
                "evaluations": [],
            }
            for image in self.images_info
        }
        for record in self.coco_eval.evalImgs:
            if record is not None:
                by_image[record["image_id"]]["evaluations"].append(_json_compatible(record))
        self.per_image_results = list(by_image.values())
        self.results = results
        return dict(results)

    def export_results(self, path=None):
        """Return an independent JSON-ready snapshot; write only if path is given.

        Keys: ground_truth (COCO dataset), predictions (bbox-free segm results),
        results, per_image_results (COCO matches/ignore flags grouped by image),
        parameters (COCO thresholds/area bins/maxDets). Before evaluate(), results
        and parameters are None and per_image_results is empty. RLE counts are
        strings. ``path`` accepts str/Path; its parent must already exist.
        """
        parameters = None
        if self.coco_eval is not None:
            params = self.coco_eval.params
            parameters = {
                key: getattr(params, key)
                for key in ("iouType", "imgIds", "catIds", "useCats", "iouThrs",
                            "recThrs", "areaRng", "areaRngLbl", "maxDets")
            }
        exported = _json_compatible({
            "ground_truth": self._ground_truth_dataset(),
            "predictions": self._segmentation_predictions(),
            "results": self.results,
            "per_image_results": self.per_image_results,
            "parameters": parameters,
        })
        if path is not None:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(exported, handle, allow_nan=False)
        return exported


class LeafMaskEvaluator:
    """Direct mask-based evaluation without COCO format.

    Computes pixel/instance IoU, Symmetric Best Dice and boundary distance.
    """

    def __init__(self):
        self.pixel_ious = []
        self.boundary_dists = []
        self.instance_ious = []

    def reset(self):
        self.pixel_ious = []
        self.boundary_dists = []
        self.instance_ious = []

    def add_sample(self, pred_mask, gt_mask, threshold=0.5):
        """Compare predicted instance mask with ground truth.

        Args:
            pred_mask: numpy array (H, W) with instance IDs (0 = background)
            gt_mask: numpy array (H, W) with instance IDs (0 = background)
            threshold: IoU threshold for matching
        """
        pred_ids = np.unique(pred_mask)
        pred_ids = pred_ids[pred_ids > 0]
        gt_ids = np.unique(gt_mask)
        gt_ids = gt_ids[gt_ids > 0]

        if len(gt_ids) == 0 and len(pred_ids) == 0:
            self.pixel_ious.append(1.0)
            return

        if len(gt_ids) == 0 or len(pred_ids) == 0:
            self.pixel_ious.append(0.0)
            return

        # Pixel-level IoU
        pred_binary = pred_mask > 0
        gt_binary = gt_mask > 0
        intersection = (pred_binary & gt_binary).sum()
        union = (pred_binary | gt_binary).sum()
        pixel_iou = intersection / max(1, union)
        self.pixel_ious.append(pixel_iou)

        # Instance-level matching
        matched_iou = 0
        n_matched = 0
        matched_gt = set()

        for pid in pred_ids:
            pred_inst = (pred_mask == pid)
            best_iou = 0
            best_gid = None

            for gid in gt_ids:
                if gid in matched_gt:
                    continue
                gt_inst = (gt_mask == gid)
                inter = (pred_inst & gt_inst).sum()
                uni = (pred_inst | gt_inst).sum()
                iou = inter / max(1, uni)
                if iou > best_iou:
                    best_iou = iou
                    best_gid = gid

            if best_iou >= threshold and best_gid is not None:
                matched_iou += best_iou
                n_matched += 1
                matched_gt.add(best_gid)

        if n_matched > 0:
            self.instance_ious.append(matched_iou / n_matched)
        else:
            self.instance_ious.append(0.0)

    def compute_sbd(self, pred_mask, gt_mask, max_dist=50):
        """Compute Symmetric Best Dice for (H,W) instance labels (0 background).

        Matches PhenoLeaf-TS's official compute_symmetric_best_dice:
        min(mean bestDice(pred->gt), mean bestDice(gt->pred)), using its 1e-8
        Dice denominator epsilon. Both empty = 1; only one empty = 0.
        ``max_dist`` is accepted but ignored for call compatibility; pixel
        distances are now available through compute_boundary_distance().
        """
        pred_mask = np.asarray(pred_mask)
        gt_mask = np.asarray(gt_mask)
        if pred_mask.ndim != 2 or gt_mask.ndim != 2 or pred_mask.shape != gt_mask.shape:
            raise ValueError("SBD requires two instance-label masks of the same (H,W) shape")
        pred_ids = np.unique(pred_mask)
        pred_ids = pred_ids[pred_ids > 0]
        gt_ids = np.unique(gt_mask)
        gt_ids = gt_ids[gt_ids > 0]
        if len(pred_ids) == 0 and len(gt_ids) == 0:
            return 1.0
        if len(pred_ids) == 0 or len(gt_ids) == 0:
            return 0.0

        pred_instances = [pred_mask == label for label in pred_ids]
        gt_instances = [gt_mask == label for label in gt_ids]
        dice_matrix = np.zeros((len(pred_instances), len(gt_instances)))
        for i, pred in enumerate(pred_instances):
            for j, gt in enumerate(gt_instances):
                intersection = np.logical_and(pred, gt).sum()
                dice_matrix[i, j] = 2.0 * intersection / (pred.sum() + gt.sum() + 1e-8)
        return float(min(dice_matrix.max(axis=1).mean(), dice_matrix.max(axis=0).mean()))

    def compute_boundary_distance(self, pred_mask, gt_mask, max_dist=50):
        """Return the legacy symmetric foreground-boundary distance in pixels.

        For (H,W) binary/instance-label masks, average the two directional mean
        boundary distances, capped at max_dist. Both empty = 0; one empty = cap.
        This ignores internal instance boundaries and is NOT Symmetric Best Dice.
        """
        from scipy import ndimage

        pred_boundary = self._get_boundary(pred_mask > 0)
        gt_boundary = self._get_boundary(gt_mask > 0)

        if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
            return 0.0
        if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
            return float(max_dist)

        # Distance transform from GT boundary
        gt_dist = ndimage.distance_transform_edt(~gt_boundary)
        pred_dist = ndimage.distance_transform_edt(~pred_boundary)

        # Mean distance from pred boundary to gt boundary
        dist_pred_to_gt = gt_dist[pred_boundary].mean()
        dist_gt_to_pred = pred_dist[gt_boundary].mean()

        distance = (dist_pred_to_gt + dist_gt_to_pred) / 2.0
        return float(min(distance, max_dist))

    def _get_boundary(self, binary_mask):
        """Extract boundary pixels from binary mask."""
        from scipy import ndimage
        eroded = ndimage.binary_erosion(binary_mask)
        boundary = binary_mask & ~eroded
        return boundary

    def get_results(self):
        """Return aggregated results."""
        results = {}
        if self.pixel_ious:
            results["pixel_iou_mean"] = float(np.mean(self.pixel_ious))
            results["pixel_iou_median"] = float(np.median(self.pixel_ious))
        if self.instance_ious:
            results["instance_iou_mean"] = float(np.mean(self.instance_ious))
        return results


def extract_instances_from_mask(mask_path, colour_palette=None):
    """Extract all nonempty exact-palette instances without filtering or repair.

    Black (0,0,0) is background. Every other pixel must exactly match a palette
    colour; unknown colours raise ValueError instead of silently dropping pixels.
    Disconnected pixels of one colour form one instance, even a single pixel.
    Image read errors propagate; no raw masks are modified.

    Args:
        mask_path: path to the mask PNG
        colour_palette: list of (R, G, B) tuples. If None, uses PhenoLeaf-TS palette.

    Returns:
        list of dicts with 'mask', 'colour', 'leaf_id', 'bbox', 'area'
    """
    if colour_palette is None:
        colour_palette = PHENOLEAF_PALETTE

    with Image.open(mask_path) as image:
        mask = np.array(image.convert("RGB"))

    instances = []
    known_pixels = np.all(mask == 0, axis=-1)
    for leaf_id, colour in enumerate(colour_palette, start=1):
        if np.all(np.asarray(colour) == 0):
            continue  # Black is always background, even in a custom palette.
        match = np.all(mask == np.array(colour), axis=-1)
        known_pixels |= match
        if not match.any():
            continue

        rows = np.any(match, axis=1)
        cols = np.any(match, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        instances.append({
            "mask": match,
            "colour": colour,
            "leaf_id": leaf_id,
            "bbox": [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)],
            "area": int(match.sum()),
        })

    if not known_pixels.all():
        unknown, counts = np.unique(mask[~known_pixels], axis=0, return_counts=True)
        details = list(zip(unknown[:10].tolist(), counts[:10].tolist()))
        raise ValueError(
            f"Unknown mask colours in {mask_path}: {details} (RGB, pixel count); "
            f"{len(unknown)} unknown colour(s). Expected exact palette colours or black background."
        )
    return instances


# PhenoLeaf-TS colour palette (from prepare.py)
PHENOLEAF_PALETTE = [
    (244, 64, 14),    # Leaf 1
    (48, 57, 249),    # Leaf 2
    (234, 250, 37),   # Leaf 3
    (24, 193, 65),    # Leaf 4
    (245, 130, 49),   # Leaf 5
    (231, 80, 219),   # Leaf 6
    (0, 182, 173),    # Leaf 7
    (115, 0, 218),    # Leaf 8
    (191, 239, 69),   # Leaf 9
    (255, 250, 200),  # Leaf 10
    (250, 190, 212),  # Leaf 11
    (66, 212, 244),   # Leaf 12
    (155, 99, 36),    # Leaf 13
    (220, 190, 255),  # Leaf 14
    (69, 158, 220),   # Leaf 15
    (255, 216, 177),  # Leaf 16
    (98, 2, 37),      # Leaf 17
    (227, 213, 12),   # Leaf 18
    (79, 159, 83),    # Leaf 19
    (170, 23, 101),   # Leaf 20
    (170, 255, 195),  # Leaf 21
    (169, 169, 169),  # Leaf 22
    (181, 111, 119),  # Leaf 23
    (144, 121, 171),  # Leaf 24
    (9, 125, 244),    # Leaf 25
    (184, 70, 30),    # Leaf 26
    (154, 35, 246),   # Leaf 27
    (229, 225, 238),  # Leaf 28
    (141, 254, 82),   # Leaf 29
    (31, 200, 209),   # Leaf 30
    (194, 217, 105),  # Leaf 31
]


def model_predict_to_instances(predictions, image_size, score_threshold=0.5):
    """Convert already-postprocessed, original-resolution Mask R-CNN output.

    Args:
        predictions: tensor dict with 'boxes', 'labels', 'scores', 'masks'.
            Masks must be [N,1,H,W] at the original image size. Let torchvision's
            internal transform/postprocess restore this size; no resizing or
            coordinate guessing is performed here.
        image_size: ORIGINAL (width, height), not the model's internal resize.
        score_threshold: minimum confidence (inclusive).

    Returns:
        list of dicts with 'mask', 'bbox', 'score'
    """
    w, h = image_size
    instances = []

    boxes = predictions["boxes"].detach().cpu().numpy()
    scores = predictions["scores"].detach().cpu().numpy()
    masks = predictions["masks"].detach().cpu().numpy()  # [N, 1, H, W]
    assert masks.ndim == 4 and masks.shape[1:] == (1, h, w), (
        f"Prediction masks must have original image shape [N,1,{h},{w}], got {masks.shape}"
    )
    assert boxes.shape == (len(masks), 4) and scores.shape == (len(masks),), (
        "Prediction boxes, scores and masks must have the same instance count"
    )

    for i in range(len(boxes)):
        if scores[i] < score_threshold:
            continue

        # Binary mask from prediction (threshold at 0.5)
        binary_mask = (masks[i, 0] > 0.5).astype(np.uint8)

        # Get bbox in xyxy format
        x1, y1, x2, y2 = boxes[i]
        bbox_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

        instances.append({
            "mask": binary_mask,
            "bbox": bbox_xywh,
            "score": float(scores[i]),
        })

    return instances
