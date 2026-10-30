"""Positive-only pseudo supervision for eager torchvision 0.21 Mask R-CNN.

No modules, parameters, forwards, or torchvision globals are replaced. Human
images use the original model forward; pseudo images reuse its transform,
backbone, RPN/ROI heads, matchers, samplers, box coders and proposal filtering.
"""

from collections import OrderedDict
import math

import torch
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import project_masks_on_boxes
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.ops import box_iou

__all__ = ["partial_maskrcnn_loss", "compute_loss_with_ignore"]


def _instance_validity(value, count, image_size, device, name):
    value = torch.as_tensor(value, device=device)
    if value.ndim == 2 and tuple(value.shape) == tuple(image_size):
        value = value.unsqueeze(0).expand(count, -1, -1)
    elif value.ndim != 3 or tuple(value.shape) != (count, *image_size):
        raise ValueError(f"{name} must have shape [H, W] or [N, H, W]")
    if not ((value == 0) | (value == 1)).all():
        raise ValueError(f"{name} must contain only boolean/0/1 values")
    return value.bool()


def _prepare_pseudo_target(target, image):
    """Copy accepted GT only; never turn rejected instances into background."""
    boxes = target["boxes"]
    if not isinstance(boxes, torch.Tensor) or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must be a tensor of shape [N, 4]")
    count = len(boxes)
    confidence = torch.as_tensor(
        target.get("confidence", torch.ones(count, device=boxes.device)),
        dtype=torch.float32, device=boxes.device,
    ).detach()
    if confidence.shape != (count,) or not torch.isfinite(confidence).all():
        raise ValueError("confidence must contain N finite per-instance weights")
    accepted = confidence > 0
    if not accepted.any():
        return None

    labels, masks = target["labels"], target["masks"]
    image_size = tuple(image.shape[-2:])
    if labels.dtype != torch.int64 or labels.shape != (count,):
        raise ValueError("labels must be int64 with shape [N]")
    if masks.ndim != 3 or tuple(masks.shape) != (count, *image_size):
        raise ValueError("masks must have shape [N, H, W] matching the input image")
    if any(value.device != image.device for value in (boxes, labels, masks)):
        raise ValueError("images, boxes, labels and masks must be on the same device")
    boxes, labels, masks = boxes.detach()[accepted], labels.detach()[accepted], masks.detach()[accepted]
    if not boxes.is_floating_point() or not torch.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError("accepted boxes must be finite floating-point xyxy with positive width/height")
    if (labels <= 0).any():
        raise ValueError("accepted labels must be foreground class indices (> 0)")

    valid = None
    for name in ("valid_mask", "ignore_mask"):
        if name in target:
            current = _instance_validity(target[name], count, image_size, boxes.device, name)[accepted]
            if name == "ignore_mask":
                current = ~current
            valid = current if valid is None else valid & current
    if valid is not None:
        # Ignored sentinels must not bleed into targets through ROIAlign.
        masks = torch.where(valid, masks, torch.zeros_like(masks))
    if not ((masks == 0) | (masks == 1)).all():
        raise ValueError("masks must be binary at valid pixels")
    result = {
        "boxes": boxes, "labels": labels, "masks": masks.to(torch.uint8),
        "confidence": confidence[accepted], "is_pseudo": True,
    }
    if valid is not None:
        result["valid_mask"] = valid
    return result


def _transform_pseudo(transform, images, targets):
    images, transformed = transform(images, targets)
    # GeneralizedRCNNTransform only resizes the built-in masks/boxes fields.
    # Use its actual (possibly randomly chosen) image sizes, not a second resize
    # decision. Nearest interpolation matches torchvision's binary-mask resize.
    for source, target, size in zip(targets, transformed, images.image_sizes):
        if "valid_mask" in source:
            target["valid_mask"] = F.interpolate(
                source["valid_mask"][:, None].float(), size=size, mode="nearest"
            )[:, 0].bool()
        boxes = target["boxes"]
        if not torch.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
            raise ValueError("transformed boxes must be finite with positive width/height")
    return images, transformed


def _positive_matches(candidates, gt_boxes, matcher):
    """Return accepted GT indices, or -1 (ignore), never background labels.

    Retain torchvision RPN low-quality matches when they overlap a GT, but not
    its pathological all-zero-IoU ties (which would promote unrelated anchors).
    """
    if len(candidates) == 0 or len(gt_boxes) == 0:
        return torch.full((len(candidates),), -1, dtype=torch.int64, device=candidates.device)
    quality = box_iou(gt_boxes, candidates)
    matched = matcher(quality)
    return torch.where((matched >= 0) & (quality.max(dim=0).values > 0), matched, -1)


def _weighted_mean(per_sample_loss, confidence):
    # Dividing by sum(q) would cancel confidence scaling, especially for one GT.
    return (per_sample_loss * confidence.to(per_sample_loss)).sum() / max(per_sample_loss.numel(), 1)


def _partial_rpn_loss(rpn, objectness, pred_bbox_deltas, anchors, targets):
    matches = [_positive_matches(a, t["boxes"], rpn.proposal_matcher) for a, t in zip(anchors, targets)]
    labels = [torch.where(m >= 0, 1, -1) for m in matches]
    positive_masks, _ = rpn.fg_bg_sampler(labels)  # No label 0 is ever supplied.
    selected = torch.where(torch.cat(positive_masks).bool())[0]
    if selected.numel() == 0:
        return objectness.sum() * 0, pred_bbox_deltas.sum() * 0

    sampled_anchors, matched_boxes, confidence = [], [], []
    for anchor, target, matched, positive in zip(anchors, targets, matches, positive_masks):
        positive = positive.bool()
        gt_indices = matched[positive]
        sampled_anchors.append(anchor[positive])
        matched_boxes.append(target["boxes"][gt_indices])
        confidence.append(target["confidence"][gt_indices])
    regression_targets = torch.cat(rpn.box_coder.encode(matched_boxes, sampled_anchors))
    confidence = torch.cat(confidence)
    logits = objectness.flatten()[selected]
    objectness_loss = _weighted_mean(
        F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="none"), confidence
    )
    regression_loss = _weighted_mean(
        F.smooth_l1_loss(pred_bbox_deltas[selected], regression_targets, beta=1 / 9, reduction="none").sum(dim=1),
        confidence,
    )
    return objectness_loss, regression_loss


def _select_positive_proposals(roi_heads, proposals, targets):
    gt_boxes = [t["boxes"].to(p.dtype) for t, p in zip(targets, proposals)]
    proposals = roi_heads.add_gt_proposals(proposals, gt_boxes)
    matches = [_positive_matches(p, g, roi_heads.proposal_matcher) for p, g in zip(proposals, gt_boxes)]
    labels = [torch.where(m >= 0, t["labels"][m.clamp(min=0)], -1) for m, t in zip(matches, targets)]
    positive_masks, _ = roi_heads.fg_bg_sampler(labels)
    sampled_proposals, sampled_matches, sampled_labels, matched_boxes, confidence = [], [], [], [], []
    for p, m, label, positive, gt, target in zip(proposals, matches, labels, positive_masks, gt_boxes, targets):
        positive = positive.bool()
        sampled_proposals.append(p[positive])
        sampled_matches.append(m[positive])
        sampled_labels.append(label[positive])
        matched_boxes.append(gt[m[positive]])
        confidence.append(target["confidence"][m[positive]])
    regression_targets = roi_heads.box_coder.encode(matched_boxes, sampled_proposals)
    return sampled_proposals, sampled_matches, sampled_labels, regression_targets, confidence


def _positive_box_loss(class_logits, box_regression, labels, regression_targets, confidence):
    selected = torch.where(labels > 0)[0]
    if selected.numel() == 0:
        return class_logits.sum() * 0, box_regression.sum() * 0
    positive_labels = labels[selected]
    q = confidence[selected]
    classification = _weighted_mean(
        F.cross_entropy(class_logits[selected], positive_labels, reduction="none"), q
    )
    box_regression = box_regression.reshape(class_logits.shape[0], -1, 4)
    regression = _weighted_mean(
        F.smooth_l1_loss(
            box_regression[selected, positive_labels], regression_targets[selected], beta=1 / 9, reduction="none"
        ).sum(dim=1), q,
    )
    return classification, regression


def _weighted_mask_bce(logits, mask_targets, confidence, valid_mask=None):
    """Mean q-weighted instance BCE, each normalized by its valid pixel count.

    Zero target pixels are valid negative *mask pixels*, not missing instances.
    Fully ignored instances contribute zero and confidence never enters a
    denominator. The denominator across instances is the sampled instance count.
    """
    if logits.numel() == 0:
        return logits.sum() * 0
    valid = torch.ones_like(mask_targets, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    mask_targets = torch.where(valid, mask_targets, torch.zeros_like(mask_targets))
    pixel_loss = F.binary_cross_entropy_with_logits(logits, mask_targets.to(logits), reduction="none")
    pixel_loss = torch.where(valid, pixel_loss, torch.zeros_like(pixel_loss))
    valid_count = valid.flatten(1).sum(dim=1).clamp(min=1)
    per_instance = pixel_loss.flatten(1).sum(dim=1) / valid_count
    return _weighted_mean(per_instance, confidence)


def _partial_mask_loss(mask_logits, proposals, targets, matched_indices):
    if mask_logits.shape[0] == 0:
        return mask_logits.sum() * 0
    size = mask_logits.shape[-1]
    mask_targets, valid_masks, labels, confidence = [], [], [], []
    for boxes, target, matched in zip(proposals, targets, matched_indices):
        masks = target["masks"]
        if "valid_mask" in target:
            masks = torch.where(target["valid_mask"], masks, torch.zeros_like(masks))
        projected = project_masks_on_boxes(masks, boxes, matched, size)
        mask_targets.append(projected)
        if "valid_mask" in target:
            # Conservative ROIAlign validity: bins touching ignored source pixels
            # are excluded rather than receiving interpolated false background.
            valid = project_masks_on_boxes(target["valid_mask"].float(), boxes, matched, size) >= 1 - 1e-6
        else:
            valid = torch.ones_like(projected, dtype=torch.bool)
        valid_masks.append(valid)
        labels.append(target["labels"][matched])
        confidence.append(target["confidence"][matched])
    labels = torch.cat(labels)
    logits = mask_logits[torch.arange(len(labels), device=labels.device), labels]
    return _weighted_mask_bce(logits, torch.cat(mask_targets), torch.cat(confidence), torch.cat(valid_masks))


def _pseudo_forward(model, images, targets):
    images, targets = _transform_pseudo(model.transform, images, targets)
    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    # Mirror torchvision 0.21 RPN.forward through proposal generation, but never
    # call its standard assign_targets/compute_loss (which introduce negatives).
    rpn = model.rpn
    feature_list = list(features.values())
    objectness, pred_bbox_deltas = rpn.head(feature_list)
    anchors = rpn.anchor_generator(images, feature_list)
    per_level = [prediction[0].numel() for prediction in objectness]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    proposals = rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors).view(len(anchors), -1, 4)
    proposals, _ = rpn.filter_proposals(proposals, objectness, images.image_sizes, per_level)
    loss_objectness, loss_rpn_box_reg = _partial_rpn_loss(rpn, objectness, pred_bbox_deltas, anchors, targets)

    roi = model.roi_heads
    proposals, matches, labels, regression_targets, confidence = _select_positive_proposals(roi, proposals, targets)
    if not any(len(p) for p in proposals):
        zero = next(iter(features.values())).sum() * 0
        return {
            "loss_classifier": zero, "loss_box_reg": zero, "loss_mask": zero,
            "loss_objectness": loss_objectness, "loss_rpn_box_reg": loss_rpn_box_reg,
        }
    box_features = roi.box_roi_pool(features, proposals, images.image_sizes)
    class_logits, box_regression = roi.box_predictor(roi.box_head(box_features))
    loss_classifier, loss_box_reg = _positive_box_loss(
        class_logits, box_regression, torch.cat(labels), torch.cat(regression_targets), torch.cat(confidence)
    )
    mask_features = roi.mask_roi_pool(features, proposals, images.image_sizes)
    mask_logits = roi.mask_predictor(roi.mask_head(mask_features))
    loss_mask = _partial_mask_loss(mask_logits, proposals, targets, matches)
    return {
        "loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg, "loss_mask": loss_mask,
        "loss_objectness": loss_objectness, "loss_rpn_box_reg": loss_rpn_box_reg,
    }


def compute_loss_with_ignore(model, images, targets, lambda_pseudo=0.5):
    """Standard human loss + lambda_pseudo * positive-only pseudo loss.

    Empty/all-rejected pseudo images are dropped before transform/backbone. An
    entirely skipped batch returns {}; callers must skip backward/optimizer for
    that batch. The human-only case returns model(images, targets) verbatim.
    """
    if len(images) != len(targets):
        raise ValueError("images and targets must have the same length")
    if not math.isfinite(lambda_pseudo) or lambda_pseudo < 0:
        raise ValueError("lambda_pseudo must be finite and nonnegative")
    if not images:
        return {}
    if not model.training:
        raise ValueError("partial supervision requires model.train(); use model(images) for inference")
    pseudo_flags = [bool(target.get("is_pseudo", False)) for target in targets]
    if not any(pseudo_flags):
        return model(images, targets)

    human_images = [image for image, pseudo in zip(images, pseudo_flags) if not pseudo]
    human_targets = [target for target, pseudo in zip(targets, pseudo_flags) if not pseudo]
    losses = model(human_images, human_targets) if human_images else {}
    if lambda_pseudo == 0:
        return losses
    pseudo_images, pseudo_targets = [], []
    for image, target, pseudo in zip(images, targets, pseudo_flags):
        if pseudo:
            prepared = _prepare_pseudo_target(target, image)
            if prepared is not None:
                pseudo_images.append(image)
                pseudo_targets.append(prepared)
    if pseudo_images:
        pseudo_losses = _pseudo_forward(model, pseudo_images, pseudo_targets)
        losses = dict(losses)
        for name, loss in pseudo_losses.items():
            losses[name] = losses.get(name, 0) + lambda_pseudo * loss
    return losses


def partial_maskrcnn_loss(model, images, targets):
    """Unscaled partial-supervision loss (human + pseudo, lambda_pseudo=1).

    Inputs are lists of [C,H,W] floating images and target dictionaries. Missing
    is_pseudo means human. Pseudo targets require boxes [N,4] (float xyxy),
    labels [N] (int64 foreground classes), and binary masks [N,H,W]. Optional
    confidence [N] defaults to one; finite q<=0 rejects the instance. All five
    losses use matched-GT q, normalized by positive sample counts, not sum(q).
    Optional valid_mask [H,W] or [N,H,W] uses 1=valid; legacy ignore_mask has the
    same shapes with 1=ignore. If both are supplied their valid sets intersect.
    Mask BCE includes valid zero pixels, excludes ignored ROIAlign bins, and
    is normalized by valid pixels per sampled instance. Targets are not mutated.

    This is an eager training API for a standard, unwrapped Mask R-CNN; inference
    and checkpoints continue to use the original model without any alterations.
    """
    return compute_loss_with_ignore(model, images, targets, lambda_pseudo=1.0)
