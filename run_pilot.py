"""
Pilot Experiment Runner for Low-Label Temporal Leaf Instance Segmentation
=========================================================================
Runs 6 methods x 2 seeds = 12 student endpoint trainings on a subsampled
40 train / 12 val plant configuration at 10% annotation budget.

Methods:
  M1: Time-coverage selection + pure human-label supervision (no pseudo)
  M2: Uncertainty selection + spatial self-training (no temporal, no SAM2)
  M3: Time-coverage selection + basic propagation (confidence only)
  M4: Event-aware selection + reliable propagation (IoU>=0.7, conf>=0.8, bidirectional)
  M5: Time-coverage selection + reliable propagation (ablation: no event-aware)
  M6: Event-aware selection + basic propagation (ablation: no reliability)

Usage:
  python run_pilot.py [--output_dir pilot_results] [--seeds 17 42]
"""

import os
import sys
import json
import math
import random
import argparse
import collections
import time
import copy
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
from PIL import Image
import torch
import torch.utils.data
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "data_audit" / "manifest.json"
SPLIT_PATH = PROJECT_ROOT / "splits" / "plant_split.json"
PALETTE_SOURCE = "data/phenoleaf_ts/PhenoLeaf-TS-main/phenoleaf_ts/data/prepare.py"

# PhenoLeaf-TS colour palette (31 leaves, from prepare.py)
COLOUR_PALETTE = [
    (244, 64, 14),    (48, 57, 249),    (234, 250, 37),   (24, 193, 65),
    (245, 130, 49),   (231, 80, 219),   (0, 182, 173),    (115, 0, 218),
    (191, 239, 69),   (255, 250, 200),  (250, 190, 212),  (66, 212, 244),
    (155, 99, 36),    (220, 190, 255),  (69, 158, 220),   (255, 216, 177),
    (98, 2, 37),      (227, 213, 12),   (79, 159, 83),    (170, 23, 101),
    (170, 255, 195),  (169, 169, 169),  (181, 111, 119),  (144, 121, 171),
    (9, 125, 244),    (184, 70, 30),    (154, 35, 246),   (229, 225, 238),
    (141, 254, 82),   (31, 200, 209),   (194, 217, 105),
]

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_manifest(path):
    """Load manifest and build efficient lookups."""
    with open(path) as f:
        manifest = json.load(f)

    frames_by_plant = collections.defaultdict(list)
    for entry in manifest:
        key = (entry["plant_id"], entry["replicate_id"])
        frames_by_plant[key].append(entry)

    # Sort by timestamp
    for key in frames_by_plant:
        frames_by_plant[key].sort(key=lambda x: x.get("timestamp", ""))

    return manifest, frames_by_plant


def load_split(path):
    """Load plant-level split."""
    with open(path) as f:
        split = json.load(f)
    return split


def subsample_plants(plants_list, n_target, frames_by_plant, seed=42):
    """Stratified subsample of plants by genotype.

    When n_target < n_genotypes, first select genotypes proportional to their
    representation, then pick 1 plant from each selected genotype.

    Args:
        plants_list: list of [plant_id, rep_id] pairs
        n_target: number of plants to select
        frames_by_plant: lookup for frame counts
        seed: random seed

    Returns:
        list of selected [plant_id, rep_id] pairs
    """
    rng = random.Random(seed)

    # Group by genotype
    by_genotype = collections.defaultdict(list)
    for p in plants_list:
        by_genotype[p[0]].append(p)

    genotypes = sorted(by_genotype.keys())
    n_genotypes = len(genotypes)

    selected = []

    if n_target >= n_genotypes:
        # Normal case: at least 1 per genotype, then allocate remainder proportionally
        allocations = {gt: 1 for gt in genotypes}
        remaining = n_target - n_genotypes

        # Distribute remaining proportionally
        if remaining > 0:
            total = len(plants_list)
            extra_fracs = []
            for gt in genotypes:
                frac = len(by_genotype[gt]) / total
                extra = frac * remaining
                extra_fracs.append((gt, extra))

            # Give integer parts first
            for gt, extra in extra_fracs:
                n_extra = int(extra)
                allocations[gt] += n_extra
                remaining -= n_extra

            # Distribute leftover by largest fractional remainder
            extras_sorted = sorted(extra_fracs, key=lambda x: -(x[1] - int(x[1])))
            for gt, extra in extras_sorted:
                if remaining <= 0:
                    break
                if allocations[gt] < len(by_genotype[gt]):
                    allocations[gt] += 1
                    remaining -= 1

        # Adjust to hit exact target
        while sum(allocations.values()) > n_target:
            gt_to_reduce = max(
                [g for g in genotypes if allocations[g] > 1],
                key=lambda g: allocations[g] / max(1, len(by_genotype[g])),
                default=None
            )
            if gt_to_reduce is None:
                break
            allocations[gt_to_reduce] -= 1

        while sum(allocations.values()) < n_target:
            gt_to_add = min(
                [g for g in genotypes if allocations[g] < len(by_genotype[g])],
                key=lambda g: allocations[g] / max(1, len(by_genotype[g])),
                default=None
            )
            if gt_to_add is None:
                break
            allocations[gt_to_add] += 1

        # Sample from each genotype
        for gt in genotypes:
            n_select = min(allocations[gt], len(by_genotype[gt]))
            chosen = rng.sample(by_genotype[gt], n_select)
            selected.extend(chosen)
    else:
        # n_target < n_genotypes: select genotypes first, then 1 plant each
        # Weight genotypes by number of available plants
        weights = [len(by_genotype[gt]) for gt in genotypes]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        # Select genotypes
        selected_genotypes = []
        available = list(range(n_genotypes))
        for _ in range(n_target):
            if not available:
                break
            avail_probs = [probs[i] for i in available]
            total_p = sum(avail_probs)
            avail_probs = [p / total_p for p in avail_probs]
            r = rng.random()
            cumsum = 0
            chosen_idx = available[-1]
            for j, idx in enumerate(available):
                cumsum += avail_probs[j]
                if r <= cumsum:
                    chosen_idx = idx
                    break
            selected_genotypes.append(genotypes[chosen_idx])
            available.remove(chosen_idx)

        for gt in selected_genotypes:
            chosen = rng.choice(by_genotype[gt])
            selected.append(chosen)

    return selected


# ---------------------------------------------------------------------------
# Mask / instance helpers
# ---------------------------------------------------------------------------

def extract_instances_from_mask(mask_path):
    """Extract leaf instances from colour-coded mask.

    Returns list of dicts: mask (H,W bool), bbox [x,y,w,h], area, leaf_id
    """
    try:
        mask_img = np.array(Image.open(mask_path).convert("RGB"))
    except Exception as e:
        print(f"Warning: Could not read mask {mask_path}: {e}")
        return []

    h, w = mask_img.shape[:2]
    instances = []

    for leaf_id, colour in enumerate(COLOUR_PALETTE, start=1):
        colour_arr = np.array(colour, dtype=np.uint8)
        match = np.all(mask_img == colour_arr, axis=-1)
        if match.sum() < 10:
            continue

        rows = np.any(match, axis=1)
        cols = np.any(match, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        instances.append({
            "mask": match,
            "bbox": [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)],
            "area": int(match.sum()),
            "leaf_id": leaf_id,
        })

    return instances


def instances_to_target(instances, image_size, is_pseudo=False, confidences=None):
    """Convert instance list to training target dict.

    Args:
        instances: list of dicts with 'mask', 'bbox'
        image_size: (width, height)
        is_pseudo: whether pseudo labels
        confidences: per-instance confidence weights

    Returns:
        target dict for Mask R-CNN training
    """
    w, h = image_size
    boxes = []
    labels = []
    masks = []
    confs = []
    areas = []

    for i, inst in enumerate(instances):
        bx, by, bw, bh = inst["bbox"]
        if bw <= 0 or bh <= 0:
            continue

        boxes.append([bx, by, bx + bw, by + bh])
        labels.append(1)
        masks.append(inst["mask"].astype(np.uint8))
        areas.append(inst.get("area", bw * bh))

        if confidences is not None:
            confs.append(confidences[i])
        else:
            confs.append(1.0)

    if not boxes:
        return None

    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "masks": torch.tensor(np.stack(masks), dtype=torch.uint8),
        "area": torch.tensor(areas, dtype=torch.float32),
        "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        "is_pseudo": is_pseudo,
        "confidence": torch.tensor(confs, dtype=torch.float32),
    }
    return target


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

class ResizeAndPad:
    """Resize image so long side = target_size, preserving aspect ratio, then pad."""

    def __init__(self, target_size=512):
        self.target_size = target_size

    def __call__(self, image, target=None):
        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        img_pil = Image.fromarray(image)
        img_pil = img_pil.resize((new_w, new_h), Image.BILINEAR)
        img_np = np.array(img_pil)

        # Pad to target_size x target_size
        padded = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = img_np

        if target is not None:
            target = target.copy()
            # Scale boxes
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, 0] *= scale
                boxes[:, 1] *= scale
                boxes[:, 2] *= scale
                boxes[:, 3] *= scale
                target["boxes"] = boxes

            # Resize masks
            if "masks" in target and len(target["masks"]) > 0:
                masks_np = target["masks"].numpy()
                resized_masks = np.zeros((masks_np.shape[0], self.target_size, self.target_size), dtype=np.uint8)
                for i in range(masks_np.shape[0]):
                    m_pil = Image.fromarray(masks_np[i])
                    m_pil = m_pil.resize((new_w, new_h), Image.NEAREST)
                    resized_masks[i, :new_h, :new_w] = np.array(m_pil)
                target["masks"] = torch.tensor(resized_masks, dtype=torch.uint8)

            return padded, target

        return padded


def default_transform(target_size=512):
    """Compose resize+pad with tensor conversion."""
    resize = ResizeAndPad(target_size)

    def transform(img, target):
        img, target = resize(img, target)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, target

    return transform


def collate_fn(batch):
    """Custom collate for detection."""
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def get_model(num_classes=2, pretrained_backbone=True):
    """Mask R-CNN R50-FPN."""
    if pretrained_backbone:
        model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    else:
        model = maskrcnn_resnet50_fpn(weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class PilotTrainer:
    """Simplified training loop for pilot experiments."""

    def __init__(self, model, device, lr=0.0025, momentum=0.9, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.lr_scheduler = None

    def train_epoch(self, dataloader, epoch, lambda_pseudo=0.5):
        """Train one epoch with mixed human + pseudo supervision."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for images, targets in pbar:
            images = [img.to(self.device) for img in images]
            targets_list = []
            for t in targets:
                td = {}
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        td[k] = v.to(self.device)
                    else:
                        td[k] = v
                targets_list.append(td)

            # Separate human and pseudo
            human_imgs, human_tgts = [], []
            pseudo_imgs, pseudo_tgts = [], []

            for img, tgt in zip(images, targets_list):
                if tgt.get("is_pseudo", False):
                    pseudo_imgs.append(img)
                    pseudo_tgts.append(tgt)
                else:
                    human_imgs.append(img)
                    human_tgts.append(tgt)

            loss_dict = {}

            # Human loss
            if human_imgs:
                h_loss = self.model(human_imgs, human_tgts)
                for k, v in h_loss.items():
                    loss_dict[k] = v

            # Pseudo loss (filtered by confidence)
            if pseudo_imgs:
                filtered_imgs, filtered_tgts = [], []
                for img, tgt in zip(pseudo_imgs, pseudo_tgts):
                    conf = tgt.get("confidence", torch.ones(len(tgt["boxes"])))
                    mask = conf > 0
                    if mask.sum() > 0:
                        ft = {
                            "boxes": tgt["boxes"][mask],
                            "labels": tgt["labels"][mask],
                            "masks": tgt["masks"][mask],
                            "area": tgt["area"][mask] if "area" in tgt else None,
                            "iscrowd": tgt["iscrowd"][mask] if "iscrowd" in tgt else None,
                        }
                        if ft["area"] is None:
                            del ft["area"]
                        if ft["iscrowd"] is None:
                            del ft["iscrowd"]
                        filtered_imgs.append(img)
                        filtered_tgts.append(ft)

                if filtered_imgs:
                    p_loss = self.model(filtered_imgs, filtered_tgts)
                    for k, v in p_loss.items():
                        if k in loss_dict:
                            loss_dict[k] = loss_dict[k] + lambda_pseudo * v
                        else:
                            loss_dict[k] = lambda_pseudo * v

            if loss_dict:
                losses = sum(loss_dict.values())
                self.optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 10.0
                )
                self.optimizer.step()

                total_loss += losses.item()
                n_batches += 1
                pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def validate(self, dataloader):
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for images, targets in dataloader:
            images = [img.to(self.device) for img in images]
            targets_list = []
            for t in targets:
                td = {}
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        td[k] = v.to(self.device)
                    else:
                        td[k] = v
                targets_list.append(td)

            loss_dict = self.model(images, targets_list)
            losses = sum(loss_dict.values())
            total_loss += losses.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    def save_checkpoint(self, path, epoch, **kwargs):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            **kwargs,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

class FrameDataset(torch.utils.data.Dataset):
    """Dataset from a list of (rgb_path, mask_path, target) tuples."""

    def __init__(self, samples, transform=None):
        """
        samples: list of (rgb_path, target_dict) pairs
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, target = self.samples[idx]
        img = Image.open(rgb_path).convert("RGB")
        img_np = np.array(img)

        if self.transform:
            img_np, target = self.transform(img_np, target)

        return img_np, target


def build_human_label_dataset(frame_keys, frames_by_plant, transform, is_pseudo=False):
    """Build dataset from human-annotated frames.

    Args:
        frame_keys: list of (plant_id, rep_id, filename) tuples
        frames_by_plant: lookup dict
        transform: image transform
        is_pseudo: mark as pseudo labels

    Returns:
        list of (rgb_path, target) pairs
    """
    samples = []
    for plant_id, rep_id, filename in frame_keys:
        frames = frames_by_plant.get((plant_id, rep_id), [])
        frame_entry = None
        for f in frames:
            if f["filename"] == filename:
                frame_entry = f
                break

        if frame_entry is None:
            continue

        rgb_path = frame_entry["rgb_path"]
        mask_path = frame_entry["mask_path"]

        instances = extract_instances_from_mask(mask_path)
        if not instances:
            continue

        img = Image.open(rgb_path).convert("RGB")
        w, h = img.size
        target = instances_to_target(instances, (w, h), is_pseudo=is_pseudo)
        if target is not None:
            samples.append((rgb_path, target))

    return samples


# ---------------------------------------------------------------------------
# Frame selection methods
# ---------------------------------------------------------------------------

def select_seed_frames(train_plants, frames_by_plant):
    """Select 1 mid-time frame per plant as common seed."""
    seeds = set()
    for plant_id, rep_id in train_plants:
        frames = frames_by_plant.get((plant_id, rep_id), [])
        if not frames:
            continue
        mid_idx = len(frames) // 2
        frame = frames[mid_idx]
        seeds.add((plant_id, rep_id, frame["filename"]))
    return seeds


def select_time_coverage(frame_keys, frames_by_plant, n_to_select, already_selected):
    """Select frames evenly spread in time, per-plant balanced."""
    # Group by plant
    by_plant = collections.defaultdict(list)
    for pid, rid, fn in frame_keys:
        if (pid, rid, fn) not in already_selected:
            by_plant[(pid, rid)].append((pid, rid, fn))

    # Compute per-plant ratios
    ratios = {}
    for (pid, rid), avail in by_plant.items():
        total = len(frames_by_plant.get((pid, rid), []))
        n_sel = sum(1 for pk in already_selected if pk[0] == pid and pk[1] == rid)
        ratios[(pid, rid)] = n_sel / max(1, total)

    # Sort plants by lowest ratio first
    plant_order = sorted(by_plant.keys(), key=lambda p: ratios.get(p, 0))

    selected = []
    selected_set = set(already_selected)

    for plant_key in plant_order:
        if len(selected) >= n_to_select:
            break
        avail = by_plant[plant_key]
        if not avail:
            continue

        # Sort by timestamp
        pid, rid = plant_key
        frames = frames_by_plant.get((pid, rid), [])
        ts_lookup = {f["filename"]: f.get("timestamp", "") for f in frames}
        avail.sort(key=lambda x: ts_lookup.get(x[2], ""))

        # Select evenly
        n_can = min(len(avail), n_to_select - len(selected))
        if n_can <= 0:
            break
        step = max(1, len(avail) // n_can)
        for i in range(0, len(avail), step):
            if len(selected) >= n_to_select:
                break
            key = avail[i]
            if key not in selected_set:
                selected.append(key)
                selected_set.add(key)

    return selected


def select_uncertainty(model, frame_keys, frames_by_plant, device, n_to_select, already_selected):
    """Select frames by TTA disagreement (U signal only)."""
    model.eval()

    scores = []
    for pid, rid, fn in frame_keys:
        if (pid, rid, fn) in already_selected:
            continue

        frames = frames_by_plant.get((pid, rid), [])
        entry = None
        for f in frames:
            if f["filename"] == fn:
                entry = f
                break
        if entry is None:
            continue

        rgb_path = entry["rgb_path"]
        try:
            img = Image.open(rgb_path).convert("RGB")
        except Exception:
            continue

        # TTA: original + hflip + vflip
        img_tensor = T.ToTensor()(img).to(device)
        with torch.no_grad():
            pred_orig = model([img_tensor])[0]

        n_orig = len(pred_orig["boxes"])
        if n_orig == 0:
            scores.append(((pid, rid, fn), 0.0))
            continue

        # Augmented predictions
        total_disagree = 0
        for aug_fn in [T.functional.hflip, T.functional.vflip]:
            img_aug = aug_fn(img)
            tensor_aug = T.ToTensor()(img_aug).to(device)
            with torch.no_grad():
                pred_aug = model([tensor_aug])[0]

            n_aug = len(pred_aug["boxes"])
            if n_aug == 0:
                total_disagree += n_orig
                continue

            # Simple IoU matching
            boxes_orig = pred_orig["boxes"]
            boxes_aug = pred_aug["boxes"]

            # Flip boxes back for hflip
            if aug_fn == T.functional.hflip:
                w = img.width
                if len(boxes_aug) > 0:
                    b = boxes_aug.clone()
                    b[:, 0] = w - boxes_aug[:, 2]
                    b[:, 2] = w - boxes_aug[:, 0]
                    boxes_aug = b

            # Compute IoU
            ious = _compute_iou(boxes_orig, boxes_aug)
            matched = 0
            for i in range(n_orig):
                if ious.shape[1] > 0 and ious[i].max() > 0.5:
                    matched += 1
                else:
                    total_disagree += 1
            total_disagree += max(0, n_aug - matched)

        instability = total_disagree / max(1, n_orig * 3)
        scores.append(((pid, rid, fn), instability))

    # Sort by score descending
    scores.sort(key=lambda x: -x[1])

    # Per-plant balanced selection
    return _balanced_select(scores, n_to_select, already_selected, frames_by_plant)


def select_event_aware(model, frame_keys, frames_by_plant, device, n_to_select,
                       already_selected, propagated_cache=None):
    """Select frames by combined (U+D+E)/3 score."""
    model.eval()
    if propagated_cache is None:
        propagated_cache = {}

    scores = []
    for pid, rid, fn in frame_keys:
        if (pid, rid, fn) in already_selected:
            continue

        frames = frames_by_plant.get((pid, rid), [])
        entry = None
        for f in frames:
            if f["filename"] == fn:
                entry = f
                break
        if entry is None:
            continue

        rgb_path = entry["rgb_path"]
        try:
            img = Image.open(rgb_path).convert("RGB")
        except Exception:
            continue

        img_tensor = T.ToTensor()(img).to(device)
        with torch.no_grad():
            pred_orig = model([img_tensor])[0]

        n_orig = len(pred_orig["boxes"])

        # U signal: TTA instability
        U = 0.0
        if n_orig > 0:
            total_disagree = 0
            for aug_fn in [T.functional.hflip, T.functional.vflip]:
                img_aug = aug_fn(img)
                tensor_aug = T.ToTensor()(img_aug).to(device)
                with torch.no_grad():
                    pred_aug = model([tensor_aug])[0]

                n_aug = len(pred_aug["boxes"])
                if n_aug == 0:
                    total_disagree += n_orig
                    continue

                boxes_aug = pred_aug["boxes"]
                if aug_fn == T.functional.hflip:
                    w = img.width
                    if len(boxes_aug) > 0:
                        b = boxes_aug.clone()
                        b[:, 0] = w - boxes_aug[:, 2]
                        b[:, 2] = w - boxes_aug[:, 0]
                        boxes_aug = b

                ious = _compute_iou(pred_orig["boxes"], boxes_aug)
                matched = 0
                for i in range(n_orig):
                    if ious.shape[1] > 0 and ious[i].max() > 0.5:
                        matched += 1
                    else:
                        total_disagree += 1
                total_disagree += max(0, n_aug - matched)

            U = total_disagree / max(1, n_orig * 3)

        # D signal: propagation divergence
        frame_key = (pid, rid, fn)
        propagated = propagated_cache.get(frame_key, [])
        D = 0.0
        if n_orig > 0:
            if len(propagated) == 0:
                D = 1.0
            else:
                prop_boxes = torch.tensor([p["bbox"] for p in propagated], dtype=torch.float32).to(device)
                ious = _compute_iou(pred_orig["boxes"], prop_boxes)
                matched_iou = 0
                n_matched = 0
                for i in range(n_orig):
                    if ious.shape[1] > 0 and ious[i].max() > 0.5:
                        matched_iou += ious[i].max().item()
                        n_matched += 1
                D = 1.0 - matched_iou / max(n_orig, len(propagated), 1)

        # E signal: uncovered candidates
        E = 0.0
        if n_orig > 0:
            if len(propagated) == 0:
                high_conf = (pred_orig["scores"] > 0.5).sum().item()
                E = high_conf / max(1, n_orig)
            else:
                prop_boxes = torch.tensor([p["bbox"] for p in propagated], dtype=torch.float32).to(device)
                ious = _compute_iou(pred_orig["boxes"], prop_boxes)
                n_uncovered = 0
                for i in range(n_orig):
                    if ious.shape[1] == 0 or ious[i].max() < 0.5:
                        n_uncovered += 1
                E = n_uncovered / max(1, n_orig)

        S = (U + D + E) / 3.0
        scores.append(((pid, rid, fn), S))

    scores.sort(key=lambda x: -x[1])
    return _balanced_select(scores, n_to_select, already_selected, frames_by_plant)


def _compute_iou(boxes1, boxes2):
    """IoU matrix [N, M] between two box tensors [N/M, 4] in xyxy."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros(len(boxes1), len(boxes2), device=boxes1.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


def _balanced_select(scores, n_to_select, already_selected, frames_by_plant):
    """Select from scored list with per-plant balance."""
    by_plant = collections.defaultdict(list)
    for key, score in scores:
        pid, rid, fn = key
        by_plant[(pid, rid)].append((key, score))

    ratios = {}
    for (pid, rid), items in by_plant.items():
        total = len(frames_by_plant.get((pid, rid), []))
        n_sel = sum(1 for pk in already_selected if pk[0] == pid and pk[1] == rid)
        ratios[(pid, rid)] = n_sel / max(1, total)

    plant_order = sorted(by_plant.keys(), key=lambda p: ratios.get(p, 0))

    selected = []
    selected_set = set(already_selected)

    for plant_key in plant_order:
        if len(selected) >= n_to_select:
            break
        items = by_plant[plant_key]
        items.sort(key=lambda x: -x[1])
        for key, score in items:
            if len(selected) >= n_to_select:
                break
            if key not in selected_set:
                selected.append(key)
                selected_set.add(key)

    return selected


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def propagate_from_anchors(model, anchor_keys, all_frame_keys, frames_by_plant,
                           device, mode="reliable", time_window_hours=48,
                           conf_threshold=0.8, iou_threshold=0.7):
    """Propagate annotations from anchor frames to nearby unannotated frames.

    Args:
        model: teacher model
        anchor_keys: set of (plant_id, rep_id, filename) with human labels
        all_frame_keys: all candidate frame keys for this plant
        frames_by_plant: lookup
        device: torch device
        mode: "basic" (conf only) or "reliable" (conf + IoU)
        time_window_hours: max hours from anchor to propagate
        conf_threshold: minimum teacher confidence
        iou_threshold: minimum IoU for reliable mode

    Returns:
        dict: frame_key -> list of propagated instances with 'mask', 'bbox', 'score', 'q_weight'
    """
    model.eval()
    propagated = {}

    # Group anchors by plant
    anchors_by_plant = collections.defaultdict(list)
    for key in anchor_keys:
        pid, rid, fn = key
        anchors_by_plant[(pid, rid)].append(key)

    for plant_key, plant_anchors in anchors_by_plant.items():
        pid, rid = plant_key
        frames = frames_by_plant.get((pid, rid), [])
        if not frames:
            continue

        # Build timestamp lookup
        ts_lookup = {}
        for f in frames:
            ts_str = f.get("timestamp", "")
            try:
                ts_lookup[f["filename"]] = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                ts_lookup[f["filename"]] = None

        # Get teacher predictions on anchor frames
        anchor_preds = {}
        for akey in plant_anchors:
            _, _, afn = akey
            entry = None
            for f in frames:
                if f["filename"] == afn:
                    entry = f
                    break
            if entry is None:
                continue

            img = Image.open(entry["rgb_path"]).convert("RGB")
            img_tensor = T.ToTensor()(img).to(device)
            with torch.no_grad():
                pred = model([img_tensor])[0]

            anchor_preds[afn] = pred

        # For each non-anchor frame, check if within time window of any anchor
        for frame in frames:
            fn = frame["filename"]
            fkey = (pid, rid, fn)
            if fkey in anchor_keys:
                continue

            f_ts = ts_lookup.get(fn)
            if f_ts is None:
                continue

            # Find closest anchor in time
            best_anchor = None
            best_dt = timedelta.max
            for akey in plant_anchors:
                _, _, afn = akey
                a_ts = ts_lookup.get(afn)
                if a_ts is None:
                    continue
                dt = abs(f_ts - a_ts)
                if dt < best_dt:
                    best_dt = dt
                    best_anchor = afn

            if best_anchor is None or best_dt > timedelta(hours=time_window_hours):
                continue

            # Run teacher on target frame
            img = Image.open(frame["rgb_path"]).convert("RGB")
            img_tensor = T.ToTensor()(img).to(device)
            with torch.no_grad():
                target_pred = model([img_tensor])[0]

            # Get anchor prediction
            anchor_pred = anchor_preds.get(best_anchor)
            if anchor_pred is None:
                continue

            # Match target detections to anchor instances
            accepted_instances = []
            n_target = len(target_pred["boxes"])
            n_anchor = len(anchor_pred["boxes"])

            if n_target == 0:
                continue

            if mode == "basic":
                # Accept if teacher confidence >= threshold
                for i in range(n_target):
                    score = target_pred["scores"][i].item()
                    if score >= conf_threshold:
                        mask_i = (target_pred["masks"][i, 0] > 0.5).cpu().numpy().astype(np.uint8)
                        box = target_pred["boxes"][i].cpu().numpy().tolist()
                        accepted_instances.append({
                            "mask": mask_i,
                            "bbox": [int(box[0]), int(box[1]),
                                     int(box[2] - box[0]), int(box[3] - box[1])],
                            "score": score,
                            "q_weight": score,
                        })

            elif mode == "reliable":
                # Require IoU(propagated_mask, target_detection_mask) >= threshold
                # AND confidence >= threshold
                # Bidirectional: also check anchor detections against target
                anchor_masks = []
                for i in range(n_anchor):
                    am = (anchor_pred["masks"][i, 0] > 0.5).cpu().numpy().astype(np.uint8)
                    anchor_masks.append(am)

                for i in range(n_target):
                    score = target_pred["scores"][i].item()
                    if score < conf_threshold:
                        continue

                    target_mask = (target_pred["masks"][i, 0] > 0.5).cpu().numpy().astype(np.uint8)
                    box = target_pred["boxes"][i].cpu().numpy().tolist()

                    # Check IoU with any anchor mask
                    best_iou = 0.0
                    for am in anchor_masks:
                        inter = (target_mask & am).sum()
                        union = (target_mask | am).sum()
                        iou_val = inter / max(1, union)
                        best_iou = max(best_iou, iou_val)

                    if best_iou >= iou_threshold:
                        q_weight = score * best_iou
                        accepted_instances.append({
                            "mask": target_mask,
                            "bbox": [int(box[0]), int(box[1]),
                                     int(box[2] - box[0]), int(box[3] - box[1])],
                            "score": score,
                            "q_weight": q_weight,
                        })

            if accepted_instances:
                propagated[fkey] = accepted_instances

    return propagated


# ---------------------------------------------------------------------------
# COCO Evaluation
# ---------------------------------------------------------------------------

def evaluate_model_coco(model, test_frame_keys, frames_by_plant, device, target_size=512):
    """Evaluate model on test frames using COCO mask AP.

    Returns dict with AP, AP50, AP75, etc.
    """
    model.eval()
    resize = ResizeAndPad(target_size)

    coco_images = []
    coco_annotations_gt = []
    coco_predictions = []
    img_id = 0
    ann_id = 0

    for pid, rid, fn in tqdm(test_frame_keys, desc="Evaluating"):
        frames = frames_by_plant.get((pid, rid), [])
        entry = None
        for f in frames:
            if f["filename"] == fn:
                entry = f
                break
        if entry is None:
            continue

        rgb_path = entry["rgb_path"]
        mask_path = entry["mask_path"]

        img = Image.open(rgb_path).convert("RGB")
        w, h = img.size
        img_id += 1

        coco_images.append({
            "id": img_id,
            "file_name": rgb_path,
            "width": w,
            "height": h,
        })

        # Ground truth
        gt_instances = extract_instances_from_mask(mask_path)
        for inst in gt_instances:
            ann_id += 1
            binary_mask = inst["mask"].astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]

            coco_annotations_gt.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "bbox": inst["bbox"],
                "area": inst["area"],
                "iscrowd": 0,
            })

        # Model prediction
        img_resized = resize(np.array(img))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            pred = model([img_tensor])[0]

        # Scale predictions back to original size
        scale_x = w / target_size
        scale_y = h / target_size

        for i in range(len(pred["boxes"])):
            score = pred["scores"][i].item()
            if score < 0.05:
                continue

            box = pred["boxes"][i].cpu().numpy()
            x1, y1, x2, y2 = box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            # Resize mask back to original size
            mask_pred = (pred["masks"][i, 0] > 0.5).cpu().numpy().astype(np.uint8)
            mask_pil = Image.fromarray(mask_pred)
            mask_pil = mask_pil.resize((w, h), Image.NEAREST)
            mask_resized = np.array(mask_pil)

            ann_id += 1
            rle = mask_utils.encode(np.asfortranarray(mask_resized))
            rle["counts"] = rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]

            coco_predictions.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "bbox": [float(x1), float(y1), float(bw), float(bh)],
                "score": score,
            })

    if not coco_annotations_gt or not coco_predictions:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "APs": 0.0, "APm": 0.0, "APl": 0.0,
                "AR1": 0.0, "AR10": 0.0, "AR100": 0.0, "n_gt": len(coco_annotations_gt),
                "n_pred": len(coco_predictions)}

    # Write GT to temp file
    coco_gt_dict = {
        "info": {"description": "Leaf Eval"},
        "licenses": [],
        "categories": [{"id": 1, "name": "leaf", "supercategory": "plant"}],
        "images": coco_images,
        "annotations": coco_annotations_gt,
    }

    tmp_gt = os.path.join(os.environ.get("TEMP", "/tmp"), f"coco_gt_{img_id}.json")
    with open(tmp_gt, "w") as f:
        json.dump(coco_gt_dict, f)

    coco_gt_obj = COCO(tmp_gt)
    coco_dt = coco_gt_obj.loadRes(coco_predictions)

    coco_eval = COCOeval(coco_gt_obj, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results = {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "APs": float(coco_eval.stats[3]),
        "APm": float(coco_eval.stats[4]),
        "APl": float(coco_eval.stats[5]),
        "AR1": float(coco_eval.stats[6]),
        "AR10": float(coco_eval.stats[7]),
        "AR100": float(coco_eval.stats[8]),
        "n_gt": len(coco_annotations_gt),
        "n_pred": len(coco_predictions),
    }

    try:
        os.remove(tmp_gt)
    except OSError:
        pass

    return results


# ---------------------------------------------------------------------------
# Method definitions
# ---------------------------------------------------------------------------

METHODS = {
    "M1": {
        "name": "Time-coverage + pure human labels (no pseudo)",
        "selection": "time_coverage",
        "propagation": None,
        "use_pseudo": False,
    },
    "M2": {
        "name": "Uncertainty selection + spatial self-training (no temporal, no SAM2)",
        "selection": "uncertainty",
        "propagation": None,
        "use_pseudo": True,
        "pseudo_mode": "spatial_self_train",
    },
    "M3": {
        "name": "Time-coverage + basic propagation (confidence only)",
        "selection": "time_coverage",
        "propagation": "basic",
        "use_pseudo": True,
    },
    "M4": {
        "name": "Event-aware + reliable propagation (full method)",
        "selection": "event_aware",
        "propagation": "reliable",
        "use_pseudo": True,
    },
    "M5": {
        "name": "Time-coverage + reliable propagation (ablation: no event-aware)",
        "selection": "time_coverage",
        "propagation": "reliable",
        "use_pseudo": True,
    },
    "M6": {
        "name": "Event-aware + basic propagation (ablation: no reliability)",
        "selection": "event_aware",
        "propagation": "basic",
        "use_pseudo": True,
    },
}


# ---------------------------------------------------------------------------
# Main pilot runner
# ---------------------------------------------------------------------------

def run_single_method(method_id, method_config, seed, train_plants, val_plants, test_plants,
                      frames_by_plant, budget, device, output_dir):
    """Run a single method+seed combination.

    Returns dict with metrics and logs.
    """
    print(f"\n{'='*60}")
    print(f"Running {method_id}: {method_config['name']} (seed={seed})")
    print(f"{'='*60}")

    method_dir = output_dir / f"{method_id}_seed{seed}"
    method_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "method": method_id,
        "method_name": method_config["name"],
        "seed": seed,
        "start_time": datetime.now().isoformat(),
    }

    # Step 1: Select seed frames
    seed_keys = select_seed_frames(train_plants, frames_by_plant)
    log["n_seed_frames"] = len(seed_keys)
    print(f"  Seed frames: {len(seed_keys)}")

    # Step 2: Compute additional budget
    n_additional = budget - len(seed_keys)
    if n_additional < 0:
        n_additional = 0
    log["n_additional_budget"] = n_additional
    print(f"  Additional budget: {n_additional}")

    # Step 3: Build candidate frame keys (all train frames not in seed)
    all_train_keys = set()
    for pid, rid in train_plants:
        for f in frames_by_plant.get((pid, rid), []):
            all_train_keys.add((pid, rid, f["filename"]))

    # Step 4: Train teacher on seed frames first (for selection scoring)
    print("  Training teacher on seed frames...")
    transform = default_transform(512)
    teacher_samples = build_human_label_dataset(seed_keys, frames_by_plant, transform)

    if len(teacher_samples) < 2:
        print("  WARNING: Too few seed samples for teacher training. Skipping method.")
        log["error"] = "Too few seed samples"
        log["end_time"] = datetime.now().isoformat()
        return log

    teacher_model = get_model(num_classes=2, pretrained_backbone=True)
    teacher_dataset = FrameDataset(teacher_samples, transform)
    teacher_loader = torch.utils.data.DataLoader(
        teacher_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0
    )

    teacher_trainer = PilotTrainer(teacher_model, device)
    teacher_epochs = min(50, max(10, 2000 // max(1, len(teacher_loader))))
    for ep in range(teacher_epochs):
        loss = teacher_trainer.train_epoch(teacher_loader, ep)
        if ep % 10 == 0:
            print(f"    Teacher epoch {ep}: loss={loss:.4f}")

    teacher_ckpt = method_dir / "teacher_seed.pth"
    teacher_trainer.save_checkpoint(str(teacher_ckpt), teacher_epochs)

    # Step 5: Select additional frames based on method
    already_selected = set(seed_keys)
    selection_method = method_config["selection"]

    if selection_method == "time_coverage":
        additional_keys = select_time_coverage(
            all_train_keys, frames_by_plant, n_additional, already_selected
        )
    elif selection_method == "uncertainty":
        additional_keys = select_uncertainty(
            teacher_model, all_train_keys, frames_by_plant, device, n_additional, already_selected
        )
    elif selection_method == "event_aware":
        # For event-aware, we need propagated cache from seed
        propagated_cache = {}
        additional_keys = select_event_aware(
            teacher_model, all_train_keys, frames_by_plant, device, n_additional,
            already_selected, propagated_cache
        )
    else:
        additional_keys = []

    all_selected = already_selected | set(additional_keys)
    log["n_total_selected"] = len(all_selected)
    log["n_additional_selected"] = len(additional_keys)
    print(f"  Total selected frames: {len(all_selected)}")

    # Save selection log
    with open(method_dir / "selection_log.json", "w") as f:
        json.dump({
            "seed": sorted([list(k) for k in seed_keys]),
            "additional": sorted([list(k) for k in additional_keys]),
            "total": sorted([list(k) for k in all_selected]),
        }, f, indent=2)

    # Step 6: Retrain teacher on all selected frames
    print("  Retraining teacher on all selected frames...")
    teacher_model2 = get_model(num_classes=2, pretrained_backbone=True)
    teacher_samples2 = build_human_label_dataset(all_selected, frames_by_plant, transform)
    teacher_dataset2 = FrameDataset(teacher_samples2, transform)
    teacher_loader2 = torch.utils.data.DataLoader(
        teacher_dataset2, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0
    )

    teacher_trainer2 = PilotTrainer(teacher_model2, device)
    teacher_epochs2 = min(100, max(20, 2000 // max(1, len(teacher_loader2))))
    for ep in range(teacher_epochs2):
        loss = teacher_trainer2.train_epoch(teacher_loader2, ep)
        if ep % 20 == 0:
            print(f"    Teacher epoch {ep}: loss={loss:.4f}")

    teacher_ckpt2 = method_dir / "teacher_full.pth"
    teacher_trainer2.save_checkpoint(str(teacher_ckpt2), teacher_epochs2)

    # Step 7: Propagation (if applicable)
    pseudo_samples = []
    if method_config["use_pseudo"] and method_config.get("propagation") is not None:
        prop_mode = method_config["propagation"]
        print(f"  Running {prop_mode} propagation...")

        if prop_mode == "spatial_self_train":
            # M2: spatial self-training without temporal propagation
            # Use teacher predictions on all selected frames as pseudo labels
            pseudo_samples = _spatial_self_train(
                teacher_model2, all_selected, frames_by_plant, device
            )
        else:
            # Temporal propagation from anchors
            propagated = propagate_from_anchors(
                teacher_model2, all_selected, all_train_keys, frames_by_plant,
                device, mode=prop_mode,
                time_window_hours=48,
                conf_threshold=0.8,
                iou_threshold=0.7 if prop_mode == "reliable" else 0.0,
            )
            log["n_propagated_frames"] = len(propagated)
            log["n_propagated_instances"] = sum(len(v) for v in propagated.values())
            print(f"  Propagated to {len(propagated)} frames, {log['n_propagated_instances']} instances")

            # Build pseudo dataset
            for fkey, instances in propagated.items():
                pid, rid, fn = fkey
                frames = frames_by_plant.get((pid, rid), [])
                entry = None
                for f in frames:
                    if f["filename"] == fn:
                        entry = f
                        break
                if entry is None:
                    continue

                rgb_path = entry["rgb_path"]
                img = Image.open(rgb_path).convert("RGB")
                w, h = img.size

                confs = [inst["q_weight"] for inst in instances]
                target = instances_to_target(instances, (w, h), is_pseudo=True, confidences=confs)
                if target is not None:
                    pseudo_samples.append((rgb_path, target))

    # Save propagation log
    with open(method_dir / "propagation_log.json", "w") as f:
        json.dump({
            "n_pseudo_samples": len(pseudo_samples),
            "propagation_mode": method_config.get("propagation", "none"),
        }, f, indent=2)

    # Step 8: Train student model
    print("  Training student model...")
    student_model = get_model(num_classes=2, pretrained_backbone=False)
    # Initialize from teacher
    student_model.load_state_dict(teacher_model2.state_dict())

    # Build combined dataset
    human_samples = build_human_label_dataset(all_selected, frames_by_plant, transform, is_pseudo=False)
    all_samples = human_samples + pseudo_samples

    if len(all_samples) < 2:
        print("  WARNING: Too few samples for student training.")
        log["error"] = "Too few training samples"
        log["end_time"] = datetime.now().isoformat()
        return log

    student_dataset = FrameDataset(all_samples, transform)
    student_loader = torch.utils.data.DataLoader(
        student_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0
    )

    student_trainer = PilotTrainer(student_model, device)
    student_epochs = min(100, max(20, 2000 // max(1, len(student_loader))))
    for ep in range(student_epochs):
        lam = 0.5 if pseudo_samples else 0.0
        loss = student_trainer.train_epoch(student_loader, ep, lambda_pseudo=lam)
        if ep % 20 == 0:
            print(f"    Student epoch {ep}: loss={loss:.4f}")

    student_ckpt = method_dir / "student.pth"
    student_trainer.save_checkpoint(str(student_ckpt), student_epochs)

    # Step 9: Evaluate on test set
    print("  Evaluating on test set...")
    test_keys = set()
    for pid, rid in test_plants:
        for f in frames_by_plant.get((pid, rid), []):
            test_keys.add((pid, rid, f["filename"]))

    test_results = evaluate_model_coco(student_model, test_keys, frames_by_plant, device)
    log["test_metrics"] = test_results
    print(f"  Test AP: {test_results['AP']:.4f}, AP50: {test_results['AP50']:.4f}")

    # Also evaluate teacher
    print("  Evaluating teacher on test set...")
    teacher_test_results = evaluate_model_coco(teacher_model2, test_keys, frames_by_plant, device)
    log["teacher_test_metrics"] = teacher_test_results
    print(f"  Teacher AP: {teacher_test_results['AP']:.4f}")

    log["end_time"] = datetime.now().isoformat()

    # Save log
    with open(method_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    return log


def _spatial_self_train(model, selected_keys, frames_by_plant, device, conf_threshold=0.8):
    """M2: Spatial self-training — use teacher predictions as pseudo labels on same frames.
    No temporal propagation, no SAM2.
    """
    model.eval()
    pseudo_samples = []
    transform = default_transform(512)

    for pid, rid, fn in selected_keys:
        frames = frames_by_plant.get((pid, rid), [])
        entry = None
        for f in frames:
            if f["filename"] == fn:
                entry = f
                break
        if entry is None:
            continue

        rgb_path = entry["rgb_path"]
        img = Image.open(rgb_path).convert("RGB")
        w, h = img.size

        img_resized, _ = transform(np.array(img), None)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            pred = model([img_tensor])[0]

        # Filter by confidence
        instances = []
        confs = []
        for i in range(len(pred["boxes"])):
            score = pred["scores"][i].item()
            if score < conf_threshold:
                continue
            mask_i = (pred["masks"][i, 0] > 0.5).cpu().numpy().astype(np.uint8)
            box = pred["boxes"][i].cpu().numpy()
            instances.append({
                "mask": mask_i,
                "bbox": [int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])],
                "area": int(mask_i.sum()),
                "leaf_id": i + 1,
            })
            confs.append(score)

        if instances:
            target = instances_to_target(instances, (w, h), is_pseudo=True, confidences=confs)
            if target is not None:
                pseudo_samples.append((rgb_path, target))

    return pseudo_samples


def run_pilot(args):
    """Main pilot experiment runner."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    print("\nLoading manifest and splits...")
    manifest, frames_by_plant = load_manifest(MANIFEST_PATH)
    split = load_split(SPLIT_PATH)
    print(f"  Total frames: {len(manifest)}")
    print(f"  Train plants: {len(split['train_plants'])}")
    print(f"  Val plants: {len(split['val_plants'])}")
    print(f"  Test plants: {len(split['test_plants'])}")

    # Subsample plants
    print(f"\nSubsampling: {args.n_train} train, {args.n_val} val plants...")
    train_plants = subsample_plants(
        [list(p) for p in split["train_plants"]],
        args.n_train,
        frames_by_plant,
        seed=args.subsample_seed,
    )
    val_plants = subsample_plants(
        [list(p) for p in split["val_plants"]],
        args.n_val,
        frames_by_plant,
        seed=args.subsample_seed,
    )
    test_plants = [list(p) for p in split["test_plants"]]

    print(f"  Selected {len(train_plants)} train plants")
    print(f"  Selected {len(val_plants)} val plants")
    print(f"  Using {len(test_plants)} test plants")

    # Compute budget
    total_train_frames = sum(
        len(frames_by_plant.get((p[0], p[1]), []))
        for p in train_plants
    )
    budget = math.ceil(args.budget_ratio * total_train_frames)
    print(f"\n  Total train frames: {total_train_frames}")
    print(f"  Budget ({args.budget_ratio*100:.0f}%): {budget} frames")

    # Save config
    config = {
        "n_train_plants": len(train_plants),
        "n_val_plants": len(val_plants),
        "n_test_plants": len(test_plants),
        "budget_ratio": args.budget_ratio,
        "total_budget": budget,
        "seeds": args.seeds,
        "methods": list(METHODS.keys()),
        "subsample_seed": args.subsample_seed,
        "train_plants": train_plants,
        "val_plants": val_plants,
        "test_plants": test_plants,
    }

    with open(output_dir / "pilot_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run all methods x seeds
    all_results = []
    total_runs = len(METHODS) * len(args.seeds)
    run_idx = 0

    for seed in args.seeds:
        for method_id, method_config in METHODS.items():
            run_idx += 1
            print(f"\n{'#'*60}")
            print(f"# Run {run_idx}/{total_runs}: {method_id} seed={seed}")
            print(f"{'#'*60}")

            # Set seeds for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            result = run_single_method(
                method_id, method_config, seed,
                train_plants, val_plants, test_plants,
                frames_by_plant, budget, device, output_dir,
            )
            all_results.append(result)

            # Save intermediate results
            with open(output_dir / "results_intermediate.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # Summary table
    print("\n" + "=" * 80)
    print("PILOT RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Method':<6} {'Seed':<6} {'AP':>8} {'AP50':>8} {'AP75':>8} {'AR100':>8} {'Teacher AP':>10}")
    print("-" * 80)

    for r in all_results:
        test_m = r.get("test_metrics", {})
        teacher_m = r.get("teacher_test_metrics", {})
        print(f"{r['method']:<6} {r['seed']:<6} "
              f"{test_m.get('AP', 0):>8.4f} {test_m.get('AP50', 0):>8.4f} "
              f"{test_m.get('AP75', 0):>8.4f} {test_m.get('AR100', 0):>8.4f} "
              f"{teacher_m.get('AP', 0):>10.4f}")

    # Aggregate by method
    print("\n" + "-" * 80)
    print("AGGREGATED BY METHOD (mean +/- std over seeds)")
    print("-" * 80)

    method_results = collections.defaultdict(list)
    for r in all_results:
        method_results[r["method"]].append(r)

    for method_id in sorted(method_results.keys()):
        results = method_results[method_id]
        aps = [r.get("test_metrics", {}).get("AP", 0) for r in results]
        ap50s = [r.get("test_metrics", {}).get("AP50", 0) for r in results]
        tap = [r.get("teacher_test_metrics", {}).get("AP", 0) for r in results]

        print(f"{method_id}: AP={np.mean(aps):.4f}+/-{np.std(aps):.4f}, "
              f"AP50={np.mean(ap50s):.4f}, Teacher AP={np.mean(tap):.4f}")

    # Save final results
    final_results = {
        "config": config,
        "runs": all_results,
        "summary": {},
        "timestamp": datetime.now().isoformat(),
    }

    for method_id, results in method_results.items():
        aps = [r.get("test_metrics", {}).get("AP", 0) for r in results]
        final_results["summary"][method_id] = {
            "AP_mean": float(np.mean(aps)),
            "AP_std": float(np.std(aps)),
            "n_runs": len(results),
        }

    with open(output_dir / "results_final.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")
    return final_results


def main():
    # Historical helper functions above remain importable for old checkpoint
    # diagnostics only. All new CLI runs use audited validation-only code.
    from training.pipeline import main as validated_main
    validated_main()


if __name__ == "__main__":
    main()
