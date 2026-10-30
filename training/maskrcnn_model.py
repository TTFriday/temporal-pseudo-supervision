"""
Mask R-CNN based leaf instance segmentation
=============================================
Uses torchvision's Mask R-CNN with ResNet-50-FPN backbone.
Supports:
- Full supervision (human labels)
- Partial-instance supervision (pseudo labels with ignore regions)
- Confidence-weighted pseudo label training
"""

import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.utils.data
import numpy as np
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm


class LeafDataset(torch.utils.data.Dataset):
    """Dataset for leaf instance segmentation.

    Supports both full human labels and partial pseudo labels.
    For pseudo labels, instances with q=0 are ignored (not treated as background).
    """

    def __init__(self, image_paths, targets, transform=None, is_pseudo=False):
        """
        Args:
            image_paths: list of image file paths
            targets: list of dicts with keys:
                - boxes: Tensor [N, 4] in xyxy format
                - labels: Tensor [N] (all 1 for leaf)
                - masks: Tensor [N, H, W]
                - confidence: Tensor [N] (1.0 for human, q_i for pseudo)
                - ignore_mask: Tensor [H, W] (1 where to ignore, 0 elsewhere)
            transform: optional image transform
            is_pseudo: whether this is a pseudo-label dataset
        """
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform
        self.is_pseudo = is_pseudo

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        target = self.targets[idx].copy()

        if self.transform:
            img, target = self.transform(img, target)

        # Convert to tensor
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, target


def get_model(num_classes=2, pretrained_backbone=True):
    """Get Mask R-CNN model with ResNet-50-FPN backbone.

    Args:
        num_classes: 2 (background + leaf)
        pretrained_backbone: use COCO pretrained weights
    """
    if pretrained_backbone:
        model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    else:
        model = maskrcnn_resnet50_fpn(weights_backbone=None)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get number of input features for mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # Replace mask predictor (torchvision >=0.21: 3-arg signature)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def collate_fn(batch):
    """Custom collate function for detection."""
    return tuple(zip(*batch))


def compute_loss_with_ignore(model, images, targets, lambda_pseudo=0.5):
    """Compatibility wrapper for confidence-weighted, positive-only pseudo loss.

    Human targets use the unchanged torchvision forward. Pseudo targets support
    confidence, valid_mask (1=valid), and ignore_mask (1=ignore). An entirely
    empty/rejected pseudo batch returns {}; skip backward/optimizer in that case.
    """
    if __package__:
        from .partial_supervision import compute_loss_with_ignore as partial_loss
    else:
        from partial_supervision import compute_loss_with_ignore as partial_loss
    return partial_loss(model, images, targets, lambda_pseudo=lambda_pseudo)


class Trainer:
    """Training loop for Mask R-CNN with mixed supervision."""

    def __init__(self, model, device, lr=0.0025, momentum=0.9, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device

        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.lr_scheduler = None

    def train_one_epoch(self, dataloader, epoch, max_epochs=10):
        self.model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for images, targets in pbar:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                       for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            total_loss += losses.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        avg_loss = total_loss / max(1, n_batches)
        return avg_loss

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


def targets_from_coco_annotation(coco_ann, image_size, is_pseudo=False, confidences=None):
    """Convert COCO-style annotation to training target dict.

    Args:
        coco_ann: dict with 'annotations' list, each having 'bbox', 'segmentation', 'area'
        image_size: (width, height)
        is_pseudo: whether this is a pseudo-label annotation
        confidences: optional list of confidence weights per instance
    """
    w, h = image_size
    boxes = []
    labels = []
    masks = []
    confs = []

    for i, ann in enumerate(coco_ann.get("annotations", [])):
        x, y, bw, bh = ann["bbox"]
        if bw <= 0 or bh <= 0:
            continue

        boxes.append([x, y, x + bw, y + bh])
        labels.append(1)  # leaf class

        # Create mask from segmentation
        if "segmentation" in ann:
            from pycocotools import mask as mask_utils
            if isinstance(ann["segmentation"], dict):
                # RLE format
                m = mask_utils.decode(ann["segmentation"])
            else:
                # Polygon format - skip for now, use bbox mask
                m = np.zeros((h, w), dtype=np.uint8)
                m[int(y):int(y+bh), int(x):int(x+bw)] = 1
            masks.append(m)

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
        "confidence": torch.tensor(confs, dtype=torch.float32),
        "is_pseudo": is_pseudo,
    }

    return target


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_model(num_classes=2, pretrained_backbone=True)
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")

    trainer = Trainer(model, device)
    print("Trainer initialized")
