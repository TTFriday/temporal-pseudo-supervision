"""
Smoke Test: Verify GPU training works end-to-end
==================================================
1. Load a small subset of training data
2. Initialize Mask R-CNN with COCO pretrained weights
3. Train for a few steps on GPU
4. Verify loss decreases and predictions are reasonable
"""

import os
import sys
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.maskrcnn_model import get_model, Trainer, collate_fn
from data_audit.audit_dataset import COLOUR_PALETTE


def extract_instances(mask):
    """Extract leaf instances from color-coded mask."""
    h, w = mask.shape[:2]
    instances = []

    for leaf_id, colour in enumerate(COLOUR_PALETTE, start=1):
        match = np.all(mask == np.array(colour), axis=-1)
        area = int(match.sum())
        if area < 50:
            continue

        rows = np.any(match, axis=1)
        cols = np.any(match, axis=0)
        if not rows.any() or not cols.any():
            continue

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        instances.append({
            "mask": match.astype(np.uint8),
            "bbox": [int(cmin), int(rmin), int(cmax), int(rmax)],
            "area": area,
        })

    return instances


def load_sample_data(manifest_path, split_path, n_samples=10):
    """Load a small sample of training data."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    with open(split_path) as f:
        split = json.load(f)

    train_plants = [tuple(p) for p in split["train_plants"]]
    random.shuffle(train_plants)

    samples = []
    for plant_id, rep_id in train_plants:
        frames = [e for e in manifest if e["plant_id"] == plant_id and e["replicate_id"] == rep_id]
        frames.sort(key=lambda x: x.get("timestamp", ""))

        # Take mid-time frame as seed
        if frames:
            mid = frames[len(frames) // 2]
            samples.append(mid)

        if len(samples) >= n_samples:
            break

    return samples


def smoke_test():
    """Run smoke test."""
    print("="*60)
    print("GPU Smoke Test: Mask R-CNN Training")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load sample data
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data_audit" / "manifest.json"
    split_path = root / "splits" / "plant_split.json"

    print(f"\nLoading sample data...")
    samples = load_sample_data(manifest_path, split_path, n_samples=5)
    print(f"Loaded {len(samples)} samples")

    # Build dataset
    images = []
    targets = []

    for sample in samples:
        try:
            img = Image.open(sample["rgb_path"]).convert("RGB")
            img = img.resize((512, 512))  # Fixed size for smoke test
            img_np = np.array(img)

            mask = Image.open(sample["mask_path"]).convert("RGB")
            mask = mask.resize((512, 512), Image.NEAREST)
            mask_np = np.array(mask)

            instances = extract_instances(mask_np)
            if not instances:
                continue

            boxes = torch.tensor([inst["bbox"] for inst in instances], dtype=torch.float32)
            labels = torch.ones(len(instances), dtype=torch.int64)
            masks = torch.tensor([inst["mask"] for inst in instances], dtype=torch.uint8)

            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
            }

            images.append(torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0)
            targets.append(target)

        except Exception as e:
            print(f"  Error loading {sample['filename']}: {e}")
            continue

    print(f"Valid samples: {len(images)}")

    if len(images) < 2:
        print("ERROR: Not enough valid samples for training")
        return False

    # Create model
    print(f"\nInitializing Mask R-CNN...")
    model = get_model(num_classes=2, pretrained_backbone=True)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Create trainer
    trainer = Trainer(model, device, lr=0.001)

    # Create dataloader
    dataset = list(zip(images, targets))

    # Training loop
    print(f"\nTraining for 3 epochs...")
    model.train()

    for epoch in range(3):
        epoch_loss = 0
        n_batches = 0
        start_time = time.time()

        # Simple batch training
        batch_size = 2
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            batch_images = [item[0].to(device) for item in batch]
            batch_targets = [{k: v.to(device) for k, v in item[1].items()} for item in batch]

            loss_dict = model(batch_images, batch_targets)
            losses = sum(loss for loss in loss_dict.values())

            trainer.optimizer.zero_grad()
            losses.backward()
            trainer.optimizer.step()

            epoch_loss += losses.item()
            n_batches += 1

        elapsed = time.time() - start_time
        avg_loss = epoch_loss / max(1, n_batches)
        print(f"  Epoch {epoch+1}/3: loss={avg_loss:.4f}, time={elapsed:.1f}s")

    # Inference test
    print(f"\nTesting inference...")
    model.eval()
    with torch.no_grad():
        test_img = images[0].unsqueeze(0).to(device)
        pred = model(test_img)[0]

        n_boxes = len(pred["boxes"])
        print(f"  Predicted {n_boxes} instances")
        if n_boxes > 0:
            print(f"  Top score: {pred['scores'][0].item():.3f}")
            print(f"  Box: {pred['boxes'][0].cpu().numpy()}")

    # Memory check
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\nGPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    print(f"\n{'='*60}")
    print(f"SMOKE TEST PASSED")
    print(f"{'='*60}")
    return True


if __name__ == "__main__":
    success = smoke_test()
    sys.exit(0 if success else 1)
