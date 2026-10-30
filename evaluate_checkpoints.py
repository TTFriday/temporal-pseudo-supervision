"""
Evaluate saved pilot checkpoints
=================================
Run COCO evaluation on saved teacher/student checkpoints.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from run_pilot import (
    load_manifest, load_split, extract_instances_from_mask,
    ResizeAndPad, get_model
)

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "data_audit" / "manifest.json"
SPLIT_PATH = PROJECT_ROOT / "splits" / "plant_split.json"
PILOT_DIR = PROJECT_ROOT / "pilot_results"


def evaluate_checkpoint(ckpt_path, test_frame_keys, frames_by_plant, device, target_size=512):
    """Evaluate a single checkpoint."""
    print(f"\nEvaluating {ckpt_path.name}...")
    
    # Load model
    model = get_model(num_classes=2, pretrained_backbone=True)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
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
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "n_gt": len(coco_annotations_gt), "n_pred": len(coco_predictions)}
    
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
        "n_gt": len(coco_annotations_gt),
        "n_pred": len(coco_predictions),
    }
    
    # Cleanup
    os.remove(tmp_gt)
    
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading manifest and splits...")
    manifest, frames_by_plant = load_manifest(MANIFEST_PATH)
    split = load_split(SPLIT_PATH)
    
    # Get test frames
    test_keys = []
    for plant in split["test_plants"]:
        pid, rid = plant[0], plant[1]
        for frame in frames_by_plant.get((pid, rid), []):
            test_keys.append((pid, rid, frame["filename"]))
    
    print(f"Test frames: {len(test_keys)}")
    
    # Evaluate all checkpoints
    all_results = {}
    
    for run_dir in sorted(PILOT_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        
        print(f"\n{'='*60}")
        print(f"Run: {run_dir.name}")
        print(f"{'='*60}")
        
        run_results = {}
        
        for ckpt_name in ["teacher_seed.pth", "teacher_full.pth", "student.pth"]:
            ckpt_path = run_dir / ckpt_name
            if ckpt_path.exists():
                results = evaluate_checkpoint(ckpt_path, test_keys, frames_by_plant, device)
                run_results[ckpt_name.replace(".pth", "")] = results
                print(f"  {ckpt_name}: AP={results['AP']:.4f}, AP50={results['AP50']:.4f}")
        
        # Save results
        with open(run_dir / "results.json", "w") as f:
            json.dump(run_results, f, indent=2)
        
        all_results[run_dir.name] = run_results
    
    # Save summary
    with open(PILOT_DIR / "results_final.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for run_name, results in all_results.items():
        print(f"\n{run_name}:")
        for ckpt_name, metrics in results.items():
            print(f"  {ckpt_name}: AP={metrics['AP']:.4f}, AP50={metrics['AP50']:.4f}, AP75={metrics['AP75']:.4f}")
    
    print(f"\nResults saved to {PILOT_DIR / 'results_final.json'}")


if __name__ == "__main__":
    main()
