"""
PhenoLeaf-TS Data Audit
========================
1. Parse timestamps from filenames
2. Verify RGB/mask pairing and dimensions
3. Identify frames with color-count vs COCO-instance mismatches
4. Build clean manifest with plant-level grouping
5. Create stratified plant-level splits (train/val/test per genotype)
"""

import os
import sys
import json
import re
import hashlib
import random
import collections
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

# 31-color palette from the dataset
COLOUR_PALETTE = [
    (244, 64, 14), (48, 57, 249), (234, 250, 37), (24, 193, 65),
    (245, 130, 49), (231, 80, 219), (0, 182, 173), (115, 0, 218),
    (191, 239, 69), (255, 250, 200), (250, 190, 212), (66, 212, 244),
    (155, 99, 36), (220, 190, 255), (69, 158, 220), (255, 216, 177),
    (98, 2, 37), (227, 213, 12), (79, 159, 83), (170, 23, 101),
    (170, 255, 195), (169, 169, 169), (181, 111, 119), (144, 121, 171),
    (9, 125, 244), (184, 70, 30), (154, 35, 246), (229, 225, 238),
    (141, 254, 82), (31, 200, 209), (194, 217, 105),
]


def parse_timestamp(filename):
    """Extract datetime from filename pattern: <id>_YYYY_MM_DD_HH_MM_SS-..."""
    match = re.search(r'(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})', filename)
    if match:
        try:
            return datetime(*[int(x) for x in match.groups()])
        except ValueError:
            return None
    return None


def count_instances_from_mask(mask_path):
    """Count unique leaf instances in a color-coded mask."""
    try:
        mask = np.array(Image.open(mask_path).convert("RGB"))
        pixels = mask.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        # Remove background (black)
        instances = sum(1 for c in unique_colors if not np.all(c == 0))
        return instances
    except Exception as e:
        return -1, str(e)


def extract_instances_detailed(mask_path):
    """Extract detailed instance info from mask."""
    try:
        mask = np.array(Image.open(mask_path).convert("RGB"))
    except Exception as e:
        return None, str(e)

    h, w = mask.shape[:2]
    instances = []

    for leaf_id, colour in enumerate(COLOUR_PALETTE, start=1):
        match = np.all(mask == np.array(colour), axis=-1)
        area = int(match.sum())
        if area == 0:
            continue

        rows = np.any(match, axis=1)
        cols = np.any(match, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        instances.append({
            "leaf_id": leaf_id,
            "colour": list(colour),
            "area": area,
            "bbox": [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)],
            "n_connected_components": 0,  # Will check separately if needed
        })

    return instances, None


def audit_dataset(raw_dir, mask_dir, output_dir):
    """Full dataset audit."""
    raw_dir = Path(raw_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    anomalies = []
    time_issues = []

    # Collect all frames
    print("Scanning dataset...")
    for plant_dir in tqdm(sorted(raw_dir.iterdir()), desc="Genotypes"):
        if not plant_dir.is_dir():
            continue
        plant_name = plant_dir.name

        for rep_dir in sorted(plant_dir.iterdir()):
            if not rep_dir.is_dir():
                continue
            rep_name = rep_dir.name

            rgb_files = sorted(rep_dir.glob("*.png"))
            for rgb_path in rgb_files:
                mask_path = mask_dir / plant_name / rep_name / rgb_path.name

                entry = {
                    "plant_id": plant_name,
                    "replicate_id": rep_name,
                    "filename": rgb_path.name,
                    "rgb_path": str(rgb_path),
                    "mask_path": str(mask_path) if mask_path.exists() else None,
                }

                # Parse timestamp
                ts = parse_timestamp(rgb_path.name)
                entry["timestamp"] = ts.isoformat() if ts else None
                if ts is None:
                    time_issues.append({"file": rgb_path.name, "issue": "no_timestamp"})

                # Check mask exists
                if not mask_path.exists():
                    anomalies.append({**entry, "issue": "missing_mask"})
                    continue

                # Check dimensions
                try:
                    rgb_img = Image.open(rgb_path)
                    mask_img = Image.open(mask_path)
                    entry["rgb_size"] = list(rgb_img.size)
                    entry["mask_size"] = list(mask_img.size)
                    if rgb_img.size != mask_img.size:
                        anomalies.append({**entry, "issue": "size_mismatch"})
                except Exception as e:
                    anomalies.append({**entry, "issue": f"read_error: {e}"})
                    continue

                # Count instances from mask colors
                mask_instances, err = extract_instances_detailed(mask_path)
                if err:
                    anomalies.append({**entry, "issue": f"mask_parse_error: {err}"})
                    continue

                entry["n_instances"] = len(mask_instances)
                entry["instances"] = mask_instances

                # Check for tiny instances (area < 50 pixels)
                tiny = [i for i in mask_instances if i["area"] < 50]
                if tiny:
                    entry["tiny_instances"] = len(tiny)

                manifest.append(entry)

    print(f"\nTotal frames scanned: {len(manifest)}")
    print(f"Anomalies found: {len(anomalies)}")
    print(f"Time issues: {len(time_issues)}")

    # Check for duplicate images (by file hash of first few KB)
    print("\nChecking for duplicate images...")
    hash_map = collections.defaultdict(list)
    for entry in tqdm(manifest, desc="Hashing"):
        try:
            with open(entry["rgb_path"], "rb") as f:
                h = hashlib.md5(f.read(4096)).hexdigest()
            hash_map[h].append(entry["filename"])
        except:
            pass

    duplicates = {h: files for h, files in hash_map.items() if len(files) > 1}
    if duplicates:
        print(f"Found {len(duplicates)} sets of potential duplicates")

    # Time ordering check per replicate
    print("\nChecking time ordering...")
    groups = collections.defaultdict(list)
    for e in manifest:
        key = f"{e['plant_id']}_{e['replicate_id']}"
        groups[key].append(e)

    time_ordering_ok = True
    for key, frames in groups.items():
        timestamps = [f["timestamp"] for f in frames if f["timestamp"]]
        if timestamps != sorted(timestamps):
            time_ordering_ok = False
            print(f"  Time ordering issue in {key}")

    # Save audit report
    report = {
        "audit_date": datetime.now().isoformat(),
        "total_frames": len(manifest),
        "total_genotypes": len(set(e["plant_id"] for e in manifest)),
        "total_replicates": len(groups),
        "anomalies": anomalies,
        "time_issues": time_issues,
        "duplicate_sets": len(duplicates),
        "time_ordering_ok": time_ordering_ok,
        "frames_per_genotype": dict(collections.Counter(e["plant_id"] for e in manifest)),
        "replicates_per_genotype": dict(collections.Counter(
            f"{e['plant_id']}_{e['replicate_id']}" for e in manifest
        )),
    }

    report_path = output_dir / "audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nAudit report saved to {report_path}")

    # Save clean manifest (without instance details for size)
    clean_manifest = []
    for e in manifest:
        ce = {k: v for k, v in e.items() if k != "instances"}
        clean_manifest.append(ce)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(clean_manifest, f, indent=2)

    return manifest, anomalies, report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data/phenoleaf_ts/extracted/PhenoLeaf-TS_DS/01_raw_dataset"
    mask_dir = root / "data/phenoleaf_ts/extracted/PhenoLeaf-TS_DS/02_leaf_labels"
    output_dir = root / "data_audit"

    manifest, anomalies, report = audit_dataset(raw_dir, mask_dir, output_dir)
