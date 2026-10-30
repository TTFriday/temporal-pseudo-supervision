"""
Plant-level stratified split for PhenoLeaf-TS
==============================================
For each genotype, shuffle replicates with fixed seed, then assign:
- val: max(1, floor(0.15*n + 0.5))
- test: max(1, floor(0.15*n + 0.5))
- train: rest

Also creates cross-genotype split for pressure test.
"""

import json
import random
import collections
from pathlib import Path
from datetime import datetime


def create_plant_splits(manifest_path, output_dir, seed=42):
    """Create plant-level stratified splits."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Group by (plant_id, replicate_id)
    replicates = collections.defaultdict(list)
    for entry in manifest:
        key = (entry["plant_id"], entry["replicate_id"])
        replicates[key].append(entry)

    # Group replicates by genotype
    genotypes = collections.defaultdict(list)
    for (plant_id, rep_id) in replicates:
        genotypes[plant_id].append(rep_id)

    rng = random.Random(seed)

    train_plants = []
    val_plants = []
    test_plants = []

    split_info = {}

    for geno in sorted(genotypes):
        reps = sorted(genotypes[geno])
        rng.shuffle(reps)

        n = len(reps)
        n_val = max(1, int(0.15 * n + 0.5))
        n_test = max(1, int(0.15 * n + 0.5))

        val_reps = reps[:n_val]
        test_reps = reps[n_val:n_val + n_test]
        train_reps = reps[n_val + n_test:]

        for rep_id in train_reps:
            train_plants.append((geno, rep_id))
        for rep_id in val_reps:
            val_plants.append((geno, rep_id))
        for rep_id in test_reps:
            test_plants.append((geno, rep_id))

        split_info[geno] = {
            "total_replicates": n,
            "train": len(train_reps),
            "val": len(val_reps),
            "test": len(test_reps),
        }

    # Build frame lists
    train_frames = []
    val_frames = []
    test_frames = []

    for plant_id, rep_id in train_plants:
        train_frames.extend(replicates[(plant_id, rep_id)])
    for plant_id, rep_id in val_plants:
        val_frames.extend(replicates[(plant_id, rep_id)])
    for plant_id, rep_id in test_plants:
        test_frames.extend(replicates[(plant_id, rep_id)])

    # Sort each by timestamp
    for frames in [train_frames, val_frames, test_frames]:
        frames.sort(key=lambda x: x.get("timestamp", ""))

    result = {
        "seed": seed,
        "created": datetime.now().isoformat(),
        "summary": {
            "train_plants": len(train_plants),
            "val_plants": len(val_plants),
            "test_plants": len(test_plants),
            "train_frames": len(train_frames),
            "val_frames": len(val_frames),
            "test_frames": len(test_frames),
        },
        "per_genotype": split_info,
        "train_plants": [list(p) for p in train_plants],
        "val_plants": [list(p) for p in val_plants],
        "test_plants": [list(p) for p in test_plants],
    }

    with open(output_dir / "plant_split.json", "w") as f:
        json.dump(result, f, indent=2)

    # Also save frame lists
    for name, frames in [("train", train_frames), ("val", val_frames), ("test", test_frames)]:
        with open(output_dir / f"{name}_frames.json", "w") as f:
            json.dump(frames, f, indent=2)

    print(f"Split created with seed={seed}")
    print(f"  Train: {len(train_plants)} plants, {len(train_frames)} frames")
    print(f"  Val:   {len(val_plants)} plants, {len(val_frames)} frames")
    print(f"  Test:  {len(test_plants)} plants, {len(test_frames)} frames")

    return result


def create_cross_genotype_split(manifest_path, output_dir, seed=42):
    """Create cross-genotype pressure test split.
    15 genotypes for train, 3 for val, 3 for test.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Group replicates by genotype
    genotypes = collections.defaultdict(set)
    for entry in manifest:
        genotypes[entry["plant_id"]].add(entry["replicate_id"])

    geno_list = sorted(genotypes.keys())
    # Sort by number of replicates (descending) for balanced assignment
    geno_by_size = sorted(geno_list, key=lambda g: len(genotypes[g]), reverse=True)

    rng = random.Random(seed + 1)  # Different seed for cross-genotype

    # Assign genotypes: 15 train, 3 val, 3 test
    # Try to balance by replicate count
    rng.shuffle(geno_by_size)

    # Simple round-robin assignment after shuffle
    train_genos = []
    val_genos = []
    test_genos = []

    for i, g in enumerate(geno_by_size):
        if i % 5 == 0 and len(val_genos) < 3:
            val_genos.append(g)
        elif i % 5 == 1 and len(test_genos) < 3:
            test_genos.append(g)
        else:
            train_genos.append(g)

    # Ensure exactly 15/3/3
    while len(train_genos) > 15:
        test_genos.append(train_genos.pop())
    while len(val_genos) > 3:
        train_genos.append(val_genos.pop())
    while len(test_genos) > 3:
        train_genos.append(val_genos.pop())

    # Build plant lists
    replicates = collections.defaultdict(list)
    for entry in manifest:
        key = (entry["plant_id"], entry["replicate_id"])
        replicates[key].append(entry)

    train_frames = []
    val_frames = []
    test_frames = []

    for g in train_genos:
        for rep_id in genotypes[g]:
            train_frames.extend(replicates[(g, rep_id)])
    for g in val_genos:
        for rep_id in genotypes[g]:
            val_frames.extend(replicates[(g, rep_id)])
    for g in test_genos:
        for rep_id in genotypes[g]:
            test_frames.extend(replicates[(g, rep_id)])

    result = {
        "seed": seed + 1,
        "created": datetime.now().isoformat(),
        "type": "cross_genotype",
        "train_genotypes": train_genos,
        "val_genotypes": val_genos,
        "test_genotypes": test_genos,
        "summary": {
            "train_genotypes": len(train_genos),
            "val_genotypes": len(val_genos),
            "test_genotypes": len(test_genos),
            "train_frames": len(train_frames),
            "val_frames": len(val_frames),
            "test_frames": len(test_frames),
        },
    }

    with open(output_dir / "cross_genotype_split.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nCross-genotype split:")
    print(f"  Train genotypes ({len(train_genos)}): {train_genos}")
    print(f"  Val genotypes ({len(val_genos)}): {val_genos}")
    print(f"  Test genotypes ({len(test_genos)}): {test_genos}")
    print(f"  Train frames: {len(train_frames)}")
    print(f"  Val frames: {len(val_frames)}")
    print(f"  Test frames: {len(test_frames)}")

    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data_audit" / "manifest.json"
    output_dir = root / "splits"

    result = create_plant_splits(manifest_path, output_dir, seed=42)
    cross_result = create_cross_genotype_split(manifest_path, output_dir, seed=42)
