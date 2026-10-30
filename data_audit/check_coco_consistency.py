"""Read-only PhenoLeaf audit; never use audit-only GT statistics for selection.

Identity is genotype/replicate/filename, NOT basename or a split-local image id.
Admissible target policy: exact RGB palette, black background, one instance per
palette id (including disconnected pixels), retain every positive-area instance.
Unknown nonblack pixels require human adjudication; never snap or erase them.
The official converter stores max-min boxes; COCO/runner boxes derived from RLE
must instead use max-min+1. Area thresholds 10/50 are NOT converter rules.

Only read_algorithm_facing_result() should cross into selection code. It returns
keys/exclusions only. This audit neither edits manifests nor launches training.
"""

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

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
PALETTE = np.asarray(COLOUR_PALETTE, dtype=np.int32)
PALETTE_CODES = (PALETTE[:, 0] << 16) | (PALETTE[:, 1] << 8) | PALETTE[:, 2]
CODE_TO_ID = {int(code): index + 1 for index, code in enumerate(PALETTE_CODES)}
TIMESTAMP_RE = re.compile(r"^\d+_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})-")
WORKSPACE = Path(__file__).resolve().parents[1]
DATA_HOME = WORKSPACE / "data" / "phenoleaf_ts"
RAW_ROOT = DATA_HOME / "extracted/PhenoLeaf-TS_DS"
CONVERTER = DATA_HOME / "PhenoLeaf-TS-main/phenoleaf_ts/data/prepare.py"


def frame_key(plant_id, replicate_id, filename):
    """Reject path traversal and ambiguous identities rather than guessing."""
    parts = (plant_id, replicate_id, filename)
    if any(not isinstance(p, str) or not p or p in (".", "..")
           or any(c in p for c in ("/", "\\", ":")) for p in parts):
        raise ValueError(f"Invalid frame identity: {parts!r}")
    return "/".join(parts)


def coco_frame_key(image):
    """Accept basename or a consistent fully qualified dataset-relative path."""
    value = image["file_name"].replace("\\", "/")
    parts = PurePosixPath(value).parts
    if ".." in parts:
        raise ValueError(f"Unsafe COCO path: {value}")
    key = frame_key(image["plant_id"], image["replicate_id"], parts[-1])
    if len(parts) > 1 and tuple(parts[-3:]) != tuple(key.split("/")):
        raise ValueError(f"COCO path disagrees with identity: {value}")
    return key


def parse_timestamp(filename):
    match = TIMESTAMP_RE.match(filename)
    if not match:
        raise ValueError(f"Missing timestamp: {filename}")
    return datetime(*(int(v) for v in match.groups())).isoformat()


def decode_palette(rgb):
    """Exact signed-integer decoding: background=0, unknown=-1, leaves=1..31.

    Near-colour distances are diagnostic ONLY, never an assignment policy.
    Packing RGB as int32 avoids both uint8 subtraction wrap and costly Nx3 sort.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an HxWx3 uint8 RGB mask")
    signed = rgb.astype(np.int32)
    packed = (signed[..., 0] << 16) | (signed[..., 1] << 8) | signed[..., 2]
    codes, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    mapping = np.asarray([0 if c == 0 else CODE_TO_ID.get(int(c), -1)
                          for c in codes], dtype=np.int16)
    labels = mapping[inverse].reshape(rgb.shape[:2])
    unknown = []
    for code, count in zip(codes[mapping == -1], counts[mapping == -1]):
        color = np.asarray([int(code) >> 16, (int(code) >> 8) & 255,
                            int(code) & 255], dtype=np.int32)
        distances = np.max(np.abs(PALETTE - color), axis=1)
        nearest = int(distances.min())
        ys, xs = np.where(packed == code)
        unknown.append({
            "rgb": color.tolist(), "pixels": int(count),
            "bbox_xywh": [int(xs.min()), int(ys.min()),
                          int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
            "nearest_palette_linf": nearest,
            "nearest_leaf_ids": (np.flatnonzero(distances == nearest) + 1).tolist(),
            "within_legacy_tolerance_5": bool(nearest <= 5),
        })
    return labels, unknown


def instances_from_labels(labels):
    """All positive-area colours; never split components or drop small leaves."""
    instances = []
    for leaf_id in np.unique(labels):
        if leaf_id <= 0:
            continue
        binary = np.asfortranarray(labels == leaf_id, dtype=np.uint8)
        rle = mask_utils.encode(binary)
        box = mask_utils.toBbox(rle).astype(int).tolist()
        instances.append({
            "leaf_id": int(leaf_id), "area": int(mask_utils.area(rle)),
            "bbox_xywh": box,
            "official_bbox_xywh": [box[0], box[1], box[2] - 1, box[3] - 1],
            "rle": {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")},
        })
    return instances


def analyze_mask(mask_path, tolerance=0):
    """Audit-only API; tolerant decoding is deliberately not supported."""
    if tolerance != 0:
        raise ValueError("Only exact palette decoding (tolerance=0) is admissible")
    with Image.open(mask_path) as image:
        labels, unknown = decode_palette(np.asarray(image.convert("RGB")))
    instances = instances_from_labels(labels)
    return {"n_palette_colors": len(instances), "n_unknown_colors": len(unknown),
            "unknown_colors": unknown, "instances": instances,
            "tiny_instances": [{k: v for k, v in i.items() if k != "rle"}
                               for i in instances if i["area"] < 50]}


def compare_annotations(labels, instances, annotations):
    """Compare every leaf id, decoded pixel, area and both bbox conventions."""
    problems = []
    expected = {i["leaf_id"]: i for i in instances}
    actual = defaultdict(list)
    standard_bbox_differences = 0
    for annotation in annotations:
        actual[annotation.get("leaf_id")].append(annotation)
    for leaf_id in sorted(set(expected) | set(actual), key=str):
        candidates = actual.get(leaf_id, [])
        if leaf_id not in expected or len(candidates) != 1:
            problems.append({"reason": "coco_leaf_identity_mismatch", "leaf_id": leaf_id,
                             "expected": int(leaf_id in expected), "actual": len(candidates)})
            continue
        instance = expected[leaf_id]
        annotation = candidates[0]
        if annotation.get("area") != instance["area"]:
            problems.append({"reason": "coco_area_mismatch", "leaf_id": leaf_id})
        if annotation.get("bbox") != instance["official_bbox_xywh"]:
            problems.append({"reason": "coco_official_bbox_mismatch", "leaf_id": leaf_id})
        if annotation.get("bbox") != instance["bbox_xywh"]:
            standard_bbox_differences += 1
        if annotation.get("category_id") != 1 or annotation.get("iscrowd") != 0:
            problems.append({"reason": "coco_category_or_crowd_mismatch", "leaf_id": leaf_id})
        try:
            segmentation = annotation["segmentation"]
            if not isinstance(segmentation, dict) or segmentation.get("size") != list(labels.shape):
                raise ValueError("RLE dimensions/type disagree")
            decoded = mask_utils.decode(segmentation)
            if not np.array_equal(decoded, labels == leaf_id):
                problems.append({"reason": "coco_rle_pixel_mismatch", "leaf_id": leaf_id})
            encoded = instance["rle"]
            if not np.array_equal(mask_utils.decode(encoded), labels == leaf_id):
                raise ValueError("Exact parser RLE round trip failed")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            problems.append({"reason": "coco_rle_invalid", "leaf_id": leaf_id, "error": str(exc)})
    return problems, standard_bbox_differences


def json_native(value):
    if isinstance(value, np.generic):
        return json_native(value.item())
    if isinstance(value, np.ndarray):
        return json_native(value.tolist())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): json_native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_native(v) for v in value]
    return value


def atomic_json_dump(value, output_path):
    """Serialize before opening anything; same-directory replace on success only."""
    payload = json.dumps(json_native(value), indent=2, allow_nan=False) + "\n"
    output_path = Path(output_path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=output_path.parent, prefix=output_path.name + ".",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def fingerprint(path):
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": path.resolve().as_posix(), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def load_coco(coco_dir):
    frames = defaultdict(list)
    problems = []
    sources = []
    totals = {}
    for split in ("train", "val", "test"):
        path = Path(coco_dir) / "standard" / f"{split}.json"
        sources.append(fingerprint(path))  # Missing splits are fatal, not silently skipped.
        coco = json.loads(path.read_text(encoding="utf-8"))
        images = defaultdict(list)
        annotations = defaultdict(list)
        ann_ids = Counter()
        for image in coco["images"]:
            images[image["id"]].append(image)
        for ann in coco["annotations"]:
            annotations[ann["image_id"]].append(ann)
            ann_ids[ann["id"]] += 1
        for image_id, items in images.items():
            if len(items) != 1:
                problems.append({"reason": "duplicate_coco_image_id", "split": split, "id": image_id})
            for image in items:
                key = coco_frame_key(image)
                frames[key].append({"split": split, "image": image,
                                    "annotations": annotations.get(image_id, [])})
        for image_id in set(annotations) - set(images):
            problems.append({"reason": "orphan_coco_annotations", "split": split, "id": image_id})
        for ann_id, count in ann_ids.items():
            if count > 1:
                problems.append({"reason": "duplicate_coco_annotation_id", "split": split, "id": ann_id})
        totals[split] = {"images": len(coco["images"]), "annotations": len(coco["annotations"])}
    return frames, problems, sources, totals


def read_verified_image(path):
    before = path.stat()
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        mode = image.mode
        pixels = np.asarray(image.convert("RGB"))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed while reading: {path}")
    return pixels, {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                    "mtime_ns": before.st_mtime_ns, "mode": mode}


def audit_frame(job):
    key, entries, records, root = job
    plant, replicate, filename = key.split("/")
    rgb_path = root / "01_raw_dataset" / key
    mask_path = root / "02_leaf_labels" / key
    result = {"frame_key": key, "rgb_path": rgb_path.as_posix(), "mask_path": mask_path.as_posix(),
              "problems": [], "differences": []}
    problems = result["problems"]
    if len(entries) != 1:
        problems.append({"reason": "manifest_identity_missing_or_duplicate", "count": len(entries)})
    if len(records) != 1:
        problems.append({"reason": "coco_identity_missing_or_duplicate", "count": len(records)})
    try:
        timestamp = parse_timestamp(filename)
        result["timestamp"] = timestamp
        # Directory identity is authoritative, as in prepare.py:159-171.
        # Filename replicate tokens are checked across the directory later.
        rgb, result["rgb_file"] = read_verified_image(rgb_path)
        mask, result["mask_file"] = read_verified_image(mask_path)
        result["rgb_size"] = [int(rgb.shape[1]), int(rgb.shape[0])]
        result["mask_size"] = [int(mask.shape[1]), int(mask.shape[0])]
        result["rgb_pixel_sha256"] = hashlib.sha256(
            str(rgb.shape).encode("ascii") + rgb.tobytes()).hexdigest()
        if result["rgb_size"] != result["mask_size"]:
            problems.append({"reason": "rgb_mask_size_mismatch"})
        if result["mask_file"]["mode"] not in ("RGB", "P"):
            problems.append({"reason": "unsupported_mask_mode", "mode": result["mask_file"]["mode"]})
        for entry in entries:
            for field, expected in (("timestamp", timestamp), ("rgb_size", result["rgb_size"]),
                                    ("mask_size", result["mask_size"])):
                if entry.get(field) != expected:
                    problems.append({"reason": "manifest_metadata_mismatch", "field": field})
            for field, expected in (("rgb_path", rgb_path), ("mask_path", mask_path)):
                if not entry.get(field) or Path(entry[field]).resolve() != expected.resolve():
                    problems.append({"reason": "manifest_path_identity_mismatch", "field": field})
        labels, unknown = decode_palette(mask)
        instances = instances_from_labels(labels)
        exact_count = len(instances)
        near_shades = sum(u["within_legacy_tolerance_5"] for u in unknown)
        # In NumPy 1.26, old np.uint8 minus Python int promotes to a signed int.
        # Record both signed and uint8-wrapped scenarios, without using either.
        wrapped_near = 0
        for u in unknown:
            delta = (np.asarray(u["rgb"], dtype=np.int32) - PALETTE) % 256
            wrapped_near += int(np.any(np.max(delta, axis=1) <= 5))
        stats = {"exact_instances": exact_count, "coco_instances": None,
                 "legacy_signed_tolerance_count": exact_count + near_shades,
                 "legacy_wrapped_tolerance_count": exact_count + wrapped_near,
                 "unknown_colors": unknown,
                 "instances_below_10": sum(i["area"] < 10 for i in instances),
                 "instances_below_50": sum(i["area"] < 50 for i in instances),
                 "standard_bbox_convention_differences": 0,
                 "minimum_instance_area": min((i["area"] for i in instances), default=None)}
        if unknown:
            problems.append({"reason": "unknown_nonblack_colors_require_adjudication"})
            result["differences"].append("official_converter_ignores_unknown_colors")
        if near_shades:
            result["differences"].append("legacy_tolerance_counts_shades_as_extra_instances")
        if stats["instances_below_10"]:
            result["differences"].append("archived_pilot_area_lt10_filter_not_official")
        if stats["instances_below_50"]:
            result["differences"].append("archived_pipeline_area_lt50_filter_not_official")
        for entry in entries:
            if entry.get("n_instances") != exact_count:
                problems.append({"reason": "manifest_exact_count_mismatch"})
        if len(records) == 1:
            record = records[0]
            image = record["image"]
            result["official_split"] = record["split"]
            result["official_image_id"] = image["id"]
            stats["coco_instances"] = len(record["annotations"])
            if [image["width"], image["height"]] != result["rgb_size"]:
                problems.append({"reason": "coco_image_size_mismatch"})
            if stats["coco_instances"] != exact_count:
                problems.append({"reason": "exact_palette_coco_count_mismatch"})
            comparison, box_count = compare_annotations(labels, instances, record["annotations"])
            problems.extend(comparison)
            stats["standard_bbox_convention_differences"] = box_count
        result["gt_statistics"] = stats
        if problems or result["differences"]:
            result["instance_details"] = [{k: v for k, v in i.items() if k != "rle"} for i in instances]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        problems.append({"reason": "frame_read_or_parse_error", "error": str(exc)})
    return result


def source_evidence(converter_path, archive_dir):
    """Inspect source as data (AST); do not execute preparation/training code."""
    source = Path(converter_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    palette = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "COLOUR_PALETTE" for t in n.targets))
    if palette != COLOUR_PALETTE:
        raise ValueError("Audit palette disagrees with official converter")
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    extractor = ast.get_source_segment(source, functions["extract_instances_from_mask"])
    converter = ast.get_source_segment(source, functions["convert_to_coco"])
    if ('if match.sum() == 0:' not in extractor or
            'int(cmax - cmin), int(rmax - rmin)' not in extractor or
            any(isinstance(n, ast.Compare) and any(isinstance(op, (ast.Lt, ast.LtE)) for op in n.ops)
                for n in ast.walk(functions["extract_instances_from_mask"]))):
        raise ValueError("Official converter policy changed; review before auditing")
    result = {"converter": fingerprint(converter_path), "extractor_source": extractor,
              "coco_converter_source": converter, "minimum_area_pixels": 1,
              "archived_evidence": []}
    zip_path = Path(converter_path).parents[2].with_suffix(".zip")
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            member = "PhenoLeaf-TS-main/phenoleaf_ts/data/prepare.py"
            original = archive.read(member)
        result["converter_archive"] = {"path": zip_path.as_posix(), "member": member,
                                        "member_sha256": hashlib.sha256(original).hexdigest(),
                                        "matches_working_source": original.decode("utf-8") == source}
    if archive_dir is not None:
        for relative in ("data_audit/check_coco_consistency.py", "data_audit/coco_issues.json",
                         "data_audit/manifest.json", "data_audit/audit_report.json",
                         "run_pilot.py", "training/pipeline.py"):
            path = Path(archive_dir) / relative
            item = fingerprint(path)
            if relative.endswith("coco_issues.json"):
                text = path.read_text(encoding="utf-8")
                try:
                    json.loads(text)
                    item["valid_json"] = True
                except json.JSONDecodeError as exc:
                    item.update(valid_json=False, parse_error=str(exc), surviving_text=text)
            result["archived_evidence"].append(item)
    result["old_audit_diagnosis"] = {
        "basename_matching": "Unsafe generally; actual collisions checked separately.",
        "tolerance": "Counts distinct shades, not unique palette IDs; not official policy.",
        "unsigned_arithmetic": "Explicit int32 decoding avoids promotion-dependent subtraction.",
        "truncation": "Old json.dump opens destination before failing on numpy.uint8 RGB values.",
        "archived_pilot": "run_pilot.py:230 drops area<10; not an official converter rule.",
        "archived_pipeline": "training/pipeline.py:190 drops area<50; not an official converter rule.",
        "historical_26": "No complete historical issue list survives; reproduced counts below are evidence, not assumed identities.",
    }
    return result


def refresh_identity_decisions(report):
    """Recheck recorded identities without repeating pixel/COCO comparisons.

    prepare.py uses directory names, not the terminal filename replicate token.
    A stable token differing from that directory is metadata, not a bad label or
    permission to merge plants. Path/manifest/COCO disagreements remain excluded.
    This also removes the superseded strict-suffix finding from an initial audit.
    """
    frames = report["audit_only"]["frame_checks"]
    groups = defaultdict(list)
    identity_reasons = {"filename_plant_replicate_mismatch", "filename_genotype_or_replicate_format_mismatch",
                        "inconsistent_filename_tokens_within_directory"}
    alias_reason = "filename_replicate_token_differs_from_authoritative_directory"
    for result in frames:
        plant, replicate, filename = result["frame_key"].split("/")
        result["problems"] = [p for p in result["problems"] if p["reason"] not in identity_reasons]
        result["differences"] = [d for d in result["differences"] if d != alias_reason]
        match = re.search(rf"_{re.escape(plant)}-(\d+)\.png$", filename)
        token = match.group(1) if match else None
        result["filename_replicate_token"] = token
        if token is None or not re.fullmatch(r"rep_\d+", replicate):
            result["problems"].append({"reason": "filename_genotype_or_replicate_format_mismatch"})
        groups[(plant, replicate)].append(result)
    aliases = []
    for (plant, replicate), items in sorted(groups.items()):
        tokens = {r["filename_replicate_token"] for r in items}
        if len(tokens) != 1 or None in tokens:
            for result in items:
                result["problems"].append({"reason": "inconsistent_filename_tokens_within_directory"})
        elif replicate != "rep_" + next(iter(tokens)):
            aliases.append({"plant_id": plant, "directory_replicate_id": replicate,
                            "filename_replicate_token": next(iter(tokens)), "frames": len(items)})
            for result in items:
                result["differences"].append(alias_reason)
    global_problems = report["audit_only"]["global_problems"]
    excluded = [r["frame_key"] for r in frames if r["problems"] or global_problems]
    excluded_set = set(excluded)
    report["algorithm_facing"] = {
        "valid_frame_keys": [r["frame_key"] for r in frames if r["frame_key"] not in excluded_set],
        "excluded_frame_keys": excluded,
    }
    report["summary"].update(
        valid_frames=len(frames) - len(excluded), excluded_frames=len(excluded),
        filename_replicate_alias_frames=sum(a["frames"] for a in aliases),
        filename_replicate_alias_directories=len(aliases),
        reason_counts=dict(Counter(p["reason"] for r in frames for p in r["problems"])))
    report["policy"]["replicate_identity"] = (
        "Directory plant/replicate plus filename, verified against manifest and COCO; "
        "a stable terminal filename replicate token is not a directory id. Never merge aliases.")
    report["audit_only"]["filename_replicate_aliases"] = aliases
    report["audit_only"]["per_file_differences"] = [
        {k: r[k] for k in ("frame_key", "rgb_path", "mask_path", "problems", "differences",
                          "gt_statistics", "instance_details") if k in r}
        for r in frames if r["problems"] or r["differences"]]
    report["audit_only"]["unresolved_human_adjudications"] = [
        {"frame_key": r["frame_key"], "reasons": sorted({p["reason"] for p in r["problems"]}),
         "required_action": "Review original RGB/mask and provenance; no automatic pixel repair."}
        for r in frames if r["problems"]]
    return report


def algorithm_facing_result(report):
    """Whitelist keys only. Never return GT counts, areas, colours, or reasons."""
    safe = report["algorithm_facing"]
    return {name: list(safe[name]) for name in ("valid_frame_keys", "excluded_frame_keys")}


def read_algorithm_facing_result(path):
    with Path(path).open(encoding="utf-8") as handle:
        return algorithm_facing_result(json.load(handle))


def check_coco_consistency(manifest_path, coco_dir, output_path, *, raw_root=RAW_ROOT,
                           converter_path=CONVERTER, archive_dir=None, workers=4,
                           expected_frames=17082):
    start = time.monotonic()
    raw_root = Path(raw_root).resolve()
    output_path = Path(output_path).resolve()
    # Fail closed if asked to overwrite any source, raw tree, split or old report.
    if output_path.parent != Path(manifest_path).resolve().parent or output_path.name != "repair_audit_20261025.json":
        raise ValueError("Output must be data_audit/repair_audit_20261025.json beside the manifest")
    if output_path.is_relative_to(raw_root):
        raise ValueError("Audit output cannot be inside the raw dataset")
    evidence = source_evidence(converter_path, archive_dir)
    manifest_source = fingerprint(manifest_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = defaultdict(list)
    for entry in manifest:
        entries[frame_key(entry["plant_id"], entry["replicate_id"], entry["filename"])].append(entry)
    coco, global_problems, coco_sources, split_counts = load_coco(coco_dir)
    raw_keys = {p.relative_to(raw_root / "01_raw_dataset").as_posix()
                for p in (raw_root / "01_raw_dataset").rglob("*.png")}
    mask_keys = {p.relative_to(raw_root / "02_leaf_labels").as_posix()
                 for p in (raw_root / "02_leaf_labels").rglob("*.png")}
    keys = sorted(set(entries) | set(coco) | raw_keys | mask_keys)
    for key in keys:
        if len(key.split("/")) != 3:
            raise ValueError(f"Unexpected raw directory layout: {key}")
        frame_key(*key.split("/"))
    if expected_frames is not None and any(n != expected_frames for n in
            (len(manifest), len(coco), len(raw_keys), len(mask_keys))):
        global_problems.append({"reason": "unexpected_dataset_coverage", "expected_frames": expected_frames})
    jobs = ((key, entries.get(key, []), coco.get(key, []), raw_root) for key in keys)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(audit_frame, jobs), 1):
            results.append(result)
            if index % 1000 == 0:
                print(f"Audited {index}/{len(keys)} frames ({time.monotonic() - start:.1f}s)", flush=True)
    timestamps = defaultdict(list)
    rgb_hashes = defaultdict(list)
    for result in results:
        if "timestamp" in result:
            timestamps[(result["frame_key"].rsplit("/", 1)[0], result["timestamp"])].append(result["frame_key"])
        if "rgb_pixel_sha256" in result:
            rgb_hashes[result["rgb_pixel_sha256"]].append(result["frame_key"])
    duplicate_times = [v for v in timestamps.values() if len(v) > 1]
    duplicate_rgb = [v for v in rgb_hashes.values() if len(v) > 1]
    duplicates = {k for group in duplicate_times + duplicate_rgb for k in group}
    for result in results:
        if result["frame_key"] in duplicates:
            result["problems"].append({"reason": "duplicate_timestamp_or_rgb_requires_adjudication"})
    # A second stat-only pass detects concurrent changes without decoding again.
    changed = []
    for result in results:
        for kind in ("rgb", "mask"):
            if kind + "_file" not in result:
                continue
            path = Path(result[kind + "_path"])
            try:
                stat = path.stat()
                old = result[kind + "_file"]
                if (stat.st_size, stat.st_mtime_ns) != (old["bytes"], old["mtime_ns"]):
                    changed.append(path.as_posix())
            except OSError:
                changed.append(path.as_posix())
    protected_sources = [manifest_source, *coco_sources, evidence["converter"], *evidence["archived_evidence"]]
    changed.extend(s["path"] for s in protected_sources if fingerprint(s["path"])["sha256"] != s["sha256"])
    if changed:
        global_problems.append({"reason": "inputs_changed_during_audit", "paths": changed})
    exclusions = [r["frame_key"] for r in results if r["problems"] or global_problems]
    excluded_set = set(exclusions)
    safe = {"valid_frame_keys": [k for k in keys if k not in excluded_set], "excluded_frame_keys": exclusions}
    stats = [r["gt_statistics"] for r in results if "gt_statistics" in r]
    reason_counts = Counter(p["reason"] for r in results for p in r["problems"])
    summary = {
        "frames_checked": len(results), "frames_decoded": len(stats),
        "valid_frames": len(safe["valid_frame_keys"]), "excluded_frames": len(exclusions),
        "exact_palette_coco_count_mismatch_frames": reason_counts["exact_palette_coco_count_mismatch"],
        "coco_annotation_problem_count": sum(n for reason, n in reason_counts.items()
                                             if reason.startswith("coco_") and reason != "coco_identity_missing_or_duplicate"),
        "legacy_signed_tolerance_count_mismatch_frames": sum(s["legacy_signed_tolerance_count"] != s["coco_instances"] for s in stats),
        "legacy_wrapped_tolerance_count_mismatch_frames": sum(s["legacy_wrapped_tolerance_count"] != s["coco_instances"] for s in stats),
        "unknown_color_frames": sum(bool(s["unknown_colors"]) for s in stats),
        "unknown_pixels": sum(u["pixels"] for s in stats for u in s["unknown_colors"]),
        "frames_with_instances_below_10": sum(s["instances_below_10"] > 0 for s in stats),
        "instances_below_10": sum(s["instances_below_10"] for s in stats),
        "frames_with_instances_below_50": sum(s["instances_below_50"] > 0 for s in stats),
        "instances_below_50": sum(s["instances_below_50"] for s in stats),
        "standard_bbox_convention_differences": sum(s["standard_bbox_convention_differences"] for s in stats),
        "reason_counts": dict(reason_counts), "global_problem_count": len(global_problems),
    }
    differences = [{k: r[k] for k in ("frame_key", "rgb_path", "mask_path", "problems", "differences",
                                     "gt_statistics", "instance_details") if k in r}
                   for r in results if r["problems"] or r["differences"]]
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "scope": "Read-only CPU audit. No training, raw-mask repair, split, manifest or result changes.",
        "policy": {
            "key": "plant_id/replicate_id/filename; image_id is scoped to its COCO split",
            "palette_rgb": COLOUR_PALETTE, "arithmetic": "signed int32; exact equality; tolerance 0",
            "background_rgb": [0, 0, 0], "minimum_instance_area": 1,
            "identity": "One palette id per instance across all components; do not relabel disconnected regions.",
            "unknown_colors": "Exclude whole frame pending human review; do not snap, ignore or erase unknown pixels.",
            "small_regions": "Retain all exact palette pixels; area<10/50 is not proof of a bad label.",
            "official_bbox": "[xmin,ymin,xmax-xmin,ymax-ymin]; source compatibility only, nonstandard COCO extent",
            "admissible_runner_bbox": "RLE-derived [xmin,ymin,xmax-xmin+1,ymax-ymin+1]; xyxy uses exclusive max+1",
            "temporal_order": "Sort parsed naive capture datetime within plant/replicate; timezone not encoded.",
            "selection_boundary": "Consume only read_algorithm_facing_result(); never feed audit_only or summary GT to selection.",
            "exclusion_caveat": "Freeze these data-validity exclusions before selection; no model-difficulty filtering.",
        },
        "provenance": {"raw_root": raw_root.as_posix(), "manifest": manifest_source,
                       "official_coco": coco_sources, "source_evidence": evidence,
                       "audit_code": fingerprint(Path(__file__)), "numpy_version": np.__version__,
                       "inputs_unchanged": not changed},
        "coverage": {"manifest_entries": len(manifest), "manifest_unique_keys": len(entries),
                     "raw_rgb_files": len(raw_keys), "raw_mask_files": len(mask_keys),
                     "coco_unique_keys": len(coco), "official_splits": split_counts,
                     "basename_collision_count": len(keys) - len({k.rsplit('/', 1)[-1] for k in keys}),
                     "raw_rgb_without_mask": sorted(raw_keys - mask_keys),
                     "raw_mask_without_rgb": sorted(mask_keys - raw_keys)},
        "summary": summary, "algorithm_facing": safe,
        "audit_only": {"warning": "GT-bearing quality-control evidence. Not a selection feature source.",
                       "frame_checks": results, "per_file_differences": differences,
                       "duplicate_capture_times": duplicate_times, "duplicate_rgb_pixel_groups": duplicate_rgb,
                       "global_problems": global_problems,
                       "unresolved_human_adjudications": [
                           {"frame_key": r["frame_key"], "reasons": sorted({p["reason"] for p in r["problems"]}),
                            "required_action": "Review original RGB/mask and provenance; no automatic pixel repair."}
                           for r in results if r["problems"]]},
    }
    refresh_identity_decisions(report)
    atomic_json_dump(report, output_path)
    print(json.dumps(summary, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=WORKSPACE / "data_audit/manifest.json")
    parser.add_argument("--coco-dir", type=Path, default=DATA_HOME / "prepared/coco")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--converter", type=Path, default=CONVERTER)
    parser.add_argument("--archive-dir", type=Path, default=WORKSPACE / "archives/pre_repair_20261025")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "data_audit/repair_audit_20261025.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    check_coco_consistency(args.manifest, args.coco_dir, args.output, raw_root=args.raw_root,
                           converter_path=args.converter, archive_dir=args.archive_dir, workers=args.workers)


if __name__ == "__main__":
    main()
