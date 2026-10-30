"""Tiny synthetic regressions; no original dataset access or GPU dependencies."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

MODULE_PATH = Path(__file__).resolve().parents[1] / "data_audit/check_coco_consistency.py"
SPEC = importlib.util.spec_from_file_location("data_audit_consistency", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_full_identity_not_basename(self):
        a = audit.coco_frame_key({"plant_id": "A", "replicate_id": "rep_01", "file_name": "same.png"})
        b = audit.coco_frame_key({"plant_id": "B", "replicate_id": "rep_01", "file_name": "same.png"})
        self.assertNotEqual(a, b)
        self.assertEqual(a, audit.coco_frame_key({"plant_id": "A", "replicate_id": "rep_01",
                                                "file_name": "D:\\raw\\A\\rep_01\\same.png"}))
        for name in ("../same.png", "B/rep_01/same.png", "A/rep_02/same.png"):
            with self.assertRaises(ValueError):
                audit.coco_frame_key({"plant_id": "A", "replicate_id": "rep_01", "file_name": name})
        with self.assertRaises(ValueError):
            audit.frame_key("A/other", "rep_01", "same.png")

    def test_timestamp_seconds_stop_at_hyphen(self):
        self.assertEqual(audit.parse_timestamp("123_2022_08_29_12_03_34-7-108.png"), "2022-08-29T12:03:34")
        for name in ("123_2022_02_30_12_03_34-x.png", "prefix2022_08_29_12_03_34-x.png"):
            with self.assertRaises(ValueError):
                audit.parse_timestamp(name)

    def test_exact_palette_signed_unknown_and_shades(self):
        color = np.array(audit.COLOUR_PALETTE[0], dtype=np.int32)
        rgb = np.array([[color, color - 1, color + 1, [0, 0, 0]]], dtype=np.uint8)
        labels, unknown = audit.decode_palette(rgb)
        self.assertEqual(labels.tolist(), [[1, -1, -1, 0]])
        self.assertEqual([u["nearest_palette_linf"] for u in unknown], [1, 1])
        self.assertEqual(len(audit.instances_from_labels(labels)), 1)
        # A negative difference involving a zero channel must not wrap to 255.
        rgb = np.array([[[0, 181, 173], [255, 182, 173]]], dtype=np.uint8)
        labels, unknown = audit.decode_palette(rgb)
        self.assertEqual(labels.tolist(), [[-1, -1]])
        self.assertEqual(unknown[0]["nearest_palette_linf"], 1)
        with self.assertRaises(ValueError):
            audit.analyze_mask("never-opened.png", tolerance=5)
        with self.assertRaises(ValueError):
            audit.decode_palette(np.zeros((2, 2), dtype=np.uint8))

    def test_all_palette_ids_are_stable_and_lossless(self):
        rgb = np.array([[(0, 0, 0), *audit.COLOUR_PALETTE]], dtype=np.uint8)
        labels, unknown = audit.decode_palette(rgb)
        self.assertEqual(labels.tolist(), [list(range(32))])
        self.assertEqual(unknown, [])
        instances = audit.instances_from_labels(labels)
        self.assertEqual([i["leaf_id"] for i in instances], list(range(1, 32)))
        self.assertTrue(all(i["area"] == 1 for i in instances))
        for instance in instances:
            decoded = audit.mask_utils.decode(instance["rle"])
            self.assertTrue(np.array_equal(decoded, labels == instance["leaf_id"]))

    def test_tiny_and_disconnected_are_one_preserved_instance(self):
        labels = np.zeros((5, 6), dtype=np.int16)
        labels[0, 0] = labels[4, 5] = 1
        labels[2, 3] = 2
        instances = audit.instances_from_labels(labels)
        self.assertEqual([i["area"] for i in instances], [2, 1])
        self.assertEqual(instances[0]["bbox_xywh"], [0, 0, 6, 5])
        self.assertEqual(instances[0]["official_bbox_xywh"], [0, 0, 5, 4])
        self.assertEqual(instances[1]["bbox_xywh"], [3, 2, 1, 1])
        self.assertEqual(instances[1]["official_bbox_xywh"], [3, 2, 0, 0])
        annotations = [dict(leaf_id=i["leaf_id"], area=i["area"], bbox=i["official_bbox_xywh"],
                            segmentation=i["rle"], category_id=1, iscrowd=0) for i in instances]
        problems, differences = audit.compare_annotations(labels, instances, annotations)
        self.assertEqual(problems, [])
        self.assertEqual(differences, 2)
        annotations[0]["segmentation"] = instances[1]["rle"]
        annotations[0]["area"] = 100
        annotations[0]["bbox"] = [0, 0, 99, 99]
        problems, _ = audit.compare_annotations(labels, instances, annotations)
        self.assertEqual({p["reason"] for p in problems},
                         {"coco_area_mismatch", "coco_official_bbox_mismatch", "coco_rle_pixel_mismatch"})

    def test_missing_duplicate_leaf_and_invalid_rle(self):
        labels = np.ones((2, 2), dtype=np.int16)
        instances = audit.instances_from_labels(labels)
        problems, _ = audit.compare_annotations(labels, instances, [])
        self.assertEqual(problems[0]["reason"], "coco_leaf_identity_mismatch")
        ann = dict(leaf_id=1, area=4, bbox=[0, 0, 1, 1], category_id=1, iscrowd=0,
                   segmentation={"size": [1, 1], "counts": "bad"})
        problems, _ = audit.compare_annotations(labels, instances, [ann])
        self.assertEqual(problems[0]["reason"], "coco_rle_invalid")
        problems, _ = audit.compare_annotations(labels, instances, [ann, ann])
        self.assertEqual(problems[0]["reason"], "coco_leaf_identity_mismatch")

    def test_json_native_atomic_success_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            audit.atomic_json_dump({"color": (np.uint8(1), np.uint8(2)),
                                    "area": np.int64(2), "ok": np.bool_(True)}, path)
            self.assertEqual(json.loads(path.read_text()), {"color": [1, 2], "area": 2, "ok": True})
            original = path.read_bytes()
            for invalid in ({"unsupported": object()}, {"nonfinite": float("nan")}):
                with self.assertRaises((TypeError, ValueError)):
                    audit.atomic_json_dump(invalid, path)
                self.assertEqual(path.read_bytes(), original)
            with patch.object(audit.os, "replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    audit.atomic_json_dump({"new": 1}, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_safe_projection_has_no_ground_truth(self):
        report = {"algorithm_facing": {"valid_frame_keys": ["A/rep_01/a.png"],
                                      "excluded_frame_keys": ["B/rep_01/b.png"], "n_instances": 55},
                  "audit_only": {"area": 100, "unknown_colors": [[1, 2, 3]]},
                  "summary": {"n_instances": 55}}
        safe = audit.algorithm_facing_result(report)
        self.assertEqual(set(safe), {"valid_frame_keys", "excluded_frame_keys"})
        safe["valid_frame_keys"].clear()
        self.assertEqual(len(report["algorithm_facing"]["valid_frame_keys"]), 1)

    def test_split_scoped_ids_and_orphans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "standard").mkdir()
            for split, plant in (("train", "A"), ("val", "B"), ("test", "C")):
                value = {"images": [{"id": 1, "plant_id": plant, "replicate_id": "rep_01",
                                     "file_name": "same.png"}],
                         "annotations": [{"id": 1, "image_id": 1, "leaf_id": 1}]}
                if split == "test":
                    value["annotations"].append({"id": 2, "image_id": 999, "leaf_id": 2})
                (root / "standard" / f"{split}.json").write_text(json.dumps(value))
            frames, problems, _, _ = audit.load_coco(root)
            self.assertEqual(len(frames), 3)
            self.assertEqual(frames["B/rep_01/same.png"][0]["split"], "val")
            self.assertEqual(problems, [{"reason": "orphan_coco_annotations", "split": "test", "id": 999}])
            (root / "standard/test.json").unlink()
            with self.assertRaises(FileNotFoundError):
                audit.load_coco(root)

    def test_one_frame_pairing_unknown_and_size_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "123_2022_08_29_12_03_34-7-RGB1_pot_C1_A-01.png"
            key = audit.frame_key("A", "rep_01", name)
            rgb_path = root / "01_raw_dataset" / key
            mask_path = root / "02_leaf_labels" / key
            rgb_path.parent.mkdir(parents=True)
            mask_path.parent.mkdir(parents=True)
            rgb = np.zeros((4, 4, 3), dtype=np.uint8)
            mask = rgb.copy()
            mask[1, 1] = audit.COLOUR_PALETTE[0]
            Image.fromarray(rgb).save(rgb_path)
            Image.fromarray(mask).save(mask_path)
            entry = dict(plant_id="A", replicate_id="rep_01", filename=name,
                         rgb_path=str(rgb_path), mask_path=str(mask_path),
                         timestamp="2022-08-29T12:03:34", rgb_size=[4, 4], mask_size=[4, 4], n_instances=1)
            labels, _ = audit.decode_palette(mask)
            instance = audit.instances_from_labels(labels)[0]
            record = {"split": "train", "image": {"id": 1, "width": 4, "height": 4},
                      "annotations": [dict(leaf_id=1, category_id=1, iscrowd=0, area=1,
                                           bbox=instance["official_bbox_xywh"], segmentation=instance["rle"])]}
            result = audit.audit_frame((key, [entry], [record], root))
            self.assertEqual(result["problems"], [])
            self.assertEqual(result["gt_statistics"]["instances_below_10"], 1)
            # Nonblack unknowns are never silently assigned to background or snapped.
            mask[2, 2] = [1, 2, 3]
            Image.fromarray(mask).save(mask_path)
            result = audit.audit_frame((key, [entry], [record], root))
            self.assertEqual([p["reason"] for p in result["problems"]],
                             ["unknown_nonblack_colors_require_adjudication"])
            Image.fromarray(rgb[:2]).save(rgb_path)
            result = audit.audit_frame((key, [entry], [record], root))
            self.assertIn("rgb_mask_size_mismatch", {p["reason"] for p in result["problems"]})

    def test_full_synthetic_audit_and_fail_closed_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            coco = root / "coco"
            (coco / "standard").mkdir(parents=True)
            manifest = []
            documents = {s: {"images": [], "annotations": []} for s in ("train", "val", "test")}
            for index, plant in enumerate(("A", "B"), 1):
                name = f"123_2022_08_29_12_03_34-7-RGB1_pot_C1_{plant}-01.png"
                key = audit.frame_key(plant, "rep_01", name)
                rgb_path = raw / "01_raw_dataset" / key
                mask_path = raw / "02_leaf_labels" / key
                rgb_path.parent.mkdir(parents=True)
                mask_path.parent.mkdir(parents=True)
                rgb = np.full((3, 4, 3), index, dtype=np.uint8)
                mask = np.zeros_like(rgb)
                mask[1, 1] = audit.COLOUR_PALETTE[0]
                if plant == "B":
                    mask[2, 2] = [1, 2, 3]
                Image.fromarray(rgb).save(rgb_path)
                Image.fromarray(mask).save(mask_path)
                manifest.append(dict(plant_id=plant, replicate_id="rep_01", filename=name,
                                     rgb_path=str(rgb_path), mask_path=str(mask_path),
                                     timestamp="2022-08-29T12:03:34", rgb_size=[4, 3],
                                     mask_size=[4, 3], n_instances=1))
                instance = audit.instances_from_labels(audit.decode_palette(mask)[0])[0]
                documents["train"]["images"].append(dict(id=index, plant_id=plant, replicate_id="rep_01",
                                                        file_name=name, width=4, height=3))
                documents["train"]["annotations"].append(dict(id=index, image_id=index, leaf_id=1,
                                                             category_id=1, iscrowd=0, area=1,
                                                             bbox=instance["official_bbox_xywh"],
                                                             segmentation=instance["rle"]))
            for split, document in documents.items():
                (coco / "standard" / f"{split}.json").write_text(json.dumps(document))
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            output = root / "repair_audit_20261025.json"
            evidence = {"converter": audit.fingerprint(MODULE_PATH), "archived_evidence": []}
            with patch.object(audit, "source_evidence", return_value=evidence), patch("builtins.print"):
                report = audit.check_coco_consistency(manifest_path, coco, output, raw_root=raw,
                                                      workers=1, expected_frames=2)
            self.assertEqual(report["summary"]["valid_frames"], 1)
            self.assertEqual(report["summary"]["excluded_frames"], 1)
            self.assertEqual(report["summary"]["exact_palette_coco_count_mismatch_frames"], 0)
            self.assertEqual(report["summary"]["standard_bbox_convention_differences"], 2)
            self.assertEqual(report["summary"]["instances_below_10"], 2)
            self.assertTrue(report["provenance"]["inputs_unchanged"])
            self.assertEqual(audit.read_algorithm_facing_result(output), report["algorithm_facing"])
            with patch.object(audit, "source_evidence", return_value=evidence), patch("builtins.print"):
                report = audit.check_coco_consistency(manifest_path, coco, output, raw_root=raw,
                                                      workers=1, expected_frames=3)
            self.assertEqual(report["algorithm_facing"]["valid_frame_keys"], [])
            self.assertEqual(report["summary"]["global_problem_count"], 1)

    def test_directory_identity_does_not_assume_filename_rep_number(self):
        frame = {"frame_key": "Col-0/rep_41/123_2022_05_11_12_00_00-x_Col-0-01.png",
                 "problems": [{"reason": "filename_plant_replicate_mismatch"}], "differences": []}
        report = {"audit_only": {"frame_checks": [frame], "global_problems": []},
                  "policy": {}, "summary": {}}
        audit.refresh_identity_decisions(report)
        self.assertEqual(report["algorithm_facing"]["valid_frame_keys"], [frame["frame_key"]])
        self.assertEqual(report["summary"]["filename_replicate_alias_frames"], 1)
        self.assertEqual(frame["filename_replicate_token"], "01")
        # An alias must not pardon a real manifest path disagreement or unknowns.
        frame["problems"].append({"reason": "manifest_path_identity_mismatch"})
        audit.refresh_identity_decisions(report)
        self.assertEqual(report["algorithm_facing"]["valid_frame_keys"], [])
        self.assertEqual(len(frame["differences"]), 1)  # Idempotent review.
        # Mixing filename tokens in a single authoritative directory needs review.
        other = {"frame_key": "Col-0/rep_41/123_2022_05_12_12_00_00-x_Col-0-02.png",
                 "problems": [], "differences": []}
        report["audit_only"]["frame_checks"].append(other)
        audit.refresh_identity_decisions(report)
        self.assertEqual(report["algorithm_facing"]["valid_frame_keys"], [])
        self.assertIn("inconsistent_filename_tokens_within_directory",
                      {p["reason"] for p in other["problems"]})

    def test_output_cannot_overwrite_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text("[]")
            with self.assertRaises(ValueError):
                audit.check_coco_consistency(path, Path(directory), path)
            self.assertEqual(path.read_text(), "[]")


if __name__ == "__main__":
    unittest.main()
