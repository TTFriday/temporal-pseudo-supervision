"""CPU contract tests plus an explicitly opt-in, three-frame CUDA smoke test."""

from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch

from propagation.sam2_adapter import SAM2Propagator


ROOT = Path(__file__).resolve().parents[1]


def make_frames(tmp_path, hours=(-24, 0, 24)):
    masks = np.zeros((2, 16, 24), dtype=np.uint8)
    masks[0, 3:8, 13:19] = 255
    masks[1, 2:7, 2:6] = 1
    frames = []
    for index, hour in enumerate(hours):
        path = tmp_path / f"rgb_{index}.png"
        Image.new("RGB", (24, 16), (20 + index, 80, 30)).save(path)
        frames.append(
            {
                "filename": path.name,
                "rgb_path": str(path),
                "timestamp": (datetime(2025, 1, 5) + timedelta(hours=hour)).isoformat(),
            }
        )
    return frames, masks


class FakePredictor:
    """Match the signatures and [N,1,H,W] outputs of the inspected source."""

    def __init__(self):
        self.states = []
        self.prompts = []
        self.runs = []
        self.folders = []
        self.fail = False
        self.drop_last = False

    def init_state(
        self,
        video_path,
        offload_video_to_cpu,
        offload_state_to_cpu,
        async_loading_frames,
    ):
        assert offload_video_to_cpu and offload_state_to_cpu
        assert not async_loading_frames
        files = sorted(Path(video_path).iterdir())
        assert [p.name for p in files] == [f"{i:06d}.jpg" for i in range(len(files))]
        for path in files:
            with Image.open(path) as image:
                assert image.format == "JPEG" and image.mode == "RGB"
        state = {"prompts": [], "num_frames": len(files)}
        self.states.append(state)
        self.folders.append(Path(video_path))
        return state

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        assert mask.dtype == np.bool_ and mask.shape == (16, 24)
        prompt = (frame_idx, obj_id, mask.copy())
        self.prompts.append(prompt)
        inference_state["prompts"].append(prompt)

    def propagate_in_video(
        self, inference_state, start_frame_idx, max_frame_num_to_track, reverse
    ):
        assert len(inference_state["prompts"]) == 2
        self.runs.append((start_frame_idx, max_frame_num_to_track, reverse))
        if self.fail:
            raise RuntimeError("synthetic inference failure")
        step = -1 if reverse else 1
        indices = list(
            range(
                start_frame_idx,
                start_frame_idx + step * (max_frame_num_to_track + 1),
                step,
            )
        )
        if self.drop_last:
            indices = indices[:-1]
        for index in indices:
            # Deliberately differ from input masks; return reversed local IDs.
            logits = torch.full((2, 1, 16, 24), -4.0)
            logits[1, 0, 5:9, 7:10] = 2.0
            yield index, [2, 1], logits


@pytest.fixture
def adapter(tmp_path):
    adapter = SAM2Propagator.__new__(SAM2Propagator)
    adapter.predictor = FakePredictor()
    adapter.cache_root = tmp_path / "rgb_cache"
    adapter._inference_context = nullcontext
    return adapter


def test_real_hours_bidirectional_schema_and_local_ids(adapter, tmp_path):
    frames, masks = make_frames(tmp_path, (-49, -48, -1.5, 0, 0.25, 48, 48.01))
    # Excluded RGB files need not exist and must never be opened.
    for index in (0, 6):
        frames[index]["rgb_path"] = str(tmp_path / "not_present.png")
    out = adapter.propagate(frames, frames[3]["filename"], masks)
    assert list(out) == [frames[i]["filename"] for i in (1, 2, 4, 5)]
    assert adapter.predictor.runs == [(2, 2, False), (2, 2, True)]
    assert len(adapter.predictor.states) == 2
    assert adapter.predictor.states[0] is not adapter.predictor.states[1]
    assert all(state == {} for state in adapter.predictor.states)
    assert all(not folder.exists() for folder in adapter.predictor.folders)
    assert not list(adapter.cache_root.iterdir())
    for offset, prompt in enumerate(adapter.predictor.prompts):
        assert prompt[0] == 2 and prompt[1] == offset % 2 + 1
        np.testing.assert_array_equal(prompt[2], masks[offset % 2] != 0)
    for index, hours in ((1, 48), (2, 1.5), (4, 0.25), (5, 48)):
        records = out[frames[index]["filename"]]
        assert len(records) == 2
        first, absent = records
        assert set(first) == {"mask", "bbox", "score", "anchor", "direction", "hours"}
        assert first["mask"].dtype == np.uint8 and first["mask"].shape == (16, 24)
        assert first["bbox"] == [7, 5, 3, 4]
        assert first["score"] == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())
        assert isinstance(first["score"], float) and 0 <= first["score"] <= 1
        assert first["anchor"] == frames[3]["filename"]
        assert first["direction"] == ("backward" if index < 3 else "forward")
        assert first["hours"] == hours and isinstance(first["hours"], float)
        assert not np.array_equal(first["mask"], masks[0] != 0)
        assert absent["bbox"] == [0, 0, 0, 0] and absent["score"] == 0
        assert not absent["mask"].any()


def test_fresh_states_across_anchors_and_calls(adapter, tmp_path):
    frames, masks = make_frames(tmp_path)
    adapter.propagate(frames, frames[0]["filename"], masks)
    adapter.propagate(frames, frames[2]["filename"], masks[::-1])
    assert adapter.predictor.runs == [(0, 2, False), (2, 2, True)]
    assert len({id(state) for state in adapter.predictor.states}) == 2
    np.testing.assert_array_equal(adapter.predictor.prompts[2][2], masks[1] != 0)


def test_subhour_limit_and_zero_window(adapter, tmp_path):
    frames, masks = make_frames(tmp_path, (-0.5, 0, 0.25))
    out = adapter.propagate(frames, frames[1]["filename"], masks, max_time_hours=0.25)
    assert list(out) == [frames[2]["filename"]]
    assert (
        adapter.propagate(frames, frames[1]["filename"], masks, max_time_hours=0) == {}
    )
    assert adapter.predictor.runs == [(0, 1, False)]


@pytest.mark.parametrize(
    "limit", [-1, 48.001, 100, float("nan"), float("inf"), "bad", None]
)
def test_invalid_time_limit_rejected(adapter, tmp_path, limit):
    frames, masks = make_frames(tmp_path)
    with pytest.raises(ValueError, match="max_time_hours"):
        adapter.propagate(frames, frames[1]["filename"], masks, limit)
    assert not adapter.predictor.states


@pytest.mark.parametrize(
    "extra", ["mask_path", "gt_id", "teacher_predictions", "instance_ids"]
)
def test_gt_and_teacher_metadata_rejected(adapter, tmp_path, extra):
    frames, masks = make_frames(tmp_path)
    frames[0][extra] = "must never read"
    with pytest.raises(ValueError, match="forbidden"):
        adapter.propagate(frames, frames[1]["filename"], masks)
    assert not adapter.predictor.states


@pytest.mark.parametrize(
    "problem",
    [
        "unsorted",
        "duplicate",
        "missing_anchor",
        "bad_time",
        "mixed_time",
        "empty_frames",
    ],
)
def test_bad_frame_metadata_rejected(adapter, tmp_path, problem):
    frames, masks = make_frames(tmp_path)
    anchor = frames[1]["filename"]
    if problem == "unsorted":
        frames.reverse()
    elif problem == "duplicate":
        frames[0]["filename"] = anchor
    elif problem == "missing_anchor":
        anchor = "absent.png"
    elif problem == "bad_time":
        frames[0]["timestamp"] = "unknown"
    elif problem == "mixed_time":
        frames[0]["timestamp"] += "+00:00"
    else:
        frames = []
    with pytest.raises(ValueError):
        adapter.propagate(frames, anchor, masks)
    assert not adapter.predictor.states


def test_timezone_offsets_use_elapsed_time(adapter, tmp_path):
    frames, masks = make_frames(tmp_path)
    frames[0]["timestamp"] = "2025-01-01T00:00:00Z"
    frames[1]["timestamp"] = datetime(
        2025, 1, 1, 4, tzinfo=timezone(timedelta(hours=2))
    )
    frames[2]["timestamp"] = "2025-01-01T05:00:00+02:00"
    out = adapter.propagate(frames, frames[1]["filename"], masks, 2)
    assert out[frames[0]["filename"]][0]["hours"] == 2
    assert out[frames[2]["filename"]][0]["hours"] == 1


@pytest.mark.parametrize(
    "problem", ["dtype", "dimensions", "empty_batch", "empty_object", "size"]
)
def test_invalid_anchor_masks_rejected(adapter, tmp_path, problem):
    frames, masks = make_frames(tmp_path)
    if problem == "dtype":
        masks = masks.astype(np.float32)
    elif problem == "dimensions":
        masks = masks[0]
    elif problem == "empty_batch":
        masks = masks[:0]
    elif problem == "empty_object":
        masks[0] = 0
    else:
        masks = masks[:, :, :20]
    with pytest.raises(ValueError):
        adapter.propagate(frames, frames[1]["filename"], masks)
    assert not adapter.predictor.states


@pytest.mark.parametrize("failure", ["exception", "missing_frame"])
def test_failures_release_state_and_jpegs(adapter, tmp_path, failure):
    frames, masks = make_frames(tmp_path)
    adapter.predictor.fail = failure == "exception"
    adapter.predictor.drop_last = failure == "missing_frame"
    with pytest.raises(RuntimeError):
        adapter.propagate(frames, frames[1]["filename"], masks)
    assert all(state == {} for state in adapter.predictor.states)
    assert not list(adapter.cache_root.iterdir())


@pytest.mark.parametrize("problem", ["shape", "ids", "nan"])
def test_invalid_predictor_outputs_fail_closed(problem):
    logits = torch.zeros(2, 1, 16, 24)
    ids = [1, 2]
    if problem == "shape":
        logits = logits[:, 0]
    elif problem == "ids":
        ids = [1, 1]
    else:
        logits[0, 0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError):
        SAM2Propagator._records(logits, ids, (2, 16, 24), "anchor.png", "forward", 1.0)


def test_cuda_is_mandatory(monkeypatch):
    with pytest.raises(ValueError, match="CUDA"):
        SAM2Propagator("missing.pt", "config", device="cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        SAM2Propagator("missing.pt", "config")


def test_builder_freezes_model_and_disables_native_holes(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    model = torch.nn.Linear(2, 2)
    calls = []

    def builder(**kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setitem(
        sys.modules,
        "sam2.build_sam",
        SimpleNamespace(build_sam2_video_predictor=builder),
    )
    checkpoint = tmp_path / "dummy.pt"
    checkpoint.write_bytes(b"mock builder only")
    adapter = SAM2Propagator(
        checkpoint, "configs/sam2.1/sam2.1_hiera_t.yaml", cache_root=tmp_path
    )
    assert adapter.predictor is model and not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert calls[0]["device"] == "cuda:0" and calls[0]["mode"] == "eval"
    assert calls[0]["apply_postprocessing"] is False
    assert "++model.fill_hole_area=0" in calls[0]["hydra_overrides_extra"]
    with pytest.raises(ValueError, match="Unavailable CUDA"):
        SAM2Propagator(checkpoint, "config", device="cuda:1")
    with pytest.raises(FileNotFoundError):
        SAM2Propagator(tmp_path / "absent.pt", "config")


@pytest.mark.skipif(
    os.environ.get("SAM2_RUN_GPU_SMOKE") != "1", reason="explicitly opt-in GPU smoke"
)
def test_gpu_smoke_three_synthetic_frames(tmp_path, monkeypatch):
    assert torch.cuda.is_available(), "GPU smoke requested but CUDA is unavailable"
    height, width = 192, 256
    frames = []
    masks = np.zeros((2, height, width), dtype=np.uint8)
    for index in range(3):
        image = Image.new("RGB", (width, height), (24, 27, 30))
        draw = ImageDraw.Draw(image)
        boxes = [
            (40 + index * 3, 45, 108 + index * 3, 132),
            (155, 55 + index * 2, 215, 138 + index * 2),
        ]
        for box, color in zip(boxes, [(60, 180, 65), (160, 215, 75)]):
            draw.ellipse(box, fill=color)
        path = tmp_path / f"synthetic_rgb_{index}.png"
        image.save(path)
        frames.append(
            {
                "filename": path.name,
                "rgb_path": str(path),
                "timestamp": f"2025-01-01T0{index}:00:00Z",
            }
        )
        if index == 1:
            # Only the human-anchor analogue has masks; local order is shuffled.
            for obj, box in enumerate(reversed(boxes)):
                plane = Image.new("L", (width, height), 0)
                ImageDraw.Draw(plane).ellipse(box, fill=1)
                masks[obj] = np.asarray(plane)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    adapter = SAM2Propagator(
        ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt",
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        cache_root=tmp_path / "video_cache",
    )
    built = time.perf_counter()
    assert adapter.predictor.fill_hole_area == 0
    assert not adapter.predictor.training
    assert all(
        not p.requires_grad and p.device.type == "cuda"
        for p in adapter.predictor.parameters()
    )
    loader_observations, logit_observations, inference_observations = [], [], []
    init_state = adapter.predictor.init_state
    propagate = adapter.predictor.propagate_in_video
    run_frame = adapter.predictor._run_single_frame_inference

    def observed_init(**kwargs):
        state = init_state(**kwargs)
        loader_observations.append(
            {
                "shape": list(state["images"].shape),
                "dtype": str(state["images"].dtype),
                "video_storage_device": str(state["images"].device),
                "compute_device": str(state["device"]),
            }
        )
        return state

    def observed_propagation(**kwargs):
        for index, ids, logits in propagate(**kwargs):
            logit_observations.append(
                {
                    "frame_index": index,
                    "object_ids": list(ids),
                    "shape": list(logits.shape),
                    "dtype": str(logits.dtype),
                    "device": str(logits.device),
                    "reverse": kwargs["reverse"],
                }
            )
            yield index, ids, logits

    def observed_frame(**kwargs):
        inference_observations.append(
            {
                "frame_index": kwargs["frame_idx"],
                "reverse": kwargs["reverse"],
                "is_init_cond_frame": kwargs["is_init_cond_frame"],
                "run_mem_encoder": kwargs["run_mem_encoder"],
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return run_frame(**kwargs)

    monkeypatch.setattr(adapter.predictor, "init_state", observed_init)
    monkeypatch.setattr(adapter.predictor, "propagate_in_video", observed_propagation)
    monkeypatch.setattr(
        adapter.predictor, "_run_single_frame_inference", observed_frame
    )
    out = adapter.propagate(frames, frames[1]["filename"], masks)
    torch.cuda.synchronize()
    finished = time.perf_counter()
    assert set(out) == {frames[0]["filename"], frames[2]["filename"]}
    assert len(loader_observations) == 2 and len(logit_observations) == 4
    assert {
        entry["reverse"] for entry in inference_observations if entry["run_mem_encoder"]
    } == {False, True}
    assert all(not entry["grad_enabled"] for entry in inference_observations)
    for filename, records in out.items():
        assert len(records) == 2
        for record in records:
            assert (
                record["mask"].shape == (height, width)
                and record["mask"].dtype == np.uint8
            )
            assert record["hours"] == 1.0 and 0.0 <= record["score"] <= 1.0
            assert record["direction"] == (
                "backward" if filename == frames[0]["filename"] else "forward"
            )
    metrics = {
        "status": "passed",
        "synthetic_only": True,
        "training": False,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "input_rgb_shape": [3, height, width, 3],
        "human_anchor_masks_shape": list(masks.shape),
        "loader_observations": loader_observations,
        "logit_observations": logit_observations,
        "inference_observations": inference_observations,
        "build_seconds": built - started,
        "propagation_seconds": finished - built,
        "total_seconds": finished - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "output": {
            filename: [
                {
                    key: (list(value.shape) if key == "mask" else value)
                    for key, value in record.items()
                }
                for record in records
            ]
            for filename, records in out.items()
        },
        "limitation": "Shape/device/memory smoke only; not biological accuracy or independent-anchor validation.",
    }
    (ROOT / ".deps" / "sam2_gpu_smoke.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
