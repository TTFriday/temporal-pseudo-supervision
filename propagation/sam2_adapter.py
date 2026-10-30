"""Frozen, CUDA-only SAM2 video propagation from one human-queried anchor.

Runtime dependencies are project-local; source .deps/sam2_env.sh in Git Bash.
The inspected SAM2 1.1.0 loader requires numerically named JPEGs, not PNG paths.
See sam2_provenance.json for source/checkpoint hashes and score limitations.
"""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import math
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch


class SAM2Propagator:
    """Propagate only supplied human masks, never labels or teacher predictions.

    ``checkpoint`` is a local checkpoint path; ``model_config`` is a SAM2 Hydra
    resource, e.g. ``configs/sam2.1/sam2.1_hiera_t.yaml``. No downloads occur here.
    ``cache_root`` holds temporary RGB-only JPEGs, removed even on failure.
    """

    def __init__(self, checkpoint, model_config, device="cuda", cache_root=None):
        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError("SAM2Propagator requires CUDA; CPU fallback is forbidden")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; SAM2 propagation cannot run")
        index = requested_device.index
        if index is None:
            index = torch.cuda.current_device()
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"Unavailable CUDA device: {requested_device}")
        self.device = torch.device("cuda", index)
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if not isinstance(model_config, str) or not model_config.strip():
            raise ValueError("model_config must name a packaged SAM2 Hydra config")
        self.model_config = model_config
        self.cache_root = (
            Path(cache_root).expanduser().resolve()
            if cache_root is not None
            else Path(__file__).resolve().parents[1] / ".deps" / "sam2_cache"
        )
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM2 dependencies are missing; source .deps/sam2_env.sh first"
            ) from exc

        # The builder appends fill_hole_area=8 when apply_postprocessing=True,
        # overriding an earlier zero. Preserve its other observed defaults but
        # explicitly disable native connected components (not built on Windows).
        overrides = [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=0",
        ]
        with torch.cuda.device(self.device):
            self.predictor = build_sam2_video_predictor(
                config_file=model_config,
                ckpt_path=str(self.checkpoint),
                device=str(self.device),
                mode="eval",
                hydra_overrides_extra=overrides,
                apply_postprocessing=False,
                vos_optimized=False,
            )
        self.predictor.eval()
        self.predictor.requires_grad_(False)

    @contextmanager
    def _inference_context(self):
        with torch.cuda.device(self.device), torch.inference_mode():
            # Unsupported bfloat16 means CUDA float32, never CPU inference.
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_bf16_supported(),
            ):
                yield

    @staticmethod
    def _timestamp(value):
        if isinstance(value, str):
            value = datetime.fromisoformat(
                value[:-1] + "+00:00" if value.endswith("Z") else value
            )
        if not isinstance(value, datetime):
            raise ValueError("timestamp must be an ISO-8601 string or datetime")
        aware = value.utcoffset() is not None
        # Naive timestamps must all use the same clock; interpret them as UTC.
        value = (
            value.astimezone(timezone.utc)
            if aware
            else value.replace(tzinfo=timezone.utc)
        )
        return value, aware

    @classmethod
    def _validate(cls, frames, anchor_key, anchor_masks, max_time_hours):
        try:
            limit = float(max_time_hours)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "max_time_hours must be finite and between 0 and 48"
            ) from exc
        if not math.isfinite(limit) or not 0 <= limit <= 48:
            raise ValueError("max_time_hours must be finite and between 0 and 48")
        frames = list(frames)
        if not frames:
            raise ValueError("frames must contain the queried anchor")
        names, times, awareness = [], [], []
        for frame in frames:
            if not isinstance(frame, dict) or set(frame) != {
                "filename",
                "rgb_path",
                "timestamp",
            }:
                raise ValueError(
                    "frames must contain only filename, rgb_path, timestamp; "
                    "GT/mask paths and teacher data are forbidden"
                )
            name = frame["filename"]
            if not isinstance(name, str) or not name:
                raise ValueError("filename must be a nonempty string")
            timestamp, aware = cls._timestamp(frame["timestamp"])
            names.append(name)
            times.append(timestamp)
            awareness.append(aware)
        if len(set(names)) != len(names):
            raise ValueError("frame filenames must be unique")
        if len(set(awareness)) != 1:
            raise ValueError("Do not mix timezone-aware and naive timestamps")
        if any(left > right for left, right in zip(times, times[1:])):
            raise ValueError("frames must be chronological by real timestamp")
        if anchor_key not in names:
            raise ValueError("anchor_key is not present in frames")
        if (
            not isinstance(anchor_masks, np.ndarray)
            or anchor_masks.dtype != np.uint8
            or anchor_masks.ndim != 3
            or any(size == 0 for size in anchor_masks.shape)
        ):
            raise ValueError("anchor_masks must be nonempty uint8[N,H,W]")
        if not np.all(np.any(anchor_masks != 0, axis=(1, 2))):
            raise ValueError("Each human anchor mask must contain foreground")
        anchor_time = times[names.index(anchor_key)]
        hours = [
            abs((timestamp - anchor_time).total_seconds()) / 3600 for timestamp in times
        ]
        # Prune before opening images or constructing memory: not a frame-count proxy.
        window = [(frame, age) for frame, age in zip(frames, hours) if age <= limit]
        anchor_index = next(
            i for i, (frame, _) in enumerate(window) if frame["filename"] == anchor_key
        )
        return window, anchor_index

    @staticmethod
    def _records(logits, obj_ids, masks_shape, anchor_key, direction, hours):
        count, height, width = masks_shape
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (
            count,
            1,
            height,
            width,
        ):
            raise RuntimeError("SAM2 must return mask logits with shape [N,1,H,W]")
        ids = list(obj_ids)
        if len(ids) != count or set(ids) != set(range(1, count + 1)):
            raise RuntimeError("SAM2 returned unexpected anchor-local object IDs")
        values = logits.detach().float().cpu()
        if not torch.isfinite(values).all().item():
            raise RuntimeError("SAM2 produced non-finite mask logits")
        records = []
        for obj_id in range(1, count + 1):
            plane = values[ids.index(obj_id), 0]
            foreground = plane > 0
            mask = foreground.numpy().astype(np.uint8)
            ys, xs = np.nonzero(mask)
            if len(xs):
                bbox = [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1),
                ]
                # A foreground mask-logit aggregate, NOT an IoU or probability of
                # correctness. Do not use this as an independent-anchor crosscheck.
                score = float(torch.sigmoid(plane[foreground]).mean().item())
            else:
                bbox, score = [0, 0, 0, 0], 0.0
            records.append(
                {
                    "mask": mask,
                    "bbox": bbox,
                    "score": min(1.0, max(0.0, score)),
                    "anchor": anchor_key,
                    "direction": direction,
                    "hours": float(hours),
                }
            )
        return records

    def propagate(self, frames, anchor_key, anchor_masks, max_time_hours=48):
        """Return ``{filename: [mask, bbox, score, anchor, direction, hours records]}``.

        Frames are chronological dicts with exactly filename, rgb_path, timestamp.
        Only RGB paths are read. Human uint8[N,H,W] masks use nonzero foreground;
        their supplied (already shuffled) order defines local IDs 1..N, never GT
        identity. All in-window images must share H,W with the anchor masks.

        Targets with absolute elapsed time <= max_time_hours (0..48) are included;
        the human anchor and out-of-window frames are omitted. Each target gets
        N records in input-mask order, including zero masks for absent objects
        (zero bbox/score). Bboxes are integer xywh with pixel-inclusive extents.
        Scores are mean sigmoid(logit) over positive-logit pixels, not calibrated
        probabilities. ``hours`` is nonnegative real elapsed time from this anchor.

        Fresh states are used for each direction and discarded after each call.
        Both directions are SAME-anchor predictions, not independent evidence.
        """
        window, anchor_index = self._validate(
            frames, anchor_key, anchor_masks, max_time_hours
        )
        self.cache_root.mkdir(parents=True, exist_ok=True)
        results = {}
        with tempfile.TemporaryDirectory(
            prefix="sam2_rgb_", dir=self.cache_root
        ) as directory:
            for index, (frame, _) in enumerate(window):
                with Image.open(
                    Path(frame["rgb_path"]).expanduser().resolve()
                ) as image:
                    if image.size != (anchor_masks.shape[2], anchor_masks.shape[1]):
                        raise ValueError("All RGB frames must match anchor_masks H,W")
                    image.convert("RGB").save(
                        Path(directory) / f"{index:06d}.jpg",
                        format="JPEG",
                        quality=100,
                        subsampling=0,
                    )
            if len(window) == 1:
                return results
            with self._inference_context():
                for reverse in (False, True):
                    distance = (
                        anchor_index if reverse else len(window) - anchor_index - 1
                    )
                    if distance == 0:
                        continue
                    direction = "backward" if reverse else "forward"
                    state = self.predictor.init_state(
                        video_path=directory,
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=True,
                        async_loading_frames=False,
                    )
                    try:
                        for obj_id, mask in enumerate(anchor_masks, start=1):
                            self.predictor.add_new_mask(
                                inference_state=state,
                                frame_idx=anchor_index,
                                obj_id=obj_id,
                                mask=(mask != 0),
                            )
                        expected = set(
                            range(anchor_index)
                            if reverse
                            else range(anchor_index + 1, len(window))
                        )
                        seen = set()
                        with closing(
                            self.predictor.propagate_in_video(
                                inference_state=state,
                                start_frame_idx=anchor_index,
                                max_frame_num_to_track=distance,
                                reverse=reverse,
                            )
                        ) as stream:
                            for index, obj_ids, logits in stream:
                                if index == anchor_index:
                                    continue  # Never re-label human input as pseudo-labels.
                                if index not in expected or index in seen:
                                    raise RuntimeError(
                                        "SAM2 returned an unexpected frame index"
                                    )
                                seen.add(index)
                                frame, hours = window[index]
                                results[frame["filename"]] = self._records(
                                    logits,
                                    obj_ids,
                                    anchor_masks.shape,
                                    anchor_key,
                                    direction,
                                    hours,
                                )
                        if seen != expected:
                            raise RuntimeError(
                                "SAM2 propagation did not return every requested target"
                            )
                    finally:
                        # Release masks, features and video memory, including on failure.
                        state.clear()
        return {
            frame["filename"]: results[frame["filename"]]
            for frame, _ in window
            if frame["filename"] in results
        }
