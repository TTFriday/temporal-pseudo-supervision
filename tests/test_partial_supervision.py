"""CPU tensor regressions; full Mask R-CNN tests require PHENOLEAF_RUN_CUDA_TESTS=1."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torchvision.models.detection._utils import (
    BalancedPositiveNegativeSampler,
    BoxCoder,
    Matcher,
)
from torchvision.models.detection.roi_heads import RoIHeads, maskrcnn_loss, project_masks_on_boxes
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from training.maskrcnn_model import compute_loss_with_ignore
from training.partial_supervision import (
    _partial_mask_loss,
    _partial_rpn_loss,
    _positive_box_loss,
    _positive_matches,
    _prepare_pseudo_target,
    _select_positive_proposals,
    _transform_pseudo,
    _weighted_mask_bce,
    partial_maskrcnn_loss,
)


def _target(q=(1.0, 1.0), size=16):
    return {
        "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0], [9.0, 9.0, 13.0, 13.0]]),
        "labels": torch.tensor([1, 2]),
        "masks": torch.zeros(2, size, size, dtype=torch.uint8),
        "confidence": torch.tensor(q),
        "is_pseudo": True,
    }


def _rpn():
    return SimpleNamespace(
        proposal_matcher=Matcher(0.7, 0.3, allow_low_quality_matches=True),
        fg_bg_sampler=BalancedPositiveNegativeSampler(16, 0.5),
        box_coder=BoxCoder((1.0, 1.0, 1.0, 1.0)),
    )


def _roi():
    return SimpleNamespace(
        proposal_matcher=Matcher(0.5, 0.5, allow_low_quality_matches=False),
        fg_bg_sampler=BalancedPositiveNegativeSampler(16, 0.5),
        box_coder=BoxCoder((10.0, 10.0, 5.0, 5.0)),
        add_gt_proposals=lambda proposals, boxes: RoIHeads.add_gt_proposals(None, proposals, boxes),
    )


def test_rpn_unmatched_and_rejected_gt_have_no_gradient():
    target = _prepare_pseudo_target(_target((1.0, 0.0)), torch.zeros(3, 16, 16))
    anchors = torch.tensor([
        [1.0, 1.0, 5.0, 5.0],    # accepted GT
        [9.0, 9.0, 13.0, 13.0],  # rejected GT is not background
        [1.0, 1.0, 7.0, 7.0],    # between IoU thresholds
        [20.0, 20.0, 24.0, 24.0],
    ])
    matches = _positive_matches(anchors, target["boxes"], _rpn().proposal_matcher)
    assert matches.tolist() == [0, -1, -1, -1]
    logits = torch.zeros(4, requires_grad=True)
    deltas = torch.ones(4, 4, requires_grad=True)
    objectness, regression = _partial_rpn_loss(_rpn(), logits, deltas, [anchors], [target])
    torch.testing.assert_close(objectness, F.softplus(torch.tensor(0.0)))
    (objectness + regression).backward()
    assert logits.grad[0] < 0
    assert torch.count_nonzero(logits.grad[1:]) == 0
    assert torch.count_nonzero(deltas.grad[1:]) == 0
    assert torch.count_nonzero(deltas.grad[0]) == 4


def test_rpn_low_quality_match_requires_some_overlap():
    matcher = _rpn().proposal_matcher
    gt = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    candidates = torch.tensor([[0.0, 0.0, 8.0, 8.0], [20.0, 20.0, 24.0, 24.0]])
    assert _positive_matches(candidates, gt, matcher).tolist() == [0, -1]
    assert _positive_matches(candidates[1:], gt, matcher).tolist() == [-1]
    assert _positive_matches(candidates, gt[:0], matcher).tolist() == [-1, -1]
    assert _positive_matches(candidates[:0], gt, matcher).numel() == 0


def test_rpn_confidence_weights_each_matched_gt_and_scales_loss():
    target = _target((0.25, 0.75))
    logits = torch.tensor([-1.0, 2.0], requires_grad=True)
    deltas = torch.ones(2, 4, requires_grad=True)
    actual = _partial_rpn_loss(_rpn(), logits, deltas, [target["boxes"]], [target])
    expected = (F.softplus(-logits) * target["confidence"]).sum() / 2
    torch.testing.assert_close(actual[0], expected)
    scaled = dict(target, confidence=target["confidence"] * 0.2)
    lower = _partial_rpn_loss(_rpn(), logits, deltas, [target["boxes"]], [scaled])
    for full, small in zip(actual, lower):
        assert full > 0
        torch.testing.assert_close(small, full * 0.2)


def test_rpn_no_positive_returns_differentiable_zero():
    target = _target()
    anchors = [torch.tensor([[30.0, 30.0, 34.0, 34.0]])]
    logits = torch.ones(1, requires_grad=True)
    deltas = torch.ones(1, 4, requires_grad=True)
    losses = _partial_rpn_loss(_rpn(), logits, deltas, anchors, [target])
    assert all(value.item() == 0 for value in losses)
    sum(losses).backward()
    assert torch.count_nonzero(logits.grad) == torch.count_nonzero(deltas.grad) == 0


def test_rpn_multi_image_flattening_and_empty_positive_image():
    targets = [_target((0.2, 0.3)), _target(), _target((0.7, 0.8))]
    far = torch.tensor([[30.0, 30.0, 34.0, 34.0]])
    anchors = [torch.cat([targets[0]["boxes"][:1], far]), far.expand(2, -1), targets[2]["boxes"]]
    logits = torch.tensor([-1.0, 3.0, 2.0, 4.0, 0.0, 1.0], requires_grad=True)
    deltas = torch.ones(6, 4, requires_grad=True)
    objectness, regression = _partial_rpn_loss(_rpn(), logits, deltas, anchors, targets)
    indices, q = torch.tensor([0, 4, 5]), torch.tensor([0.2, 0.7, 0.8])
    torch.testing.assert_close(objectness, (F.softplus(-logits[indices]) * q).sum() / 3)
    (objectness + regression).backward()
    assert torch.count_nonzero(logits.grad[1:4]) == 0
    assert torch.count_nonzero(deltas.grad[1:4]) == 0
    assert (logits.grad[indices] < 0).all()


def test_roi_samples_only_accepted_positive_proposals():
    target = _prepare_pseudo_target(_target((0.0, 0.4)), torch.zeros(3, 16, 16))
    proposals = [torch.tensor([[1.0, 1.0, 5.0, 5.0], [30.0, 30.0, 34.0, 34.0]])]
    boxes, matches, labels, regression, confidence = _select_positive_proposals(_roi(), proposals, [target])
    torch.testing.assert_close(boxes[0], target["boxes"])
    assert matches[0].tolist() == [0]
    assert labels[0].tolist() == [2]
    torch.testing.assert_close(regression[0], torch.zeros(1, 4))
    torch.testing.assert_close(confidence[0], torch.tensor([0.4]))


def test_roi_multi_image_matching_keeps_per_gt_labels_and_confidence():
    image = torch.zeros(3, 16, 16)
    targets = [_prepare_pseudo_target(_target(q), image) for q in [(0.0, 0.8), (0.3, 0.0)]]
    proposals = [_target()["boxes"], torch.tensor([[30.0, 30.0, 34.0, 34.0]])]
    boxes, matches, labels, regression, confidence = _select_positive_proposals(_roi(), proposals, targets)
    assert [len(p) for p in boxes] == [2, 1]
    assert [m.tolist() for m in matches] == [[0, 0], [0]]
    assert [label.tolist() for label in labels] == [[2, 2], [1]]
    torch.testing.assert_close(torch.cat(confidence), torch.tensor([0.8, 0.8, 0.3]))
    torch.testing.assert_close(torch.cat(regression), torch.zeros(3, 4))


def test_roi_classification_and_regression_confidence_and_no_negatives():
    # The helper also ignores nonpositive labels defensively, even though the
    # actual ROI sampler never sends them to the box head.
    logits = torch.zeros(4, 3, requires_grad=True)
    deltas = torch.ones(4, 12, requires_grad=True)
    labels = torch.tensor([1, 2, 0, -1])
    q = torch.tensor([0.25, 0.75, 1.0, 1.0])
    regression = torch.zeros(4, 4)
    losses = _positive_box_loss(logits, deltas, labels, regression, q)
    torch.testing.assert_close(losses[0], torch.log(torch.tensor(3.0)) * 0.5)
    lower = _positive_box_loss(logits, deltas, labels, regression, q * 0.2)
    for full, small in zip(losses, lower):
        torch.testing.assert_close(small, full * 0.2)
    sum(losses).backward()
    assert torch.count_nonzero(logits.grad[2:]) == 0
    assert torch.count_nonzero(deltas.grad[2:]) == 0
    assert torch.count_nonzero(deltas.grad[0, :4]) == 0  # background regressor
    assert torch.count_nonzero(deltas.grad[0, 4:8]) == 4
    assert torch.count_nonzero(deltas.grad[1, 8:12]) == 4


def test_box_loss_no_positive_is_zero():
    logits = torch.zeros(2, 3, requires_grad=True)
    deltas = torch.zeros(2, 12, requires_grad=True)
    losses = _positive_box_loss(logits, deltas, torch.tensor([0, -1]), torch.zeros(2, 4), torch.ones(2))
    sum(losses).backward()
    assert all(value.item() == 0 for value in losses)
    assert torch.count_nonzero(logits.grad) == torch.count_nonzero(deltas.grad) == 0


def test_mask_valid_zero_pixels_train_and_ignored_pixels_have_zero_gradient():
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    masks = torch.tensor([[[1.0, 0.0], [-1.0, -1.0]]])
    valid = torch.tensor([[[True, True], [False, False]]])
    loss = _weighted_mask_bce(logits, masks, torch.tensor([0.4]), valid)
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)) * 0.4)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[[-0.1, 0.1], [0.0, 0.0]]]))


def test_mask_normalizes_per_instance_by_valid_pixels_not_confidence_sum():
    logits = torch.tensor([[[0.0, 0.0]], [[2.0, 2.0]]], requires_grad=True)
    masks = torch.zeros_like(logits)
    valid = torch.tensor([[[True, False]], [[True, True]]])
    q = torch.tensor([0.2, 0.8])
    actual = _weighted_mask_bce(logits, masks, q, valid)
    expected = (0.2 * F.softplus(torch.tensor(0.0)) + 0.8 * F.softplus(torch.tensor(2.0))) / 2
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(_weighted_mask_bce(logits, masks, q * 0.25, valid), actual * 0.25)


@pytest.mark.parametrize("empty", [False, True])
def test_mask_all_ignored_or_empty_is_differentiable_zero(empty):
    count = 0 if empty else 2
    logits = torch.zeros(count, 2, 2, requires_grad=True)
    loss = _weighted_mask_bce(logits, torch.zeros_like(logits), torch.ones(count), torch.zeros_like(logits, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_projected_mask_loss_uses_matched_class_confidence_and_validity():
    target = _target((0.2, 0.6))
    target["valid_mask"] = torch.ones_like(target["masks"], dtype=torch.bool)
    target["valid_mask"][1, :, 11:] = False
    target["masks"][1, 9:11, 9:11] = 1
    proposals = [target["boxes"][[1, 0]]]
    matches = [torch.tensor([1, 0])]
    logits = torch.zeros(2, 3, 4, 4, requires_grad=True)
    loss = _partial_mask_loss(logits, proposals, [target], matches)
    projected_valid = project_masks_on_boxes(target["valid_mask"].float(), proposals[0], matches[0], 4) >= 1 - 1e-6
    assert projected_valid[0].any() and (~projected_valid[0]).any()
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)) * 0.4)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, 0]) == 0
    assert torch.count_nonzero(logits.grad[0, 1]) == 0
    assert torch.count_nonzero(logits.grad[0, 2][~projected_valid[0]]) == 0
    projected_targets = project_masks_on_boxes(target["masks"], proposals[0], matches[0], 4)
    # ROIAlign creates soft targets; a target of 0.5 correctly has zero gradient
    # at a zero logit even when the pixel is valid.
    expected_gradient = (0.5 - projected_targets[0]) * projected_valid[0] * 0.6 / projected_valid[0].sum() / 2
    torch.testing.assert_close(logits.grad[0, 2], expected_gradient)
    assert torch.count_nonzero(logits.grad[1, 1]) == 16


@pytest.mark.parametrize("explicit_validity", [False, True])
def test_q_one_mask_helper_matches_torchvision_loss_and_gradients(explicit_validity):
    target = _target()
    target["masks"][0, 1:3, 1:4] = 1
    target["masks"][1, 9:12, 9:11] = 1
    if explicit_validity:
        target["valid_mask"] = torch.ones_like(target["masks"], dtype=torch.bool)
    proposals, matches = [target["boxes"][[1, 0]]], [torch.tensor([1, 0])]
    logits = torch.linspace(-2, 2, 96).reshape(2, 3, 4, 4).requires_grad_()
    reference_logits = logits.detach().clone().requires_grad_()
    actual = _partial_mask_loss(logits, proposals, [target], matches)
    expected = maskrcnn_loss(reference_logits, proposals, [target["masks"]], [target["labels"]], matches)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_filter_and_resize_validity_without_mutating_input():
    image = torch.zeros(3, 16, 16)
    target = _target((0.0, 0.6))
    target["valid_mask"] = torch.ones_like(target["masks"], dtype=torch.bool)
    target["valid_mask"][1, :8] = False
    target["ignore_mask"] = torch.zeros(16, 16, dtype=torch.bool)
    target["ignore_mask"][:, 8:] = True
    before = {key: value.clone() for key, value in target.items() if torch.is_tensor(value)}
    prepared = _prepare_pseudo_target(target, image)
    expected = target["valid_mask"][[1]] & ~target["ignore_mask"][None]
    assert torch.equal(prepared["valid_mask"], expected)
    assert prepared["confidence"].tolist() == pytest.approx([0.6])
    transform = GeneralizedRCNNTransform(8, 8, [0.0] * 3, [1.0] * 3, size_divisible=8)
    images, transformed = _transform_pseudo(transform, [image], [prepared])
    assert images.image_sizes == [(8, 8)]
    assert transformed[0]["masks"].shape == (1, 8, 8)
    assert torch.equal(transformed[0]["valid_mask"], expected[:, ::2, ::2])
    torch.testing.assert_close(transformed[0]["boxes"], target["boxes"][[1]] / 2)
    for key, original in before.items():
        assert torch.equal(target[key], original)


@pytest.mark.parametrize("field", ["valid_mask", "ignore_mask"])
def test_shared_and_per_instance_validity_supported(field):
    target = _target()
    target[field] = torch.zeros(16, 16, dtype=torch.bool)
    shared = _prepare_pseudo_target(target, torch.zeros(3, 16, 16))
    target[field] = target[field][None].expand(2, -1, -1).clone()
    per_instance = _prepare_pseudo_target(target, torch.zeros(3, 16, 16))
    assert torch.equal(shared["valid_mask"], per_instance["valid_mask"])


def test_invalid_pixels_are_cleared_before_projection_and_default_q_is_one():
    target = _target()
    del target["confidence"]
    target["masks"] = target["masks"].float()
    target["masks"][:, :8] = -1
    target["valid_mask"] = torch.ones(16, 16, dtype=torch.bool)
    target["valid_mask"][:8] = False
    prepared = _prepare_pseudo_target(target, torch.zeros(3, 16, 16))
    assert torch.equal(prepared["confidence"], torch.ones(2))
    assert (prepared["masks"] == 0).all()


class _ForwardSpy:
    """No detector or CPU training: only verifies delegation and object identity."""
    training = True

    def __init__(self):
        self.calls = []
        self.losses = {"loss_classifier": torch.tensor(3.0)}

    def __call__(self, images, targets):
        self.calls.append((images, targets))
        return self.losses


@pytest.mark.parametrize("api", [partial_maskrcnn_loss, compute_loss_with_ignore])
def test_human_branch_delegates_exact_input_and_result(api):
    model = _ForwardSpy()
    images = [torch.zeros(3, 16, 16)]
    targets = [dict(_target((0.0, 0.0)), is_pseudo=False)]
    expected = model(images, targets)
    actual = api(model, images, targets)
    assert actual is expected
    assert model.calls[-1][0] is images
    assert model.calls[-1][1] is targets


@pytest.mark.parametrize("api", [partial_maskrcnn_loss, compute_loss_with_ignore])
@pytest.mark.parametrize("empty", [False, True])
def test_empty_or_allzero_pseudo_skips_model(api, empty):
    model = _ForwardSpy()
    target = _target((0.0, -1.0))
    if empty:
        target = {key: value[:0] if torch.is_tensor(value) else value for key, value in target.items()}
    assert api(model, [torch.zeros(3, 16, 16)], [target]) == {}
    assert not model.calls


def test_mixed_empty_pseudo_does_not_change_human_loss():
    model = _ForwardSpy()
    human = dict(_target(), is_pseudo=False)
    pseudo = _target((0.0, 0.0))
    images = [torch.zeros(3, 16, 16), torch.ones(3, 16, 16)]
    actual = compute_loss_with_ignore(model, images, [pseudo, human])
    assert actual is model.losses
    assert len(model.calls) == 1
    assert model.calls[0][0][0] is images[1]
    assert model.calls[0][1][0] is human


def test_zero_lambda_skips_nonempty_pseudo():
    model = _ForwardSpy()
    assert compute_loss_with_ignore(model, [torch.zeros(3, 16, 16)], [_target()], lambda_pseudo=0) == {}
    assert not model.calls


def test_input_validation():
    image = torch.zeros(3, 16, 16)
    with pytest.raises(ValueError, match="same length"):
        partial_maskrcnn_loss(_ForwardSpy(), [image], [])
    with pytest.raises(ValueError, match="confidence"):
        _prepare_pseudo_target(_target((float("nan"), 1.0)), image)
    with pytest.raises(ValueError, match="valid_mask"):
        _prepare_pseudo_target(dict(_target(), valid_mask=torch.ones(3, 16, 16)), image)
    bad_boxes = _target()
    bad_boxes["boxes"][0, 2] = 0
    with pytest.raises(ValueError, match="boxes"):
        _prepare_pseudo_target(bad_boxes, image)


_RUN_CUDA = os.environ.get("PHENOLEAF_RUN_CUDA_TESTS") == "1"


def _cuda_model_and_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; never fall back to CPU model training")
    from torchvision.models.detection import maskrcnn_resnet50_fpn

    model = maskrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=3,
        min_size=64, max_size=64,
        rpn_pre_nms_top_n_train=32, rpn_post_nms_top_n_train=16,
        rpn_pre_nms_top_n_test=16, rpn_post_nms_top_n_test=8,
        rpn_batch_size_per_image=16, box_batch_size_per_image=16,
    ).cuda().train()
    target = _target(size=64)
    target["boxes"] *= 4
    target["masks"][0, 4:20, 4:20] = 1
    target["masks"][1, 36:52, 36:52] = 1
    target = {key: value.cuda() if torch.is_tensor(value) else value for key, value in target.items()}
    return model, [torch.rand(3, 64, 64, device="cuda")], [target]


@pytest.mark.skipif(not _RUN_CUDA, reason="opt-in CUDA full-model equivalence test")
def test_cuda_human_matches_standard_torchvision_exactly():
    with torch.random.fork_rng(), torch.backends.cudnn.flags(deterministic=True, benchmark=False):
        model, images, targets = _cuda_model_and_inputs()
        targets[0]["is_pseudo"] = False
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        with torch.no_grad():
            expected = model(images, targets)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            actual = compute_loss_with_ignore(model, images, targets)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert torch.equal(actual[key], expected[key]), key


@pytest.mark.skipif(not _RUN_CUDA, reason="opt-in CUDA pseudo forward/backward test")
def test_cuda_pseudo_scales_all_losses_and_preserves_standard_model():
    with torch.random.fork_rng(), torch.backends.cudnn.flags(deterministic=True, benchmark=False):
        model, images, targets = _cuda_model_and_inputs()
        parameters = {name: id(parameter) for name, parameter in model.named_parameters()}
        state_keys = tuple(model.state_dict())
        original_forward = model.forward.__func__
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        with torch.no_grad():
            full = partial_maskrcnn_loss(model, images, targets)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        lower = partial_maskrcnn_loss(model, images, [dict(targets[0], confidence=targets[0]["confidence"] * 0.25)])
        assert set(lower) == {"loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"}
        for key in full:
            assert torch.isfinite(lower[key]) and full[key] > 0, key
            torch.testing.assert_close(lower[key], full[key] * 0.25, rtol=1e-5, atol=1e-7)
        sum(lower.values()).backward()
        for parameter in (
            model.backbone.body.conv1.weight, model.rpn.head.cls_logits.weight,
            model.rpn.head.bbox_pred.weight, model.roi_heads.box_predictor.cls_score.weight,
            model.roi_heads.box_predictor.bbox_pred.weight, model.roi_heads.mask_predictor.mask_fcn_logits.weight,
        ):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad) > 0
        assert parameters == {name: id(parameter) for name, parameter in model.named_parameters()}
        assert tuple(model.state_dict()) == state_keys
        assert model.forward.__func__ is original_forward
        model.eval()
        with torch.no_grad():
            predictions = model(images)
        assert {"boxes", "labels", "scores", "masks"} <= predictions[0].keys()
