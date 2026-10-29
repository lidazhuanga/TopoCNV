"""Reusable sliding-window, TTA, ensemble, and segmentation evaluation."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

ArrayLike = Union[np.ndarray, torch.Tensor]
SpatialShape = Tuple[int, int, int]
FlipAxes = Sequence[Tuple[int, ...]]


@dataclass(frozen=True)
class Window:
    """Spatial metadata for one inference window."""

    start: SpatialShape
    size: SpatialShape
    full_shape: SpatialShape


@dataclass(frozen=True)
class SegmentationMetrics:
    """Dice summary with explicit class-presence information."""

    mean_dice: float
    per_class_dice: Tuple[float, ...]
    present: Tuple[bool, ...]


Predictor = Callable[[nn.Module, torch.Tensor, Sequence[Window]], Any]


def as_probabilities(output: Any, output_is_probability: Optional[bool] = None) -> torch.Tensor:
    """Extract a prediction tensor and normalize logits to probabilities.

    Tuple/list model outputs default to treating the first element as an
    already-normalized primary prediction, matching hierarchical and binary-Y
    heads. Set ``output_is_probability`` explicitly for other model contracts.
    """
    structured = isinstance(output, (tuple, list))
    primary = output[0] if structured else output
    if not isinstance(primary, torch.Tensor):
        raise TypeError("Model output must be a tensor or a sequence beginning with a tensor")
    is_probability = structured if output_is_probability is None else output_is_probability
    if is_probability:
        return primary.float()
    if primary.shape[1] == 1:
        return torch.sigmoid(primary.float())
    return torch.softmax(primary.float(), dim=1)


def _default_predictor(model: nn.Module, patches: torch.Tensor, windows: Sequence[Window]) -> Any:
    del windows
    return model(patches)


def _positions(length: int, patch: int, overlap: float) -> List[int]:
    if length <= patch:
        return [0]
    stride = max(int(round(patch * (1.0 - overlap))), 1)
    positions = list(range(0, length - patch + 1, stride))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions


def _importance_map(patch_size: SpatialShape, mode: str) -> torch.Tensor:
    if mode == "constant":
        return torch.ones(patch_size, dtype=torch.float32)
    if mode != "gaussian":
        raise ValueError("blend_mode must be 'gaussian' or 'constant'")
    axes = []
    for size in patch_size:
        coordinate = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        sigma = max(size / 8.0, 1e-6)
        axes.append(torch.exp(-0.5 * (coordinate / sigma) ** 2))
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return (weight / weight.max()).clamp_min_(1e-7)


def _predict_with_flips(
    model: nn.Module,
    patches: torch.Tensor,
    windows: Sequence[Window],
    predictor: Predictor,
    output_is_probability: Optional[bool],
    flip_axes: FlipAxes,
) -> torch.Tensor:
    predictions: List[torch.Tensor] = []
    transforms = list(flip_axes) or [()]
    if () not in transforms:
        transforms.insert(0, ())
    for axes in transforms:
        invalid = [axis for axis in axes if axis not in (0, 1, 2)]
        if invalid:
            raise ValueError(f"TTA axes must be spatial indices 0, 1, or 2; got {invalid}")
        tensor_dims = tuple(axis + 2 for axis in axes)
        augmented = torch.flip(patches, tensor_dims) if tensor_dims else patches
        probability = as_probabilities(predictor(model, augmented, windows), output_is_probability)
        if tensor_dims:
            probability = torch.flip(probability, tensor_dims)
        predictions.append(probability)
    return torch.stack(predictions).mean(dim=0)


@torch.no_grad()
def sliding_window_inference(
    image: ArrayLike,
    model: nn.Module,
    patch_size: SpatialShape,
    num_classes: int,
    device: Union[str, torch.device],
    *,
    overlap: float = 0.5,
    batch_size: int = 1,
    blend_mode: str = "gaussian",
    predictor: Optional[Predictor] = None,
    output_is_probability: Optional[bool] = None,
    tta_flip_axes: FlipAxes = ((),),
) -> ArrayLike:
    """Predict a 3D volume using weighted overlapping windows.

    Input may be ``[D,H,W]`` or channel-first ``[C,D,H,W]``. The result is
    ``[num_classes,D,H,W]`` and follows the input container type: NumPy in,
    NumPy out; tensor in, CPU tensor out. A custom ``predictor`` adapts models
    that require coordinate metadata or additional inference arguments.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if len(patch_size) != 3 or any(size <= 0 for size in patch_size):
        raise ValueError("patch_size must contain three positive integers")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    return_numpy = isinstance(image, np.ndarray)
    image_tensor = torch.as_tensor(image, dtype=torch.float32).detach().cpu()
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.ndim != 4:
        raise ValueError("image must have shape [D,H,W] or [C,D,H,W]")

    original_shape = tuple(int(value) for value in image_tensor.shape[-3:])
    padded_shape = tuple(max(original_shape[index], patch_size[index]) for index in range(3))
    padded = torch.zeros((image_tensor.shape[0], *padded_shape), dtype=torch.float32)
    padded[:, : original_shape[0], : original_shape[1], : original_shape[2]] = image_tensor

    locations = list(
        itertools.product(
            _positions(padded_shape[0], patch_size[0], overlap),
            _positions(padded_shape[1], patch_size[1], overlap),
            _positions(padded_shape[2], patch_size[2], overlap),
        )
    )
    importance = _importance_map(patch_size, blend_mode)
    accumulator = torch.zeros((num_classes, *padded_shape), dtype=torch.float32)
    normalizer = torch.zeros(padded_shape, dtype=torch.float32)
    predict = predictor or _default_predictor
    resolved_device = torch.device(device)
    was_training = model.training
    model.eval()

    try:
        for offset in range(0, len(locations), batch_size):
            starts = locations[offset : offset + batch_size]
            windows = [Window(tuple(start), patch_size, padded_shape) for start in starts]
            patch_batch = torch.stack(
                [
                    padded[
                        :,
                        start[0] : start[0] + patch_size[0],
                        start[1] : start[1] + patch_size[1],
                        start[2] : start[2] + patch_size[2],
                    ]
                    for start in starts
                ]
            ).to(resolved_device)
            probabilities = _predict_with_flips(
                model,
                patch_batch,
                windows,
                predict,
                output_is_probability,
                tta_flip_axes,
            ).detach().float().cpu()
            if probabilities.ndim == 4:
                probabilities = probabilities.unsqueeze(1)
            if probabilities.ndim != 5:
                raise ValueError("Predictor output must have shape [B,C,D,H,W]")
            if probabilities.shape[0] != len(starts):
                raise ValueError("Predictor output batch size does not match input")
            if tuple(probabilities.shape[-3:]) != patch_size:
                raise ValueError("Predictor spatial output must match patch_size")
            if probabilities.shape[1] < num_classes:
                raise ValueError("Predictor returned fewer channels than num_classes")

            for index, start in enumerate(starts):
                region = (
                    slice(start[0], start[0] + patch_size[0]),
                    slice(start[1], start[1] + patch_size[1]),
                    slice(start[2], start[2] + patch_size[2]),
                )
                accumulator[(slice(None), *region)] += probabilities[index, :num_classes] * importance
                normalizer[region] += importance
    finally:
        model.train(was_training)

    result = accumulator / normalizer.clamp_min(1e-8).unsqueeze(0)
    result = result[:, : original_shape[0], : original_shape[1], : original_shape[2]]
    return result.numpy() if return_numpy else result


@torch.no_grad()
def tta_predict(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    flip_axes: FlipAxes = ((), (0,), (1,), (2,)),
    predict_fn: Optional[Callable[[nn.Module, torch.Tensor], Any]] = None,
    output_is_probability: Optional[bool] = None,
) -> torch.Tensor:
    """Average identity and flip test-time augmentations for a tensor batch."""
    if inputs.ndim < 5:
        raise ValueError("inputs must have shape [B,C,D,H,W]")
    fn = predict_fn or (lambda current_model, value: current_model(value))
    predictions: List[torch.Tensor] = []
    transforms = list(flip_axes) or [()]
    if () not in transforms:
        transforms.insert(0, ())
    was_training = model.training
    model.eval()
    try:
        for axes in transforms:
            invalid = [axis for axis in axes if axis not in (0, 1, 2)]
            if invalid:
                raise ValueError(f"TTA axes must be spatial indices 0, 1, or 2; got {invalid}")
            dims = tuple(axis + 2 for axis in axes)
            augmented = torch.flip(inputs, dims) if dims else inputs
            probability = as_probabilities(fn(model, augmented), output_is_probability)
            predictions.append(torch.flip(probability, dims) if dims else probability)
    finally:
        model.train(was_training)
    return torch.stack(predictions).mean(dim=0)


@torch.no_grad()
def ensemble_predict(
    models: Sequence[nn.Module],
    inputs: torch.Tensor,
    *,
    predict_fn: Optional[Callable[[nn.Module, torch.Tensor], Any]] = None,
    output_is_probability: Optional[bool] = None,
    tta_flip_axes: Optional[FlipAxes] = None,
) -> torch.Tensor:
    """Average probability predictions from one or more models."""
    if not models:
        raise ValueError("models must not be empty")
    fn = predict_fn or (lambda current_model, value: current_model(value))
    predictions = []
    for model in models:
        if tta_flip_axes is not None:
            probability = tta_predict(
                model,
                inputs,
                flip_axes=tta_flip_axes,
                predict_fn=fn,
                output_is_probability=output_is_probability,
            )
        else:
            was_training = model.training
            model.eval()
            try:
                probability = as_probabilities(fn(model, inputs), output_is_probability)
            finally:
                model.train(was_training)
        predictions.append(probability)
    return torch.stack(predictions).mean(dim=0)


def evaluate_segmentation(
    prediction: ArrayLike,
    target: ArrayLike,
    num_classes: int,
    *,
    include_background: bool = False,
    ignore_empty: bool = True,
) -> SegmentationMetrics:
    """Compute hard-label per-class and mean Dice scores."""
    predicted = torch.as_tensor(prediction)
    truth = torch.as_tensor(target)
    if predicted.ndim == truth.ndim + 1:
        predicted = predicted.argmax(dim=0)
    if predicted.shape != truth.shape:
        raise ValueError("prediction and target spatial shapes must match")
    first_class = 0 if include_background else 1
    scores: List[float] = []
    present: List[bool] = []
    for class_index in range(first_class, num_classes):
        predicted_mask = predicted == class_index
        truth_mask = truth == class_index
        denominator = int(predicted_mask.sum() + truth_mask.sum())
        is_present = denominator > 0
        present.append(is_present)
        scores.append(
            float(2.0 * (predicted_mask & truth_mask).sum().item() / denominator)
            if is_present
            else float("nan")
        )
    valid = [score for score, exists in zip(scores, present) if exists or not ignore_empty]
    if not ignore_empty:
        valid = [0.0 if np.isnan(score) else score for score in valid]
    mean_dice = float(np.mean(valid)) if valid else 0.0
    return SegmentationMetrics(mean_dice, tuple(scores), tuple(present))


@torch.no_grad()
def legacy_sliding_window_inference(image, model, patch_size, num_classes, device,
                                    step_size=0.5, coord_aware=False, *,
                                    gaussian_filter_fn=None):
    """历史 cascade 滑窗；NumPy float32 融合，int 步长，不执行 TTA。

    兼容层显式传入 SciPy gaussian_filter；缺失时使用全 1 权重。
    保留 impulse Gaussian 中心、坐标元数据及 structured output 概率语义。
    """
    model.eval()
    orig_shape = image.shape[-3:]
    d, h, w = orig_shape if image.ndim == 3 else image.shape[1:]
    pd, ph, pw = patch_size
    image_work = image[None] if image.ndim == 3 else image
    C_in = image_work.shape[0]
    nd, nh, nw = max(d, pd), max(h, ph), max(w, pw)
    if nd != d or nh != h or nw != w:
        padded = np.zeros((C_in, nd, nh, nw), dtype=np.float32)
        padded[:, :d, :h, :w] = image_work
        image_work = padded
        d, h, w = nd, nh, nw
    agg_results = np.zeros((num_classes, d, h, w), dtype=np.float32)
    agg_weights = np.zeros((d, h, w), dtype=np.float32)
    if gaussian_filter_fn is not None:
        sigmas = [s / 8.0 for s in patch_size]
        tmp = np.zeros(patch_size, dtype=np.float32)
        center = tuple(s // 2 for s in patch_size)
        tmp[center] = 1.0
        gw = gaussian_filter_fn(tmp, sigma=sigmas, mode='constant', cval=0.0)
        gw = (gw / gw.max()).astype(np.float32)
    else:
        gw = np.ones(patch_size, dtype=np.float32)
    import itertools
    steps = []
    for dim_size, p_size in zip([d, h, w], [pd, ph, pw]):
        if dim_size <= p_size:
            steps.append([0])
        else:
            step = max(int(p_size * step_size), 1)
            pos = list(range(0, dim_size - p_size + 1, step))
            if pos[-1] + p_size < dim_size:
                pos.append(dim_size - p_size)
            steps.append(pos)
    positions = list(itertools.product(*steps))
    for z1, y1, x1 in positions:
        patch = image_work[:, z1:z1+pd, y1:y1+ph, x1:x1+pw]
        patch_t = torch.from_numpy(patch[None]).float().to(device)
        if coord_aware:
            meta = torch.tensor([[z1, y1, x1, d, h, w, 0., 0., 0.]],
                                dtype=torch.float32, device=device)
            out = model(patch_t, None, (meta[:, 0:3], meta[:, 3:6], meta[:, 6:9]))
        else:
            out = model(patch_t)
        if isinstance(out, (list, tuple)):
            prob = out[0].detach().float().cpu().numpy()[0]
        else:
            prob = torch.softmax(out, dim=1).detach().float().cpu().numpy()[0]
        if prob.ndim == 3:
            prob = prob[None]
        n_out = min(num_classes, prob.shape[0])
        for c in range(n_out):
            agg_results[c, z1:z1+pd, y1:y1+ph, x1:x1+pw] += prob[c] * gw
        agg_weights[z1:z1+pd, y1:y1+ph, x1:x1+pw] += gw
    agg_weights = np.maximum(agg_weights, 1e-8)
    for c in range(num_classes):
        agg_results[c] /= agg_weights
    return agg_results[:, :orig_shape[0], :orig_shape[1], :orig_shape[2]]


def legacy_fold_mean(probabilities):
    """按历史 fold 顺序执行 np.mean；不增添 TTA 或不同精度的融合。"""
    return np.mean(probabilities, axis=0)


__all__ = [
    "legacy_sliding_window_inference",
    "legacy_fold_mean",
    "Predictor",
    "SegmentationMetrics",
    "Window",
    "as_probabilities",
    "ensemble_predict",
    "evaluate_segmentation",
    "sliding_window_inference",
    "tta_predict",
]
