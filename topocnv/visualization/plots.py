"""通用 CT/MR 分割叠加与训练曲线工具，不包含论文布局或实验路径。

NumPy 图像工具不依赖 matplotlib；绘图时才加载该可选依赖。
新建图形使用独立的 Agg canvas，不修改 pyplot 后端或全局 rcParams。
所有输入须已配准到同一网格；不猜测医学图像方向、不自动转置或重采样。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

Color = Union[str, Sequence[float]]
LabelColors = Mapping[int, Color]
# 延用现有绘图工具的区分类别配色，但不绑定任何论文类别名称。
DEFAULT_LABEL_COLORS: Dict[int, Tuple[float, float, float]] = {
    1: (58 / 255, 111 / 255, 196 / 255),
    2: (51 / 255, 187 / 255, 238 / 255),
    3: (34 / 255, 136 / 255, 51 / 255),
    4: (238 / 255, 102 / 255, 119 / 255),
    5: (170 / 255, 51 / 255, 119 / 255),
}
CT_WINDOW: Tuple[float, float] = (-160.0, 240.0)


def _image_array(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 0 or x.size == 0:
        raise ValueError("图像必须是具有非零尺寸的数组")
    return x


def _label_array(labels: np.ndarray, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    lab = np.asarray(labels)
    if lab.ndim == 0 or lab.size == 0:
        raise ValueError("标签必须是具有非零尺寸的数组")
    if shape is not None and lab.shape != shape:
        raise ValueError("图像与标签的形状必须相同，不允许隐式广播")
    if not np.all(np.isfinite(lab)) or np.any(lab < 0) or np.any(lab != np.floor(lab)):
        raise ValueError("标签必须是有限、非负整数；概率预测请先阈值化或 argmax")
    return lab


def normalize_image(
    image: np.ndarray,
    *,
    window: Optional[Tuple[float, float]] = None,
    percentiles: Tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """归一化灰度图为 float32 [0,1]，不修改输入。

    window=(下限,上限) 用于原始 CT HU（可用 CT_WINDOW）或缓存 (0,1)。
    window=None 时按有限值百分位自动归一化，适用于 MR。
    常量/全非有限图像返回零；NaN/Inf 显示为黑色。
    """
    x = _image_array(image)
    finite = np.isfinite(x)
    if window is None:
        low, high = percentiles
        if not 0 <= low < high <= 100:
            raise ValueError("percentiles 必须满足 0 <= 下限 < 上限 <= 100")
        if not finite.any():
            return np.zeros_like(x)
        lo, hi = np.percentile(x[finite], percentiles)
    else:
        lo, hi = (float(v) for v in window)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError("window 必须是两个有限且递增的强度值")
    if hi <= lo:
        return np.zeros_like(x)
    safe = np.where(finite, x, lo)
    return np.clip((safe - lo) / (hi - lo), 0, 1).astype(np.float32)


def _rgb(color: Color) -> np.ndarray:
    if isinstance(color, str):
        if not color.startswith("#") or len(color) != 7:
            raise ValueError("颜色字符串必须为 #RRGGBB 格式")
        try:
            color = tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
        except ValueError as exc:
            raise ValueError("颜色必须是有效的 #RRGGBB 字符串") from exc
    rgb = np.asarray(color, dtype=np.float32)
    if rgb.shape != (3,) or not np.all(np.isfinite(rgb)) or np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("RGB 颜色必须包含三个 [0,1] 范围内的有限值")
    return rgb


def _blend(gray: np.ndarray, labels: np.ndarray, alpha: float, colors: Optional[LabelColors]) -> np.ndarray:
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha 必须位于 [0,1]")
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    palette = tuple(DEFAULT_LABEL_COLORS.values())
    for label in np.unique(labels):
        if label == 0:
            continue
        index = int(label)
        fallback = DEFAULT_LABEL_COLORS.get(index, palette[(index - 1) % len(palette)])
        color = fallback if colors is None else colors.get(index, fallback)
        mask = labels == label
        rgb[mask] = rgb[mask] * (1 - alpha) + _rgb(color) * alpha
    return rgb


def make_overlay(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    *,
    alpha: float = 0.45,
    colors: Optional[LabelColors] = None,
    window: Optional[Tuple[float, float]] = None,
    percentiles: Tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """生成二维灰度切片与二值/多分类标签的 RGB 叠加图，仅需 NumPy。

    输出为 [H,W,3] float32；背景标签 0 保持原灰度。
    可用 colors={类别: RGB三元组或 '#RRGGBB'} 覆盖颜色；未指定类别采用默认色。
    输入图像和 mask 必须同形；归一化参数与 normalize_image 相同。
    """
    image = _image_array(image_slice)
    if image.ndim != 2:
        raise ValueError("叠加图需要二维灰度切片 [H,W]")
    labels = _label_array(mask_slice, image.shape)
    return _blend(normalize_image(image, window=window, percentiles=percentiles), labels, alpha, colors)


def select_slice(mask: np.ndarray, *, axis: int = 0) -> int:
    """选择三维标签沿指定轴前景最多的切片，平局取最接近前景质心者。

    mask 可为二值或多分类标签。全背景时返回中心切片，不产生 NaN。
    """
    labels = _label_array(mask)
    if labels.ndim != 3 or axis not in (0, 1, 2):
        raise ValueError("需要三维标签和 axis=0、1 或 2")
    foreground = np.moveaxis(labels > 0, axis, 0)
    counts = foreground.reshape(foreground.shape[0], -1).sum(axis=1)
    if not counts.any():
        return foreground.shape[0] // 2
    center = float(np.dot(np.arange(counts.size), counts) / counts.sum())
    candidates = np.flatnonzero(counts == counts.max())
    return int(candidates[np.argmin(np.abs(candidates - center))])


def _new_figure(figsize: Tuple[float, float]) -> Figure:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ImportError("绘图需要可选依赖 matplotlib；NumPy make_overlay 不需要该依赖") from exc
    fig = Figure(figsize=figsize, constrained_layout=True)
    FigureCanvasAgg(fig)
    return fig


def _aspect(spacing: Optional[Sequence[float]], ndim: int) -> float:
    if spacing is None:
        return 1.0
    sp = tuple(float(s) for s in spacing)
    if len(sp) != ndim or any(not np.isfinite(s) or s <= 0 for s in sp):
        raise ValueError("spacing 必须按数组轴顺序提供有限正数间距")
    return sp[0] / sp[1]


def plot_overlay(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    *,
    ax: Optional[Axes] = None,
    alpha: float = 0.45,
    colors: Optional[LabelColors] = None,
    window: Optional[Tuple[float, float]] = None,
    spacing: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
) -> Axes:
    """在指定轴（或新建独立图形）绘制分割叠加，返回可继续定制的 Axes。

    spacing=(行间距,列间距) 控制物理长宽比，不改变输入方向。
    新图形不注册到 pyplot，无需 pyplot.close；可使用 save_figure 保存。
    """
    overlay = make_overlay(image_slice, mask_slice, alpha=alpha, colors=colors, window=window)
    aspect = _aspect(spacing, 2)
    if ax is None:
        ax = _new_figure((5.0, 5.0)).subplots()
    ax.imshow(overlay, interpolation="nearest", origin="upper", aspect=aspect)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    return ax


def plot_segmentation(
    image: np.ndarray,
    prediction: np.ndarray,
    target: Optional[np.ndarray] = None,
    *,
    axis: int = 0,
    slice_index: Optional[int] = None,
    spacing: Optional[Sequence[float]] = None,
    window: Optional[Tuple[float, float]] = None,
    alpha: float = 0.45,
    colors: Optional[LabelColors] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, np.ndarray]:
    """绘制原图、可选参考及预测对比，返回 (Figure, 一维 Axes 数组)。

    支持 [H,W] 切片或 [D,H,W] 体积；三维输入沿 axis 取同一 slice_index。
    未指定切片时按参考（缺省为预测）的前景面积选择；使用同一强度窗。
    spacing 按完整输入轴顺序提供；不假定轴对应某个解剖方向。
    """
    x = _image_array(image)
    if x.ndim not in (2, 3):
        raise ValueError("图像必须是二维切片或三维体积")
    p = _label_array(prediction, x.shape)
    g = None if target is None else _label_array(target, x.shape)
    if spacing is not None:
        sp = tuple(float(s) for s in spacing)
        if len(sp) != x.ndim or any(not np.isfinite(s) or s <= 0 for s in sp):
            raise ValueError("spacing 必须为每个输入轴提供有限正数间距")
    else:
        sp = None
    if x.ndim == 3:
        if axis not in (0, 1, 2):
            raise ValueError("axis 必须为 0、1 或 2")
        index = select_slice(p if g is None else g, axis=axis) if slice_index is None else slice_index
        if not isinstance(index, (int, np.integer)) or not 0 <= index < x.shape[axis]:
            raise ValueError("slice_index 必须是有效的非负切片索引")
        x, p = np.take(x, index, axis=axis), np.take(p, index, axis=axis)
        g = None if g is None else np.take(g, index, axis=axis)
        sp = None if sp is None else tuple(s for i, s in enumerate(sp) if i != axis)
    elif slice_index is not None:
        raise ValueError("二维输入不接受 slice_index")
    gray = normalize_image(x, window=window)
    panels = [("Image", np.repeat(gray[..., None], 3, axis=-1))]
    if g is not None:
        panels.append(("Reference", _blend(gray, g, alpha, colors)))
    panels.append(("Prediction", _blend(gray, p, alpha, colors)))
    fig = _new_figure((4.0 * len(panels), 4.0))
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    for ax, (label, data) in zip(axes, panels):
        ax.imshow(data, interpolation="nearest", origin="upper", aspect=_aspect(sp, 2))
        ax.set_title(label)
        ax.set_axis_off()
    if title is not None:
        fig.suptitle(title)
    return fig, axes


def _metric_name(name: str) -> str:
    for prefix in ("train_", "val_", "validation_", "test_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    for suffix in ("_train", "_val", "_validation", "_test"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def plot_training_curves(
    history: Mapping[str, Sequence[float]],
    *,
    epochs: Optional[Sequence[float]] = None,
    groups: Optional[Mapping[str, Sequence[str]]] = None,
    smoothing: int = 1,
    title: Optional[str] = None,
) -> Tuple[Figure, np.ndarray]:
    """绘制任意训练指标，返回 (Figure, 一维 Axes 数组)。

    history 如 train_loss/val_loss/val_dice 到数值序列的映射；默认按去除
    train/val 等前后缀的指标名分面，也可用 groups={面板名: [序列名]} 分组。
    不传 epochs 时各曲线独立从 1 开始，允许验证曲线短于训练曲线；显式
    epochs 必须与每条所绘序列长度一致。smoothing 为因果滑动均值窗口，
    NaN/Inf 作为缺失值保留断点，不修改原始历史数据。
    """
    if not history:
        raise ValueError("history 不能为空")
    if not isinstance(smoothing, (int, np.integer)) or smoothing < 1:
        raise ValueError("smoothing 必须是正整数")
    if groups is None:
        automatic: Dict[str, list] = {}
        for name in history:
            automatic.setdefault(_metric_name(name), []).append(name)
        groups = automatic
    if not groups or any(not names or isinstance(names, str) for names in groups.values()):
        raise ValueError("groups 必须包含非空序列名称列表")
    xvalues = None if epochs is None else np.asarray(epochs, dtype=float)
    if xvalues is not None and (xvalues.ndim != 1 or not np.all(np.isfinite(xvalues))):
        raise ValueError("epochs 必须是一维有限数值序列")
    prepared = {}
    for names in groups.values():
        for name in names:
            if name not in history:
                raise ValueError(f"history 缺少序列：{name}")
            values = np.asarray(history[name], dtype=float)
            if values.ndim != 1 or values.size == 0:
                raise ValueError(f"曲线必须是非空的一维数值序列：{name}")
            if xvalues is not None and xvalues.size != values.size:
                raise ValueError(f"epochs 与曲线长度不一致：{name}")
            valid = np.isfinite(values)
            if smoothing > 1:
                kernel = np.ones(min(smoothing, values.size))
                sums = np.convolve(np.where(valid, values, 0), kernel, mode="full")[:values.size]
                counts = np.convolve(valid.astype(float), kernel, mode="full")[:values.size]
                values = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
            values = np.where(valid, values, np.nan)
            prepared[name] = (np.arange(1, values.size + 1) if xvalues is None else xvalues, values)
    fig = _new_figure((5.0 * len(groups), 3.8))
    axes = fig.subplots(1, len(groups), squeeze=False)[0]
    for ax, (metric, names) in zip(axes, groups.items()):
        for name in names:
            x, y = prepared[name]
            ax.plot(x, y, label=name, linewidth=1.6)
        ax.set(title=metric, xlabel="Epoch", ylabel=metric)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
    if title is not None:
        fig.suptitle(title)
    return fig, axes


def save_figure(figure: Figure, path: Union[str, Path], *, dpi: int = 150) -> Path:
    """保存图形并按需创建父目录；格式由后缀决定，返回输出路径。

    不关闭或清空图形，调用者可继续定制或保存其他格式。
    """
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi 必须为正数")
    output = Path(path)
    if not output.suffix:
        raise ValueError("输出路径必须包含图像格式后缀")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    return output


__all__ = [
    "CT_WINDOW", "DEFAULT_LABEL_COLORS", "make_overlay", "normalize_image",
    "plot_overlay", "plot_segmentation", "plot_training_curves", "save_figure", "select_slice",
]
