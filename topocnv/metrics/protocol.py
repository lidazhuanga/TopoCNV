"""可复用的 NumPy 分割指标及 CNV v0.3/v0.4 数组层评估协议。

源自 MIA/scripts/metrics_protocol.py；不依赖实验目录、模型或文件加载。
主要终点始终是 Y（标签 1..4，排除 class5）的 Dice 和 HD95。
保留原协议的边界到实心掩膜距离、L1 v2、空预测惩罚及可见性评估口径。
"""

from __future__ import annotations

import os
from numbers import Real
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = np.ndarray
MetricRow = Dict[str, Union[float, str]]
Y_LABELS: Tuple[int, ...] = (1, 2, 3, 4)
BONY_ROI_LABEL: int = 4
BYPASS_ROI_LABEL: int = 5
L1_DILATE_R_MM: float = 3.0
BONE_HU_LOW: float = 200.0
BONE_HU_HIGH: float = 3000.0
SURF_TOL_MM: float = 2.0
VIS_LABELS: Tuple[int, ...] = (2, 3)
VIS_DILATE_R_MM: float = 3.0
TIER_LABELS: Tuple[int, ...] = (1, 2, 3, 4)
V04_KEYS: Tuple[str, ...] = ("SurfDice_Y", "Recall_vis", "Dice_vis", "Lvis_over_Y") + tuple(
    f"Dice_tier{k}" for k in TIER_LABELS
)
DEFAULT_SPACING_ZYX: Tuple[float, float, float] = (2.5, 1.1328120231628418, 1.1328120231628418)
DEFAULT_DATA_ROOT = Path(os.environ.get(
    "CNV_DATA_ROOT", str(Path(__file__).resolve().parents[2] / "npc_data" / "train")))


def _same_shape(pred: ArrayLike, gt: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    p, g = np.asarray(pred), np.asarray(gt)
    if p.shape != g.shape:
        raise ValueError("预测和参考掩膜的形状必须相同，不允许隐式广播")
    if p.ndim == 0 or p.size == 0:
        raise ValueError("输入必须是具有非零空间尺寸的数组")
    return p, g


def _masks(pred: ArrayLike, gt: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    p, g = _same_shape(pred, gt)
    return p.astype(bool, copy=False), g.astype(bool, copy=False)


def _spacing(spacing: Optional[Sequence[float]], ndim: int) -> Tuple[float, ...]:
    sp = (1.0,) * ndim if spacing is None else tuple(float(s) for s in spacing)
    if len(sp) != ndim or any(not np.isfinite(s) or s <= 0 for s in sp):
        raise ValueError("spacing 必须为每个数组轴提供一个有限、正数间距")
    return sp


def mask_Y(labels: ArrayLike) -> np.ndarray:
    """合并标签 1..4 为主要评估区域 Y；class5 不属于 Y。"""
    lab = np.asarray(labels)
    return (lab >= 1) & (lab <= 4)


def mask_123(labels: ArrayLike) -> np.ndarray:
    """返回标签 1、2、3 的联合掩膜。"""
    lab = np.asarray(labels)
    return (lab >= 1) & (lab <= 3)


def mask_c4(labels: ArrayLike) -> np.ndarray:
    """返回骨性区域（仅 class4）。"""
    return np.asarray(labels) == BONY_ROI_LABEL


def mask_bypass(labels: ArrayLike) -> np.ndarray:
    """返回绕行区域 class5；该区域不参与 Y/L1 的参考掩膜。"""
    return np.asarray(labels) == BYPASS_ROI_LABEL


def pred_to_Y(pred: ArrayLike, *, binary: bool = False) -> np.ndarray:
    """转换预测为 Y；多分类输入合并 1..4。

    binary=True 时，布尔值和最大值不超过 1 的输入按 >0.5 二值化，
    其他正整数掩膜按 >0 二值化，与已有协议一致。logits 请先转概率。
    """
    p = np.asarray(pred)
    if not binary:
        return mask_Y(p)
    if p.size == 0 or p.dtype == np.bool_ or p.max() <= 1.0:
        return p > 0.5
    return p > 0


def dice_score(pred: ArrayLike, gt: ArrayLike, *, empty_value: float = float("nan")) -> float:
    """计算硬二值 Dice（非零值视为前景）；双方为空时默认 NaN。

    概率输入须先阈值化。empty_value 可显式设为 1，但协议使用默认值。
    多分类硬标签评估复用 metrics.evaluate_segmentation。
    """
    p, g = _masks(pred, gt)
    denominator = int(p.sum()) + int(g.sum())
    return float(2.0 * (p & g).sum() / denominator) if denominator else float(empty_value)


def iou_score(pred: ArrayLike, gt: ArrayLike, *, empty_value: float = float("nan")) -> float:
    """计算二值交并比（Jaccard）；双方为空时默认 NaN。"""
    p, g = _masks(pred, gt)
    union = int((p | g).sum())
    return float((p & g).sum() / union) if union else float(empty_value)


def recall_score(pred: ArrayLike, gt: ArrayLike) -> float:
    """计算前景召回率；参考为空时为 NaN。"""
    p, g = _masks(pred, gt)
    count = int(g.sum())
    return float((p & g).sum() / count) if count else float("nan")


def precision_score(pred: ArrayLike, gt: ArrayLike) -> float:
    """计算前景精确率；预测为空时为 NaN。"""
    p, g = _masks(pred, gt)
    count = int(p.sum())
    return float((p & g).sum() / count) if count else float("nan")


def dilate_mm(mask: ArrayLike, spacing_zyx: Sequence[float], r_mm: float = L1_DILATE_R_MM) -> np.ndarray:
    """按已有协议的 ceil(r/spacing) 椭球足迹膨胀，不改动冻结的 L1 定义。

    间距顺序必须与数组轴一致；该体素椭球是物理球的近似。
    """
    from scipy.ndimage import binary_dilation

    m = np.asarray(mask, dtype=bool)
    sp = _spacing(spacing_zyx, m.ndim)
    if m.ndim != 3:
        raise ValueError("协议膨胀要求三维数组 [D,H,W]")
    if not np.isfinite(r_mm) or r_mm < 0:
        raise ValueError("r_mm 必须是有限非负数")
    if not m.any() or r_mm == 0:
        return m.copy()
    radii = tuple(max(1, int(np.ceil(r_mm / s))) for s in sp)
    z, y, x = np.ogrid[tuple(slice(-r, r + 1) for r in radii)]
    footprint = (z / radii[0]) ** 2 + (y / radii[1]) ** 2 + (x / radii[2]) ** 2 <= 1.0
    return binary_dilation(m, structure=footprint)


def bone_window_high_contrast(
    hu: ArrayLike,
    hu_low: float = BONE_HU_LOW,
    hu_high: float = BONE_HU_HIGH,
    *,
    use_grad: bool = False,
    grad_percentile: float = 90.0,
) -> np.ndarray:
    """根据原始 CT HU 提取骨窗区域；禁止用 [0,1] 归一化缓存替代 HU。

    可选梯度交集保留原实现的中心差分及骨窗内百分位阈值。
    """
    if not np.isfinite(hu_low) or not np.isfinite(hu_high) or hu_low > hu_high:
        raise ValueError("HU 阈值必须有限且 hu_low 不大于 hu_high")
    if not 0 <= grad_percentile <= 100:
        raise ValueError("grad_percentile 必须位于 [0,100]")
    x = np.asarray(hu, dtype=np.float32)
    bone = (x >= hu_low) & (x <= hu_high)
    if not use_grad or not bone.any():
        return bone
    if x.ndim != 3:
        raise ValueError("梯度骨窗要求三维 HU 数组")
    gz, gy, gx = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    gz[1:-1] = x[2:] - x[:-2]
    gy[:, 1:-1] = x[:, 2:] - x[:, :-2]
    gx[:, :, 1:-1] = x[:, :, 2:] - x[:, :, :-2]
    gradient = np.sqrt(gz * gz + gy * gy + gx * gx)
    return bone & (gradient >= np.percentile(gradient[bone], grad_percentile))


def build_H_bone(
    hu: ArrayLike,
    labels: ArrayLike,
    spacing_zyx: Sequence[float],
    r_mm: float = L1_DILATE_R_MM,
    *,
    hu_low: float = BONE_HU_LOW,
    hu_high: float = BONE_HU_HIGH,
) -> np.ndarray:
    """返回 H = 骨窗 HU 区域 ∪ 膨胀的 class4，不使用 class5 作为膨胀源。"""
    hu, labels = _same_shape(hu, labels)
    return bone_window_high_contrast(hu, hu_low, hu_high) | dilate_mm(mask_c4(labels), spacing_zyx, r_mm)


def build_L1(
    labels: ArrayLike,
    hu: ArrayLike,
    spacing_zyx: Sequence[float],
    r_mm: float = L1_DILATE_R_MM,
    **bone_kw,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (Y, H, L)，其中低证据区域 L = Y 去除 H。"""
    y = mask_Y(labels)
    h = build_H_bone(hu, labels, spacing_zyx, r_mm, **bone_kw)
    return y, h, y & ~h


def _directed_distances(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Sequence[float],
    *,
    boundary_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import binary_erosion, distance_transform_edt

    # 联合包围盒加一层背景；保持原图像边缘语义并降低距离变换内存开销。
    coordinates = np.nonzero(pred | gt)
    box = tuple(slice(max(int(c.min()) - 1, 0), min(int(c.max()) + 2, n))
                for c, n in zip(coordinates, pred.shape))
    p, g = pred[box], gt[box]
    bp = p & ~binary_erosion(p, border_value=0)
    bg = g & ~binary_erosion(g, border_value=0)
    destination_g, destination_p = (bg, bp) if boundary_only else (g, p)
    to_g = distance_transform_edt(~destination_g, sampling=spacing)[bp]
    to_p = distance_transform_edt(~destination_p, sampling=spacing)[bg]
    return to_g, to_p


def hd95(pred: ArrayLike, gt: ArrayLike, spacing_zyx: Optional[Sequence[float]] = None) -> float:
    """协议 HD95：两个方向边界到实心掩膜距离的第 95 百分位之最大值。

    间距按数组轴顺序提供，省略时为单位体素。参考为空为 NaN；
    预测为空而参考非空时，使用原图像的物理对角线作为失败惩罚。
    此口径刻意与现有协议一致，不使用合并方向后的百分位。
    """
    p, g = _masks(pred, gt)
    sp = _spacing(spacing_zyx, p.ndim)
    if not g.any():
        return float("nan")
    if not p.any():
        return float(np.sqrt(sum((n * s) ** 2 for n, s in zip(g.shape, sp))))
    d1, d2 = _directed_distances(p, g, sp)
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


def assd(pred: ArrayLike, gt: ArrayLike, spacing_zyx: Optional[Sequence[float]] = None) -> float:
    """协议 ASSD：两个方向边界到实心掩膜距离均值的平均；任一为空为 NaN。"""
    p, g = _masks(pred, gt)
    sp = _spacing(spacing_zyx, p.ndim)
    if not p.any() or not g.any():
        return float("nan")
    d1, d2 = _directed_distances(p, g, sp)
    return float(0.5 * (d1.mean() + d2.mean()))


def hausdorff_distance(
    pred: ArrayLike, gt: ArrayLike, spacing_zyx: Optional[Sequence[float]] = None
) -> float:
    """计算体素边界之间的最大 Hausdorff 距离；空掩膜语义与 hd95 相同。"""
    p, g = _masks(pred, gt)
    sp = _spacing(spacing_zyx, p.ndim)
    if not g.any():
        return float("nan")
    if not p.any():
        return float(np.sqrt(sum((n * s) ** 2 for n, s in zip(g.shape, sp))))
    d1, d2 = _directed_distances(p, g, sp, boundary_only=True)
    return float(max(d1.max(), d2.max()))


def surface_dice(
    pred: ArrayLike,
    gt: ArrayLike,
    spacing_zyx: Optional[Sequence[float]] = None,
    tol_mm: float = SURF_TOL_MM,
) -> float:
    """计算容差内的体素边界 Dice（非面积加权）；双方空为 NaN，单方空为 0。

    边界由一次轴连接腐蚀得到；距离目标是另一掩膜的边界，而非实心区域。
    """
    p, g = _masks(pred, gt)
    sp = _spacing(spacing_zyx, p.ndim)
    if not np.isfinite(tol_mm) or tol_mm < 0:
        raise ValueError("tol_mm 必须是有限非负数")
    if not p.any() and not g.any():
        return float("nan")
    if not p.any() or not g.any():
        return 0.0
    d1, d2 = _directed_distances(p, g, sp, boundary_only=True)
    return float(((d1 <= tol_mm).sum() + (d2 <= tol_mm).sum()) / (d1.size + d2.size))


def _c3w_counts(pY: ArrayLike, gt_labels: ArrayLike) -> Dict[str, int]:
    p, labels = _same_shape(pY, gt_labels)
    p, g3, g = p > 0, labels == 3, mask_Y(labels)
    return {"i3": int(np.count_nonzero(p & g3)), "io": int(np.count_nonzero(p & g & ~g3)),
            "p3": int(np.count_nonzero(p & g3)), "po": int(np.count_nonzero(p & ~g3)),
            "G3": int(np.count_nonzero(g3)), "Go": int(np.count_nonzero(g & ~g3))}


def _wdice_from_counts(counts: Mapping[str, int], lam: float) -> float:
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam 必须是有限非负权重")
    numerator = 2.0 * (counts["io"] + lam * counts["i3"])
    denominator = counts["po"] + lam * counts["p3"] + counts["Go"] + lam * counts["G3"]
    return float(numerator / denominator) if denominator else float("nan")


def weighted_dice_y(pred: ArrayLike, gt_labels: ArrayLike, lam: float = 0.5) -> float:
    """计算参考证据加权的二值 Y Dice；仅参考 class3 的权重为 lam。

    pred 必须已经是二值 Y。背景及 class5 上的假阳性始终保持权重 1。
    lam=1 退化为标准 Dice；双方为空为 NaN。此指标不替代主要终点。
    """
    return _wdice_from_counts(_c3w_counts(pred, gt_labels), lam)


def recall_gt3(pred: ArrayLike, gt_labels: ArrayLike) -> float:
    """计算二值预测 Y 对参考 class3 的覆盖率；无 class3 时为 NaN。"""
    counts = _c3w_counts(pred, gt_labels)
    return float(counts["i3"] / counts["G3"]) if counts["G3"] else float("nan")


def partition_Y_by_posterior(pred_Y: ArrayLike, s1_softmax: ArrayLike) -> np.ndarray:
    """用 Stage-1 的 1..4 类后验划分 Stage-2 Y；输出前景联合严格等于 Y。

    s1_softmax 为 [C>=5,D,H,W]（支持 mmap）；Y 外输出 0。
    后验必须来自模型，不能用参考标签替代真实预测。
    """
    p = np.asarray(pred_Y, dtype=bool)
    probabilities = np.asarray(s1_softmax)
    if p.ndim != 3 or probabilities.ndim != 4 or probabilities.shape[0] < 5:
        raise ValueError("需要三维 Y 和 [C>=5,D,H,W] 后验")
    if probabilities.shape[1:] != p.shape:
        raise ValueError("Stage-1 后验和 Y 的空间形状必须相同")
    out = np.zeros(p.shape, dtype=np.int16)
    if p.any():
        coordinates = np.nonzero(p)
        box = tuple(slice(int(c.min()), int(c.max()) + 1) for c in coordinates)
        sub = probabilities[(slice(1, 5),) + box]
        out[box] = np.where(p[box], np.argmax(sub, axis=0) + 1, 0)
    return out


def score_v04(
    pY: ArrayLike,
    gt: ArrayLike,
    spacing_zyx: Sequence[float],
    vis_pred: Optional[ArrayLike] = None,
) -> Dict[str, float]:
    """计算表面 Dice、不可见区域召回/局部 Dice 以及可选可见性层级 Dice。"""
    p, gt = _same_shape(pY, gt)
    p = p.astype(bool, copy=False)
    g = mask_Y(gt)
    out = {"SurfDice_Y": surface_dice(p, g, spacing_zyx)}
    invisible = np.isin(gt, VIS_LABELS)
    if invisible.any():
        region = dilate_mm(invisible, spacing_zyx, VIS_DILATE_R_MM)
        out.update(Recall_vis=recall_score(p, invisible), Dice_vis=dice_score(p & region, g & region),
                   Lvis_over_Y=float(invisible.sum() / g.sum()))
    else:
        out.update({key: float("nan") for key in ("Recall_vis", "Dice_vis", "Lvis_over_Y")})
    v = None if vis_pred is None else _same_shape(vis_pred, gt)[0]
    for k in TIER_LABELS:
        out[f"Dice_tier{k}"] = float("nan") if v is None else dice_score(v == k, gt == k)
    return out


def score_protocol(
    pred: ArrayLike,
    gt_labels: ArrayLike,
    hu: Optional[ArrayLike] = None,
    spacing_zyx: Sequence[float] = DEFAULT_SPACING_ZYX,
    *,
    pred_binary_Y: bool = False,
    secondary_pred: Optional[ArrayLike] = None,
    secondary_source: str = "s1",
    vis_pred: Optional[ArrayLike] = None,
    r_mm: float = L1_DILATE_R_MM,
    hu_low: float = BONE_HU_LOW,
    hu_high: float = BONE_HU_HIGH,
) -> MetricRow:
    """计算 CNV 协议指标；输入均须在同一 [D,H,W] 网格，HU 必须是原始 HU。

    输出主要 Dice_Y/HD95_Y、ASSD_Y、加权 Dice、class3 覆盖、次要分类、
    v0.4 可见性指标，以及 L1 v2：Dice(P 去除 H,L)，假阳性不被裁掉。
    二值预测无 secondary_pred 时，次要分类为 NaN、来源为 N/A。
    缺少 HU 时 L1 字段为 NaN；secondary_pred 和 vis_pred 只能来自模型。
    """
    pred, gt = _same_shape(pred, gt_labels)
    if gt.ndim != 3:
        raise ValueError("CNV 协议要求三维数组 [D,H,W]")
    sp = _spacing(spacing_zyx, gt.ndim)
    p, g = pred_to_Y(pred, binary=pred_binary_Y), mask_Y(gt)
    out: MetricRow = {"Dice_Y": dice_score(p, g), "HD95_Y": hd95(p, g, sp),
                      "Empty_pred_Y": float(not p.any() and g.any()), "ASSD_Y": assd(p, g, sp)}
    counts = _c3w_counts(p, gt)
    for lam in (0.25, 0.5, 0.75):
        out[f"wDSC_Y_{lam:.2f}"] = _wdice_from_counts(counts, lam)
    out["Recall_GT3"] = float(counts["i3"] / counts["G3"]) if counts["G3"] else float("nan")
    if secondary_pred is not None:
        secondary = _same_shape(secondary_pred, gt)[0]
        out["Secondary_source"] = secondary_source
    elif pred_binary_Y:
        secondary = None
        out["Secondary_source"] = "N/A"
    else:
        secondary = pred
        out["Secondary_source"] = "pred"
    out["Dice_123"] = float("nan") if secondary is None else dice_score(mask_123(secondary), mask_123(gt))
    out["Dice_c4"] = float("nan") if secondary is None else dice_score(mask_c4(secondary), mask_c4(gt))
    out.update(score_v04(p, gt, sp, vis_pred))
    if hu is None:
        out.update({key: float("nan") for key in
                    ("Dice_L", "Precision_L", "Recall_L", "L_over_Y", "Dice_L_legacy", "Dice_H")})
    else:
        y, h, low = build_L1(gt, hu, sp, r_mm, hu_low=hu_low, hu_high=hu_high)
        pred_low = p & ~h
        out.update(Dice_L=dice_score(pred_low, low), Precision_L=precision_score(pred_low, low),
                   Recall_L=recall_score(p, low), Dice_L_legacy=dice_score(p & low, low),
                   Dice_H=dice_score(p & h, y & h),
                   L_over_Y=float(low.sum() / y.sum()) if y.any() else float("nan"))
    return out


def load_raw_hu_dhw(
    case_id: str,
    data_root: Union[str, Path] = DEFAULT_DATA_ROOT,
    target_spacing_zyx: Optional[Sequence[float]] = None,
    reference_shape_dhw: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """按缓存轴顺序读取原始 CT HU；可重采样并对齐参考数组形状。"""
    import nibabel as nib

    data_root = Path(data_root)
    ct_path = data_root / "image" / f"{case_id}.nii.gz"
    if not ct_path.exists():
        ct_path = data_root / "image" / f"{case_id}.nii"
    if not ct_path.exists():
        raise FileNotFoundError(f"Raw CT not found for {case_id} under {data_root}/image/")
    nii = nib.load(str(ct_path))
    hu = np.transpose(nii.get_fdata().astype(np.float32), (2, 0, 1))
    zooms = nii.header.get_zooms()[:3]
    orig_sp = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    if target_spacing_zyx is not None:
        try:
            from ..data import preprocess as pp
            hu = pp.resample_image_to_spacing(
                hu, orig_sp, tuple(target_spacing_zyx), order=1
            ).astype(np.float32)
        except Exception:
            from scipy.ndimage import zoom
            factors = [orig_sp[i] / float(target_spacing_zyx[i]) for i in range(3)]
            hu = zoom(hu, factors, order=1).astype(np.float32)
        spacing = tuple(float(s) for s in target_spacing_zyx)
    else:
        spacing = orig_sp
    if reference_shape_dhw is not None and hu.shape != tuple(reference_shape_dhw):
        out = np.zeros(reference_shape_dhw, dtype=np.float32)
        slices = tuple(slice(0, min(hu.shape[i], reference_shape_dhw[i])) for i in range(3))
        out[slices] = hu[slices]
        hu = out
    return hu, spacing


def score_case_from_paths(
    case_id: str,
    pred: ArrayLike,
    gt_labels: ArrayLike,
    data_root: Union[str, Path] = DEFAULT_DATA_ROOT,
    spacing_zyx: Sequence[float] = DEFAULT_SPACING_ZYX,
    *,
    pred_binary_Y: bool = False,
    secondary_pred: Optional[ArrayLike] = None,
    secondary_source: str = "s1",
    vis_pred: Optional[ArrayLike] = None,
    load_hu: bool = True,
) -> MetricRow:
    """评估单病例，可加载原始 HU 计算 L1；保留旧版评估脚本调用接口。"""
    hu = None
    sp = tuple(spacing_zyx)
    if load_hu:
        hu, sp = load_raw_hu_dhw(
            case_id, data_root=data_root, target_spacing_zyx=spacing_zyx,
            reference_shape_dhw=tuple(np.asarray(gt_labels).shape),
        )
    row = score_protocol(
        pred, gt_labels, hu=hu, spacing_zyx=sp,
        pred_binary_Y=pred_binary_Y, secondary_pred=secondary_pred,
        secondary_source=secondary_source, vis_pred=vis_pred,
    )
    row["case_id"] = case_id
    return row


def _self_test() -> None:
    """运行旧版协议的 CPU 自检，无需数据文件或 GPU。"""
    rng = np.random.RandomState(0)
    gt = np.zeros((20, 32, 32), dtype=np.int16)
    gt[5:8, 10:14, 10:14] = 1
    gt[8:10, 12:16, 12:16] = 2
    gt[10:12, 14:18, 14:18] = 3
    gt[12:15, 8:20, 8:20] = 4
    gt[16:18, 20:24, 20:24] = 5
    assert mask_Y(gt).sum() == (gt >= 1).sum() - mask_bypass(gt).sum()
    assert not (mask_Y(gt) & mask_bypass(gt)).any()
    hu = rng.randn(*gt.shape).astype(np.float32) * 50
    hu[mask_c4(gt)] = 800
    Y, H, L = build_L1(gt, hu, DEFAULT_SPACING_ZYX, r_mm=3.0)
    assert (L & mask_bypass(gt)).sum() == 0
    assert (Y & mask_bypass(gt)).sum() == 0
    pred = gt.copy()
    pred[mask_bypass(gt)] = 0
    m = score_protocol(pred, gt, hu=hu, spacing_zyx=DEFAULT_SPACING_ZYX)
    assert m["Dice_Y"] > 0.99 and m["Secondary_source"] == "pred"
    assert all(k in m for k in ("Dice_L", "Recall_L", "L_over_Y", "HD95_Y", "ASSD_Y"))
    m_bin = score_protocol(mask_Y(pred), gt, hu=hu, pred_binary_Y=True)
    assert m_bin["Secondary_source"] == "N/A" and np.isnan(m_bin["Dice_123"])
    m_cas = score_protocol(mask_Y(pred), gt, hu=hu, pred_binary_Y=True, secondary_pred=pred)
    assert m_cas["Secondary_source"] == "s1"
    assert m_cas["Dice_123"] > 0.99 and m_cas["Dice_c4"] > 0.99
    _self_test_L1(gt, hu)
    _self_test_partition(gt)
    _self_test_v04(gt)
    print("[metrics_protocol] self-test OK", {k: (round(v, 4) if isinstance(v, float) else v)
                                              for k, v in m.items()})


def _self_test_L1(gt: np.ndarray, hu: np.ndarray) -> None:
    """验证 L1 v2 的假阳性惩罚、漏检、空预测与缺失 HU 语义。"""
    sp = DEFAULT_SPACING_ZYX
    Y, H, L = build_L1(gt, hu, sp, r_mm=3.0)
    assert L.any() and (Y & H).any()

    def run(p):
        return score_protocol(p.astype(np.uint8), gt, hu=hu, spacing_zyx=sp, pred_binary_Y=True)

    def close(a, b):
        return abs(a - b) < 1e-9

    m = run(Y)
    for k in ("Dice_L", "Precision_L", "Recall_L", "Dice_L_legacy", "Dice_H"):
        assert close(m[k], 1.0), (k, m[k])
    hu2 = hu.copy()
    blob = np.zeros_like(Y)
    blob[1:4, 24:30, 1:7] = True
    hu2[blob] = 40.0
    assert not (blob & Y).any()
    Y2, H2, L2 = build_L1(gt, hu2, sp, r_mm=3.0)
    assert not (blob & H2).any()
    m = score_protocol((Y | blob).astype(np.uint8), gt, hu=hu2, spacing_zyx=sp, pred_binary_Y=True)
    n_l, n_fp = int(L2.sum()), int(blob.sum())
    assert close(m["Recall_L"], 1.0) and close(m["Dice_L_legacy"], 1.0)
    assert close(m["Precision_L"], n_l / (n_l + n_fp))
    assert close(m["Dice_L"], 2 * n_l / (2 * n_l + n_fp))
    assert close(m["Dice_H"], 1.0)
    rim = H & ~Y
    assert rim.any()
    m = run(Y | rim)
    assert close(m["Dice_L"], 1.0) and close(m["Precision_L"], 1.0)
    assert m["Dice_H"] < 1.0 and m["Dice_Y"] < 1.0
    miss = L.copy()
    idx = np.flatnonzero(miss)
    miss.flat[idx[: len(idx) // 2]] = False
    m = run((Y & ~L) | miss)
    r = miss.sum() / L.sum()
    assert close(m["Recall_L"], r)
    assert close(m["Dice_L_legacy"], 2 * r / (1 + r))
    assert close(m["Dice_L"], m["Dice_L_legacy"]) and close(m["Precision_L"], 1.0)
    m = run(Y | blob)
    assert close(m["Dice_L_legacy"], 2 * m["Recall_L"] / (1 + m["Recall_L"]))
    m = run(np.zeros_like(Y))
    assert m["Dice_L"] == 0.0 and m["Recall_L"] == 0.0 and np.isnan(m["Precision_L"])
    m = score_protocol(Y.astype(np.uint8), gt, hu=None, pred_binary_Y=True)
    assert all(np.isnan(m[k]) for k in ("Dice_L", "Precision_L", "Recall_L", "Dice_L_legacy", "Dice_H"))
    print("[metrics_protocol] L1 v2 self-test OK")


def _self_test_partition(gt: np.ndarray) -> None:
    """验证后验划分类别，前景联合始终等于 Stage-2 Y。"""
    onehot = np.stack([(gt == c) for c in range(6)]).astype(np.float32)
    Y = mask_Y(gt)
    lab = partition_Y_by_posterior(Y, onehot)
    assert np.array_equal(lab > 0, Y) and np.array_equal(lab[Y], gt[Y])
    m = score_protocol(Y.astype(np.uint8), gt, pred_binary_Y=True, secondary_pred=lab,
                       secondary_source="s1_inY")
    assert m["Secondary_source"] == "s1_inY" and m["Dice_123"] == 1.0 and m["Dice_c4"] == 1.0
    bg = np.zeros_like(onehot)
    bg[0], bg[5] = 0.9, 0.1
    bg[2, Y] = 0.05
    lab = partition_Y_by_posterior(Y, bg)
    assert np.array_equal(lab > 0, Y) and set(np.unique(lab[Y])) == {2}
    p = Y.copy()
    p[0, 0, 0] = True
    assert np.array_equal(partition_Y_by_posterior(p, onehot) > 0, p)
    assert not partition_Y_by_posterior(np.zeros_like(Y), onehot).any()
    print("[metrics_protocol] A_inY partition self-test OK")


def _self_test_v04(gt: np.ndarray) -> None:
    """验证表面 Dice、不可见片段指标及分层可见性 Dice。"""
    sp = DEFAULT_SPACING_ZYX
    Y = mask_Y(gt)
    L_vis = np.isin(gt, VIS_LABELS)
    R_vis = dilate_mm(L_vis, sp, r_mm=VIS_DILATE_R_MM)

    def close(a, b):
        return abs(a - b) < 1e-9

    def run(p, vis=None):
        return score_protocol(p.astype(np.uint8), gt, spacing_zyx=sp, pred_binary_Y=True, vis_pred=vis)

    m = run(Y)
    for k in ("SurfDice_Y", "Recall_vis", "Dice_vis"):
        assert close(m[k], 1.0), (k, m[k])
    assert close(m["Lvis_over_Y"], L_vis.sum() / Y.sum())
    assert all(np.isnan(m[f"Dice_tier{k}"]) for k in TIER_LABELS)
    assert close(run(np.roll(Y, 1, axis=2))["SurfDice_Y"], 1.0)
    sd3 = run(np.roll(Y, 3, axis=2))["SurfDice_Y"]
    assert 0.0 < sd3 < 1.0
    assert run(np.zeros_like(Y))["SurfDice_Y"] == 0.0
    assert np.isnan(surface_dice(np.zeros_like(Y), np.zeros_like(Y), sp))
    assert run(np.roll(Y, 1, axis=0))["SurfDice_Y"] < 1.0
    fp = np.zeros_like(Y)
    fp[9, 11, 13] = True
    assert R_vis[9, 11, 13] and not Y[9, 11, 13]
    m = run(Y | fp)
    n = int((Y & R_vis).sum())
    assert close(m["Recall_vis"], 1.0) and close(m["Dice_vis"], 2 * n / (2 * n + 1))
    far = np.zeros_like(Y)
    far[1:4, 24:30, 1:7] = True
    assert not (far & R_vis).any()
    m = run(Y | far)
    assert close(m["Dice_vis"], 1.0) and m["Dice_Y"] < 1.0
    m = run(Y & ~(gt == 3))
    assert close(m["Recall_vis"], (gt == 2).sum() / L_vis.sum()) and m["Dice_vis"] < 1.0
    vis = np.where(gt == 5, 0, gt)
    m = run(Y, vis)
    assert all(close(m[f"Dice_tier{k}"], 1.0) for k in TIER_LABELS)
    sw = vis.copy()
    sw[vis == 2], sw[vis == 3] = 3, 2
    m = run(Y, sw)
    assert m["Dice_tier2"] == 0.0 and m["Dice_tier3"] == 0.0 and close(m["Dice_tier1"], 1.0)
    g2 = np.where(np.isin(gt, VIS_LABELS), 1, gt)
    m = score_protocol(mask_Y(g2).astype(np.uint8), g2, spacing_zyx=sp, pred_binary_Y=True)
    assert np.isnan(m["Recall_vis"]) and np.isnan(m["Dice_vis"])
    print("[metrics_protocol] v0.4 self-test OK")


def aggregate_metrics(
    rows: Sequence[Mapping[str, Union[float, str]]], keys: Optional[Sequence[str]] = None
) -> Dict[str, float]:
    """逐病例等权汇总数值指标，忽略缺失/NaN，不丢弃无穷值或空预测惩罚。

    自动选择数值字段，排除 case_id/Secondary_source 等文本元数据。
    全部缺失的指定字段返回 NaN；不要把多分类均值当作主要 Dice_Y。
    """
    if keys is None:
        keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, Real)})
    result = {}
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), Real)]
        valid = [value for value in values if not np.isnan(value)]
        result[key] = float(np.mean(valid)) if valid else float("nan")
    return result


# 旧评估脚本迁移时可直接改模块路径，私有别名不列入公共导出。
_dice = dice_score
_recall = recall_score
_precision = precision_score
_hd95 = hd95
_assd = assd
_surface_dice = surface_dice
_wdice = weighted_dice_y
_recall_gt3 = recall_gt3

__all__ = [
    "BONE_HU_HIGH", "BONE_HU_LOW", "BONY_ROI_LABEL", "BYPASS_ROI_LABEL",
    "DEFAULT_DATA_ROOT", "DEFAULT_SPACING_ZYX", "L1_DILATE_R_MM", "SURF_TOL_MM", "TIER_LABELS", "V04_KEYS",
    "VIS_DILATE_R_MM", "VIS_LABELS", "Y_LABELS", "aggregate_metrics", "assd",
    "bone_window_high_contrast", "build_H_bone", "build_L1", "dice_score", "dilate_mm",
    "hausdorff_distance", "hd95", "iou_score", "load_raw_hu_dhw", "mask_123", "mask_Y", "mask_bypass", "mask_c4",
    "partition_Y_by_posterior", "precision_score", "pred_to_Y", "recall_gt3", "recall_score",
    "score_case_from_paths", "score_protocol", "score_v04", "surface_dice", "weighted_dice_y",
]
