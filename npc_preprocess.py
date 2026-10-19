import json
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import numpy as np
import torch

try:
    import scipy.ndimage as ndi
    from scipy.ndimage import zoom
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:  # pragma: no cover - 如果没有 scipy 就不做仿射
    ndi = None
    zoom = None
    gaussian_filter = None
    HAS_SCIPY = False


def normalize_ct(image: np.ndarray, clip_lower: float = -200.0, clip_upper: float = 300.0) -> np.ndarray:
    """CT 归一化：窗宽裁剪 + z-score 标准化。
    
    Args:
        image: 输入CT图像
        clip_lower: HU值下界，默认-200（保留空气腔、脂肪等组织）
        clip_upper: HU值上界，默认300（保留骨骼边界，排除金属伪影）
        
    Returns:
        归一化后的图像
        
    Note:
        默认范围 [-200, 300] HU 专门针对鼻咽癌放疗勾画任务优化：
        - 保留：空气腔、脂肪、软组织、肿瘤、肌肉、淋巴结、脑组织、骨骼边界
        - 排除：金属伪影（>300 HU）和极端低值（<-200 HU）
    """
    image = np.clip(image, clip_lower, clip_upper)
    mean = image.mean()
    std = image.std()
    if std < 1e-6:
        std = 1.0
    image = (image - mean) / std
    return image.astype(np.float32)


def normalize_percentile(image: np.ndarray, lower: float = 1.0, upper: float = 99.0,
                          b_min: float = 0.0, b_max: float = 1.0, clip: bool = True) -> np.ndarray:
    """Percentile-based intensity normalization to [b_min, b_max].

    Used by CachedSOTADataset — must match the normalization applied during
    cache building so that validation-time preprocessing is identical to training.
    """
    a_min = np.percentile(image, lower)
    a_max = np.percentile(image, upper)
    if a_max - a_min < 1e-8:
        return np.full_like(image, (b_min + b_max) / 2, dtype=np.float32)
    image = (image - a_min) / (a_max - a_min) * (b_max - b_min) + b_min
    if clip:
        image = np.clip(image, b_min, b_max)
    return image.astype(np.float32)


def compute_median_spacing(spacing_list: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """计算训练集的中位数 spacing（参考 nnUNet）。
    
    Args:
        spacing_list: 所有训练样本的 spacing 列表，格式为 [(z, y, x), ...]
        
    Returns:
        中位数 spacing (z, y, x)
    """
    if not spacing_list:
        raise ValueError("spacing_list 不能为空")
    
    spacing_array = np.array(spacing_list)  # shape: (N, 3)
    median_spacing = np.median(spacing_array, axis=0)
    return tuple(median_spacing.tolist())


def resample_image_to_spacing(
    image: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    order: int = 3,
    cval: float = 0.0,
) -> np.ndarray:
    """将图像重采样到目标 spacing（参考 nnUNet）。
    
    Args:
        image: 输入图像，shape (D, H, W)
        original_spacing: 原始 spacing (z, y, x)
        target_spacing: 目标 spacing (z, y, x)
        order: 插值阶数，图像用3（三次），标签用0（最近邻）
        cval: 边界外的填充值
        
    Returns:
        重采样后的图像
    """
    if not HAS_SCIPY:
        raise ImportError("需要安装 scipy 才能进行 spacing 重采样")
    
    # 计算缩放因子 zoom_factors = original_spacing / target_spacing
    zoom_factors = [
        original_spacing[i] / target_spacing[i]
        for i in range(3)
    ]
    
    # 处理多通道图像 (shape: D,H,W,C)
    if image.ndim == 4:
        # 对每个通道分别进行重采样
        channels = []
        for c in range(image.shape[3]):
            ch_resampled = zoom(image[..., c], zoom_factors, order=order, mode='constant', cval=cval)
            channels.append(ch_resampled)
        resampled = np.stack(channels, axis=-1)
    elif image.ndim == 3:
        # 使用 scipy.ndimage.zoom 进行重采样
        resampled = zoom(image, zoom_factors, order=order, mode='constant', cval=cval)
    else:
        raise ValueError(f"Expected image with 3 or 4 dimensions, got {image.ndim}D")
    
    return resampled.astype(image.dtype)


def save_normalization_params(params: dict, save_path: str) -> None:
    """保存归一化参数到 JSON 文件。
    
    Args:
        params: 归一化参数字典，包含 'target_spacing' 等
        save_path: 保存路径
    """
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    print(f"归一化参数已保存到: {save_path}")


def load_normalization_params(load_path: str) -> dict:
    """从 JSON 文件加载归一化参数。
    
    Args:
        load_path: 参数文件路径
        
    Returns:
        归一化参数字典
    """
    if not Path(load_path).exists():
        raise FileNotFoundError(f"归一化参数文件不存在: {load_path}")
    
    with open(load_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    return params


def _pad_to_shape_3d(arr: np.ndarray, target_shape) -> np.ndarray:
    assert arr.ndim == 3
    pads = []
    for curr, target in zip(arr.shape, target_shape):
        total = max(target - curr, 0)
        before = total // 2
        after = total - before
        pads.append((before, after))
    return np.pad(arr, pads, mode="constant", constant_values=0)


def _pad_or_crop_to_shape_3d(arr: np.ndarray, target_shape) -> np.ndarray:
    """
    将3D数组裁剪或填充到目标形状。
    用于处理spacing重采样后的舍入误差。
    
    Args:
        arr: 输入数组 (D, H, W)
        target_shape: 目标形状 (D, H, W)
        
    Returns:
        调整后的数组
    """
    assert arr.ndim == 3
    result = arr
    
    for axis in range(3):
        curr_size = result.shape[axis]
        target_size = target_shape[axis]
        
        if curr_size > target_size:
            # 需要裁剪
            diff = curr_size - target_size
            start = diff // 2
            slices = [slice(None)] * 3
            slices[axis] = slice(start, start + target_size)
            result = result[tuple(slices)]
        elif curr_size < target_size:
            # 需要填充
            diff = target_size - curr_size
            before = diff // 2
            after = diff - before
            pads = [(0, 0)] * 3
            pads[axis] = (before, after)
            result = np.pad(result, pads, mode="constant", constant_values=0)
    
    return result


def random_crop_3d(
    image: np.ndarray,
    label: np.ndarray,
    patch_size,
    ensure_foreground: bool = True,
    foreground_ratio: float = 0.001,  # 再次提高,确保patch中有足够前景
    max_attempts: int = 200,  # 大幅增加尝试次数
) -> Tuple[np.ndarray, np.ndarray]:
    """3D 随机裁剪，确保前景存在。
    
    针对极度稀疏目标(前景<0.03%),必须强制采样到前景区域
    
    Args:
        image: 输入图像
        label: 输入标签
        patch_size: 裁剪块大小
        ensure_foreground: 是否确保裁剪的patch中包含前景
        foreground_ratio: 前景像素占比的最小阈值
        max_attempts: 最大尝试次数
    """
    assert image.shape == label.shape
    d, h, w = image.shape
    pd, ph, pw = patch_size

    if d < pd or h < ph or w < pw:
        image = _pad_to_shape_3d(image, (max(d, pd), max(h, ph), max(w, pw)))
        label = _pad_to_shape_3d(label, (max(d, pd), max(h, ph), max(w, pw)))
        d, h, w = image.shape

    if not ensure_foreground:
        # 原始随机裁剪
        z1 = np.random.randint(0, d - pd + 1)
        y1 = np.random.randint(0, h - ph + 1)
        x1 = np.random.randint(0, w - pw + 1)
        z2, y2, x2 = z1 + pd, y1 + ph, x1 + pw
        return image[z1:z2, y1:y2, x1:x2], label[z1:z2, y1:y2, x1:x2]
    
    # 确保前景的裁剪策略
    # 先找到所有前景像素的坐标
    foreground_coords = np.argwhere(label > 0)
    
    if len(foreground_coords) == 0:
        # 如果没有前景，回退到随机裁剪
        z1 = np.random.randint(0, d - pd + 1)
        y1 = np.random.randint(0, h - ph + 1)
        x1 = np.random.randint(0, w - pw + 1)
        z2, y2, x2 = z1 + pd, y1 + ph, x1 + pw
        return image[z1:z2, y1:y2, x1:x2], label[z1:z2, y1:y2, x1:x2]
    
    # 尝试多次裁剪，确保包含足够的前景
    patch_volume = pd * ph * pw
    min_foreground_voxels = max(1, int(patch_volume * foreground_ratio))
    
    for attempt in range(max_attempts):
        # 随机选择一个前景点作为中心附近的裁剪
        fg_idx = np.random.randint(0, len(foreground_coords))
        fg_z, fg_y, fg_x = foreground_coords[fg_idx]
        
        # 在前景点附近随机偏移，确保前景点在patch内
        # 计算有效的裁剪范围
        z1_min = max(0, fg_z - pd + 1)
        z1_max = min(d - pd, fg_z)
        y1_min = max(0, fg_y - ph + 1)
        y1_max = min(h - ph, fg_y)
        x1_min = max(0, fg_x - pw + 1)
        x1_max = min(w - pw, fg_x)
        
        if z1_max < z1_min:
            z1_min = z1_max = max(0, min(d - pd, fg_z))
        if y1_max < y1_min:
            y1_min = y1_max = max(0, min(h - ph, fg_y))
        if x1_max < x1_min:
            x1_min = x1_max = max(0, min(w - pw, fg_x))
        
        z1 = np.random.randint(z1_min, z1_max + 1) if z1_max > z1_min else z1_min
        y1 = np.random.randint(y1_min, y1_max + 1) if y1_max > y1_min else y1_min
        x1 = np.random.randint(x1_min, x1_max + 1) if x1_max > x1_min else x1_min
        
        z2, y2, x2 = z1 + pd, y1 + ph, x1 + pw
        
        # 检查裁剪区域的前景像素数
        label_patch = label[z1:z2, y1:y2, x1:x2]
        num_foreground = np.sum(label_patch > 0)
        
        if num_foreground >= min_foreground_voxels:
            return image[z1:z2, y1:y2, x1:x2], label_patch
    
    # 如果尝试多次仍未找到合适的patch，返回最后一次尝试的结果
    return image[z1:z2, y1:y2, x1:x2], label[z1:z2, y1:y2, x1:x2]


def center_crop_3d(image: np.ndarray, label: np.ndarray, patch_size) -> Tuple[np.ndarray, np.ndarray]:
    assert image.shape == label.shape
    d, h, w = image.shape
    pd, ph, pw = patch_size

    if d < pd or h < ph or w < pw:
        image = _pad_to_shape_3d(image, (max(d, pd), max(h, ph), max(w, pw)))
        label = _pad_to_shape_3d(label, (max(d, pd), max(h, ph), max(w, pw)))
        d, h, w = image.shape

    z1 = (d - pd) // 2
    y1 = (h - ph) // 2
    x1 = (w - pw) // 2
    z2, y2, x2 = z1 + pd, y1 + ph, x1 + pw

    return image[z1:z2, y1:y2, x1:x2], label[z1:z2, y1:y2, x1:x2]


def random_flip_3d(image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """在三个轴上独立 0.5 概率翻转。"""
    assert image.shape == label.shape
    for axis in range(3):
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)
    return image, label


def random_rotate90_3d(image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """在 (H, W) 平面做 0/90/180/270 度旋转。"""
    assert image.shape == label.shape
    k = np.random.randint(0, 4)
    if k == 0:
        return image, label
    image = np.rot90(image, k=k, axes=(1, 2))
    label = np.rot90(label, k=k, axes=(1, 2))
    return image, label


def _rotation_matrix_3d(angles_rad: Tuple[float, float, float]) -> np.ndarray:
    ax, ay, az = angles_rad
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(ax), -np.sin(ax)],
            [0, np.sin(ax), np.cos(ax)],
        ],
        dtype=np.float32,
    )
    Ry = np.array(
        [
            [np.cos(ay), 0, np.sin(ay)],
            [0, 1, 0],
            [-np.sin(ay), 0, np.cos(ay)],
        ],
        dtype=np.float32,
    )
    Rz = np.array(
        [
            [np.cos(az), -np.sin(az), 0],
            [np.sin(az), np.cos(az), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return Rz @ Ry @ Rx


def random_affine_3d(
    image: np.ndarray,
    label: np.ndarray,
    p: float = 0.3,
    rotation_deg: float = 15.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
) -> Tuple[np.ndarray, np.ndarray]:
    """近似 nnUNet 的 3D 仿射增强（需要 scipy）。"""
    if not HAS_SCIPY or np.random.rand() > p:
        return image, label

    assert image.shape == label.shape
    d, h, w = image.shape

    # 随机旋转（每个轴）和各向同性缩放
    angles = np.deg2rad(
        np.random.uniform(-rotation_deg, rotation_deg, size=3).astype(np.float32)
    )
    R = _rotation_matrix_3d(tuple(angles))
    scale = np.random.uniform(scale_range[0], scale_range[1])
    M = R * scale

    center = np.array([d / 2.0, h / 2.0, w / 2.0], dtype=np.float32)
    offset = center - M @ center

    image_affine = ndi.affine_transform(
        image,
        M,
        offset=offset,
        order=1,
        mode="constant",
        cval=0.0,
    )
    label_affine = ndi.affine_transform(
        label,
        M,
        offset=offset,
        order=0,
        mode="constant",
        cval=0.0,
    )
    return image_affine, label_affine


def random_gamma(
    image: np.ndarray,
    p: float = 0.3,
    gamma_range: Tuple[float, float] = (0.7, 1.5),
) -> np.ndarray:
    """随机 gamma 变换（在 [min, max] 归一化后再做）。"""
    if np.random.rand() > p:
        return image

    img_min = float(image.min())
    img_max = float(image.max())
    if img_max - img_min < 1e-6:
        return image

    img_norm = (image - img_min) / (img_max - img_min)
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    img_norm = np.power(img_norm, gamma)
    image_aug = img_norm * (img_max - img_min) + img_min
    return image_aug.astype(np.float32)


def random_gaussian_noise(
    image: np.ndarray,
    p: float = 0.3,
    sigma_range: Tuple[float, float] = (0.0, 0.1),
) -> np.ndarray:
    """随机高斯噪声。"""
    if np.random.rand() > p:
        return image

    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    if sigma <= 0.0:
        return image
    noise = np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return (image + noise).astype(np.float32)


def preprocess_training(
    image: np.ndarray,
    label: np.ndarray,
    patch_size,
    original_spacing: Optional[Tuple[float, float, float]] = None,
    target_spacing: Optional[Tuple[float, float, float]] = None,
    ensure_foreground: bool = True,
    clip_lower: float = -200.0,
    clip_upper: float = 300.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """训练阶段预处理 + 数据增强。
    
    Args:
        image: 输入图像 (D, H, W)
        label: 输入标签 (D, H, W)
        patch_size: 裁剪块大小
        original_spacing: 原始 spacing (z, y, x)，如果提供则进行 spacing 归一化
        target_spacing: 目标 spacing (z, y, x)，如果提供则进行 spacing 归一化
        ensure_foreground: 是否确保裁剪的patch中包含前景
        clip_lower: HU值下界，用于CT强度归一化
        clip_upper: HU值上界，用于CT强度归一化
    """
    # 1. Spacing 归一化（如果提供了参数）
    if original_spacing is not None and target_spacing is not None:
        image = resample_image_to_spacing(image, original_spacing, target_spacing, order=3)
        label = resample_image_to_spacing(label, original_spacing, target_spacing, order=0)
    
    # 2. CT 强度归一化
    image = normalize_ct(image, clip_lower=clip_lower, clip_upper=clip_upper)
    
    # 3. 数据增强
    image, label = random_crop_3d(image, label, patch_size, ensure_foreground=ensure_foreground)
    image, label = random_flip_3d(image, label)
    image, label = random_rotate90_3d(image, label)
    image, label = random_affine_3d(image, label)
    image = random_gamma(image)
    image = random_gaussian_noise(image)

    image = np.ascontiguousarray(image)
    label = np.ascontiguousarray(label)

    image_t = torch.from_numpy(image[None, ...]).float()
    label_t = torch.from_numpy(label.astype(np.int64))
    return image_t, label_t


def preprocess_validation(
    image: np.ndarray,
    label: np.ndarray,
    patch_size,
    original_spacing: Optional[Tuple[float, float, float]] = None,
    target_spacing: Optional[Tuple[float, float, float]] = None,
    clip_lower: float = -200.0,
    clip_upper: float = 300.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """验证阶段：只做确定性裁剪和归一化。
    
    Args:
        image: 输入图像 (D, H, W)
        label: 输入标签 (D, H, W)
        patch_size: 裁剪块大小
        original_spacing: 原始 spacing (z, y, x)，如果提供则进行 spacing 归一化
        target_spacing: 目标 spacing (z, y, x)，如果提供则进行 spacing 归一化
        clip_lower: HU值下界，用于CT强度归一化
        clip_upper: HU值上界，用于CT强度归一化
    """
    # 1. Spacing 归一化（使用训练时保存的目标 spacing）
    if original_spacing is not None and target_spacing is not None:
        image = resample_image_to_spacing(image, original_spacing, target_spacing, order=3)
        label = resample_image_to_spacing(label, original_spacing, target_spacing, order=0)
    
    # 2. CT 强度归一化
    image = normalize_ct(image, clip_lower=clip_lower, clip_upper=clip_upper)
    
    # 3. 中心裁剪
    image, label = center_crop_3d(image, label, patch_size)

    image = np.ascontiguousarray(image)
    label = np.ascontiguousarray(label)

    image_t = torch.from_numpy(image[None, ...]).float()
    label_t = torch.from_numpy(label.astype(np.int64))
    return image_t, label_t


# ========== 滑窗推理相关函数 ==========

def compute_gaussian_weights(patch_size: Tuple[int, int, int], sigma_scale: float = 1.0 / 8) -> np.ndarray:
    """
    计算3D高斯权重图，用于滑窗推理时的加权平均（参考 nnUNet）。
    
    Args:
        patch_size: patch大小 (D, H, W)
        sigma_scale: sigma缩放因子，默认为patch_size的1/8
        
    Returns:
        高斯权重图，shape为patch_size，中心权重最大，边缘权重最小
    """
    if not HAS_SCIPY:
        # 如果没有scipy，返回全1权重（等权重平均）
        return np.ones(patch_size, dtype=np.float32)
    
    # 计算每个维度的sigma
    sigmas = [s * sigma_scale for s in patch_size]
    
    # 创建中心为1的图像
    tmp = np.zeros(patch_size, dtype=np.float32)
    center = tuple([s // 2 for s in patch_size])
    tmp[center] = 1.0
    
    # 应用高斯滤波，生成高斯权重图
    gaussian_weight = gaussian_filter(tmp, sigma=sigmas, mode='constant', cval=0.0)
    
    # 归一化到[0, 1]，中心为1
    gaussian_weight = gaussian_weight / gaussian_weight.max()
    
    return gaussian_weight.astype(np.float32)


def compute_steps_for_sliding_window(
    image_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    step_size: float = 0.5,
) -> List[Tuple[int, int, int]]:
    """
    计算滑窗推理的起始位置列表。
    
    Args:
        image_shape: 图像大小 (D, H, W)
        patch_size: patch大小 (D, H, W)
        step_size: 步长比例，0.5表示50%重叠
        
    Returns:
        起始位置列表 [(z1, y1, x1), ...]
    """
    steps = []
    
    for dim_idx in range(3):
        img_size = image_shape[dim_idx]
        patch_dim = patch_size[dim_idx]
        
        if img_size <= patch_dim:
            # 如果图像比patch小，只需要一个位置（从0开始）
            steps.append([0])
        else:
            # 计算步长
            step = max(int(patch_dim * step_size), 1)
            
            # 生成起始位置
            positions = list(range(0, img_size - patch_dim + 1, step))
            
            # 确保最后一个位置正好覆盖到图像末尾
            if positions[-1] + patch_dim < img_size:
                positions.append(img_size - patch_dim)
            
            steps.append(positions)
    
    # 生成所有组合
    import itertools
    all_steps = list(itertools.product(steps[0], steps[1], steps[2]))
    
    return all_steps


def sliding_window_inference(
    image: np.ndarray,
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    num_classes: int,
    step_size: float = 0.5,
    device: str = 'cuda',
    use_gaussian: bool = True,
    original_spacing: Optional[Tuple[float, float, float]] = None,
    target_spacing: Optional[Tuple[float, float, float]] = None,
    clip_lower: float = -200.0,
    clip_upper: float = 300.0,
    normalization: str = 'ct',  # 'ct' (z-score) or 'percentile' (1-99% -> [0,1])
) -> np.ndarray:
    """\
    使用滑窗推理和高斯加权拼接进行预测（参考 nnUNet）。
    
    Args:
        image: 输入图像 (D, H, W)
        model: PyTorch模型
        patch_size: patch大小
        num_classes: 类别数（包括背景）
        step_size: 滑窗步长比例，0.5表示50%重叠
        device: 设备
        use_gaussian: 是否使用高斯权重，False则使用均匀权重
        original_spacing: 原始spacing，用于重采样
        target_spacing: 目标spacing，用于重采样
        clip_lower: HU值下界，用于CT强度归一化
        clip_upper: HU值上界，用于CT强度归一化
        normalization: 'ct' (z-score) 或 'percentile' (1-99% -> [0,1])
        
    Returns:
        预测的分割结果 (D, H, W)，值为类别索引
    """
    model.eval()
    
    # 保存原始shape (在spacing归一化之前)
    original_shape = image.shape
    
    # 1. Spacing 归一化
    if original_spacing is not None and target_spacing is not None:
        image = resample_image_to_spacing(image, original_spacing, target_spacing, order=3)
    
    # 保存重采样后的shape (用于后续移除padding)
    resampled_shape = image.shape
    
    # 2. CT 强度归一化
    if normalization == 'percentile':
        image = normalize_percentile(image, lower=1.0, upper=99.0, b_min=0.0, b_max=1.0)
    else:
        image = normalize_ct(image, clip_lower=clip_lower, clip_upper=clip_upper)
    
    # 3. Padding到patch_size的倍数
    d, h, w = image.shape
    pd, ph, pw = patch_size
    
    # 计算需要的padding
    pad_d = max(pd - d, 0)
    pad_h = max(ph - h, 0)
    pad_w = max(pw - w, 0)
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = _pad_to_shape_3d(image, (max(d, pd), max(h, ph), max(w, pw)))
        d, h, w = image.shape
    
    # 4. 准备累加数组
    # aggregated_results: (num_classes, D, H, W) 累加预测概率
    # aggregated_weights: (D, H, W) 累加权重
    aggregated_results = np.zeros((num_classes, d, h, w), dtype=np.float32)
    aggregated_weights = np.zeros((d, h, w), dtype=np.float32)
    
    # 5. 计算高斯权重
    if use_gaussian:
        gaussian_weight = compute_gaussian_weights(patch_size)
    else:
        gaussian_weight = np.ones(patch_size, dtype=np.float32)
    
    # 6. 计算滑窗位置
    steps = compute_steps_for_sliding_window((d, h, w), patch_size, step_size)
    
    print(f"  滑窗推理: 图像shape {(d, h, w)}, patch_size {patch_size}, 共 {len(steps)} 个窗口")
    
    # 7. 滑窗推理
    with torch.no_grad():
        for idx, (z1, y1, x1) in enumerate(steps):
            z2, y2, x2 = z1 + pd, y1 + ph, x1 + pw
            
            # 提取patch
            patch = image[z1:z2, y1:y2, x1:x2]
            
            # 转换为tensor并添加batch和channel维度
            patch_tensor = torch.from_numpy(patch[None, None, ...]).float().to(device)
            
            # 模型推理
            output = model(patch_tensor)  # (1, num_classes, D, H, W) or list
            
            # 处理deep supervision的情况（如果返回列表，只取主输出）
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            # Softmax得到概率
            prob = torch.softmax(output, dim=1).cpu().numpy()[0]  # (num_classes, D, H, W)
            
            # 应用高斯权重并累加
            for c in range(num_classes):
                aggregated_results[c, z1:z2, y1:y2, x1:x2] += prob[c] * gaussian_weight
            
            aggregated_weights[z1:z2, y1:y2, x1:x2] += gaussian_weight
            
            if (idx + 1) % 10 == 0 or (idx + 1) == len(steps):
                print(f"    进度: {idx + 1}/{len(steps)}")
    
    # 8. 归一化（加权平均）
    # 避免除以0
    aggregated_weights = np.maximum(aggregated_weights, 1e-8)
    
    for c in range(num_classes):
        aggregated_results[c] /= aggregated_weights
    
    # 9. 取最大概率的类别
    prediction = np.argmax(aggregated_results, axis=0).astype(np.uint8)
    
    # 10. 移除padding (裁剪回重采样后的原始shape)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        prediction = prediction[:resampled_shape[0], :resampled_shape[1], :resampled_shape[2]]
    
    # 11. 重采样回原始spacing
    if original_spacing is not None and target_spacing is not None:
        prediction = resample_image_to_spacing(
            prediction.astype(np.float32),
            target_spacing,
            original_spacing,
            order=0  # 使用最近邻插值
        ).astype(np.uint8)
        
        # 12. 确保重采样后的形状与原始图像一致 (处理舍入误差)
        if prediction.shape != original_shape:
            print(f"    警告: 重采样后形状 {prediction.shape} 与原始形状 {original_shape} 不一致,进行裁剪/填充")
            # 使用裁剪或填充来匹配原始形状
            prediction = _pad_or_crop_to_shape_3d(prediction, original_shape)
    
    return prediction
