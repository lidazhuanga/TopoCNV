"""共享标签加载与二值 Y 数据集适配，保留历史标签覆盖顺序。"""

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import nrrd as _nrrd
except ImportError:
    _nrrd = None


def convert_label_to_binary_Y(label):
    """将原始标签 1..4 合并为 int64 前景；标签 5 仍为背景。"""
    lab = np.asarray(label)
    return ((lab >= 1) & (lab <= 4)).astype(np.int64)


def load_label_volume(label_path, *, nrrd_module=_nrrd):
    """保留 baseline/SOTA 的 NIfTI 与 channel-first NRRD 读取策略。

    4D NRRD 中后出现的通道覆盖先出现的通道，并裁剪到 0..5；
    其他 NRRD 和 NIfTI 直接转 int16，不添加裁剪或方向变换。
    """
    if str(label_path).endswith('.nrrd') and nrrd_module is not None:
        data, _ = nrrd_module.read(str(label_path))
        if data.ndim == 4:
            label = np.zeros(data.shape[1:], dtype=np.int16)
            for ci in range(data.shape[0]):
                label[data[ci] > 0] = ci + 1
            return np.clip(label, 0, 5)
        return data.astype(np.int16)
    return nib.load(str(label_path)).get_fdata().astype(np.int16)


class BinaryYDataset(Dataset):
    """包装字典或序列数据集，保持原 SOTA 二值转换和张量类型。"""

    def __init__(self, base, convert_label=convert_label_to_binary_Y):
        self.base = base
        self.convert_label = convert_label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        if isinstance(item, dict):
            item = dict(item)
            for key in ("label", "label_meta"):
                if key in item and item[key] is not None:
                    lab = item[key]
                    lab_np = lab.numpy() if torch.is_tensor(lab) else np.asarray(lab)
                    item[key] = torch.as_tensor(self.convert_label(lab_np))
            return item
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            img, lab = item[0], item[1]
            lab_np = lab.numpy() if torch.is_tensor(lab) else np.asarray(lab)
            lab_t = torch.as_tensor(self.convert_label(lab_np),
                                    dtype=lab.dtype if torch.is_tensor(lab) else torch.long)
            return (img, lab_t) + tuple(item[2:])
        return item


__all__ = ["BinaryYDataset", "convert_label_to_binary_Y", "load_label_volume"]
