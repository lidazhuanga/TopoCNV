"""v4 Stage-1/Stage-2 patch 数据集，保持 SOTA 裁剪、增强与先验采样行为。"""

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from topocnv.data.utils import CACHE_DIR, PRIOR_NAMES, PRIORS_DIR, SKEL_DIR, load_image, load_label
from topocnv.data import augmentation as pp
from topocnv.priors.roi import roi_boxes
from topocnv.utils.box_ops import boxes_mask, case_shift, region_from_box


def load_fg(case_id):
    p = CACHE_DIR / f"{case_id}_fg_indices.npz"
    if not p.exists():
        return None
    d = np.load(p)
    _, h, w = [int(x) for x in d["shape"]]
    r = d["indices"]
    return np.stack([r // (h * w), (r % (h * w)) // w, r % w], axis=-1)


def prior_dir(fold):
    """整数 fold 转为 fold{k}；字符串标签保持原样，用于 dev/report 层。"""
    return PRIORS_DIR / (fold if isinstance(fold, str) else f"fold{fold}")


def load_roi_record(fold, case_id):
    with open(prior_dir(fold) / f"{case_id}_roi.json") as f:
        return json.load(f)


def load_prior(fold, case_id):
    """返回三通道 uint8 先验数组及其在全体积内的包围盒。"""
    rec = load_roi_record(fold, case_id)
    arr = np.load(prior_dir(fold) / f"{case_id}_prior.npy", mmap_mode="r")
    return arr, rec["prior_box"]


def load_skeleton(case_id):
    with open(SKEL_DIR / f"{case_id}_skel.json") as f:
        box = json.load(f)["box"]
    return np.load(SKEL_DIR / f"{case_id}_skel.npy", mmap_mode="r"), box


def stage1_target(raw):
    """Stage-1 标签为背景 0、类别 1 对应 1、类别 4 对应 2。"""
    t = np.zeros(raw.shape, dtype=np.int64)
    t[raw == 1] = 1
    t[raw == 4] = 2
    return t


def _augment(arrays, image_idx=0):
    """联合空间增强所有数组，强度增强仅作用于 CT 通道。"""
    for ax in range(3):
        if np.random.rand() < 0.5:
            arrays = [np.flip(a, axis=a.ndim - 3 + ax) for a in arrays]
    k = np.random.randint(0, 4)
    if k > 0:
        arrays = [np.rot90(a, k=k, axes=(a.ndim - 2, a.ndim - 1)) for a in arrays]
    img = arrays[image_idx]
    ct = pp.random_gamma(np.ascontiguousarray(img[0]), p=0.15, gamma_range=(0.7, 1.5))
    ct = pp.random_gaussian_noise(ct, p=0.1, sigma_range=(0.0, 0.05))
    img = np.array(img, dtype=np.float32)
    img[0] = np.clip(ct, 0.0, 1.0)
    arrays[image_idx] = img
    return arrays


class V4PatchDataset(Dataset):
    """stage=1 返回 (x, target)；stage=2 返回 x、原标签、ROI 与骨架字典。"""

    def __init__(self, cases, patch_size, stage, fold=None, cfg=None, random_crop_prob=0.33):
        self.cases = list(cases)
        self.patch = tuple(patch_size)
        self.stage = stage
        self.fold = fold
        self.cfg = cfg or {}
        self.rcp = float(random_crop_prob)
        self.roi_m = self.cfg.get("roi", {}).get("margin_mm") if self.cfg.get("roi", {}).get("enabled") else None
        self.prior_idx = [PRIOR_NAMES.index(p) for p in self.cfg.get("priors", [])]
        self.shuffle = bool(self.cfg.get("prior_shuffle", False))
        self.use_skel = bool(self.cfg.get("skeleton", {}).get("enabled", False))
        self.fg, self.roi, self.fg_roi = {}, {}, {}
        for c in self.cases:
            fg = load_fg(c)
            self.fg[c] = fg
            if stage == 2 and self.roi_m is not None:
                boxes = roi_boxes(load_roi_record(fold, c), self.roi_m)
                self.roi[c] = boxes
                if boxes and fg is not None:
                    inside = np.zeros(len(fg), dtype=bool)
                    for b in boxes:
                        inside |= np.all((fg >= [x[0] for x in b]) & (fg < [x[1] for x in b]), axis=1)
                    self.fg_roi[c] = fg[inside]

    def __len__(self):
        return len(self.cases)

    def _corner(self, c, shape):
        pd, ph, pw = self.patch
        d, h, w = shape
        boxes = self.roi.get(c) if self.roi_m is not None else None
        if self.rcp > 0 and np.random.rand() < self.rcp:
            if boxes:
                vols = np.array([np.prod([b1 - b0 for b0, b1 in b]) for b in boxes], dtype=np.float64)
                b = boxes[np.random.choice(len(boxes), p=vols / vols.sum())]
                cz, cy, cx = [np.random.randint(b0, b1) for b0, b1 in b]
                return (int(np.clip(cz - pd // 2, 0, d - pd)), int(np.clip(cy - ph // 2, 0, h - ph)),
                        int(np.clip(cx - pw // 2, 0, w - pw)))
            return (np.random.randint(0, max(1, d - pd + 1)), np.random.randint(0, max(1, h - ph + 1)),
                    np.random.randint(0, max(1, w - pw + 1)))
        fg = self.fg_roi.get(c) if boxes else self.fg.get(c)
        if fg is None or len(fg) == 0:
            fg = np.argwhere(np.asarray(load_label(c)) > 0)
        if len(fg) == 0:
            return (np.random.randint(0, max(1, d - pd + 1)), np.random.randint(0, max(1, h - ph + 1)),
                    np.random.randint(0, max(1, w - pw + 1)))
        cz, cy, cx = fg[np.random.randint(len(fg))]
        return (int(np.clip(cz - pd // 2, 0, d - pd)), int(np.clip(cy - ph // 2, 0, h - ph)),
                int(np.clip(cx - pw // 2, 0, w - pw)))

    def __getitem__(self, i):
        c = self.cases[i]
        img_mm, lab_mm = load_image(c), load_label(c)
        shape = img_mm.shape
        if any(s < p for s, p in zip(shape, self.patch)):
            raise ValueError(f"{c}: volume {shape} smaller than patch {self.patch}")
        z1, y1, x1 = self._corner(c, shape)
        region = [[z1, z1 + self.patch[0]], [y1, y1 + self.patch[1]], [x1, x1 + self.patch[2]]]
        sl = tuple(slice(a, b) for a, b in region)
        image = np.array(img_mm[sl], dtype=np.float32)[None]
        label = np.array(lab_mm[sl], dtype=np.int16)

        if self.stage == 1:
            image, label = _augment([image, label])
            x = torch.from_numpy(np.ascontiguousarray(image)).float()
            return x, torch.from_numpy(stage1_target(np.ascontiguousarray(label)))

        chans = [image]
        if self.prior_idx:
            arr, pbox = load_prior(self.fold, c)
            shift = case_shift(c, shape) if self.shuffle else None
            pri = region_from_box(arr, pbox, region, full_shape=shape, shift=shift)[self.prior_idx]
            chans.append(pri.astype(np.float32) / 255.0)
        x = np.concatenate(chans, axis=0)
        if self.roi_m is not None and self.roi.get(c):
            roi = boxes_mask(self.roi[c], region).astype(np.uint8)
        else:
            roi = np.ones(self.patch, dtype=np.uint8)
        if self.use_skel:
            sk, sbox = load_skeleton(c)
            skel = region_from_box(sk, sbox, region).astype(np.uint8)
        else:
            skel = np.zeros(self.patch, dtype=np.uint8)
        x, label, roi, skel = _augment([x, label, roi, skel])
        return {"x": torch.from_numpy(np.ascontiguousarray(x)).float(),
                "label": torch.from_numpy(np.ascontiguousarray(label).astype(np.int64)),
                "roi": torch.from_numpy(np.ascontiguousarray(roi)),
                "skel": torch.from_numpy(np.ascontiguousarray(skel))}


__all__ = ["V4PatchDataset", "load_fg", "prior_dir", "load_roi_record", "load_prior", "load_skeleton", "stage1_target", "_augment"]
