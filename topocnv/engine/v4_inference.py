"""v4 全体积与 ROI 滑窗推理，保留 SOTA 窗口、融合及导出设置。"""

import numpy as np
import torch

from topocnv.data.utils import PRIOR_NAMES, load_image
from topocnv.data.v4_dataset import load_prior, load_roi_record
from topocnv.priors.roi import roi_boxes
from topocnv.utils.box_ops import box_shape, box_slices, boxes_mask, case_shift, region_from_box, union_box

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16
else:
    AMP_DTYPE = None


def sliding_window_logits(model, x, patch, step_size, device):
    """将 numpy 输入 [C,D,H,W] 转为设备上的 float32 logits [K,D,H,W]。"""
    from monai.inferers import sliding_window_inference
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None].to(device)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE or torch.float16,
                                             enabled=AMP_DTYPE is not None and device.type == "cuda"):
        out = sliding_window_inference(t, roi_size=patch, sw_batch_size=2, predictor=model,
                                       overlap=1.0 - step_size, mode="gaussian", sigma_scale=0.125)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out[0].float()


def inference_region(boxes, shape, patch):
    """扩展 ROI 并集半个 patch，至少覆盖一个完整 patch，并裁剪至体积边界。"""
    u = union_box(boxes)
    reg = []
    for (b0, b1), p, n in zip(u, patch, shape):
        lo, hi = max(b0 - p // 2, 0), min(b1 + p // 2, n)
        if hi - lo < p:
            lo = max(min(lo, n - p), 0)
            hi = min(lo + p, n)
        reg.append([int(lo), int(hi)])
    return reg


def stage2_input(case_id, fold, cfg, region, image=None):
    """构造 region 内的 CT 与所选先验通道。"""
    image = load_image(case_id) if image is None else image
    chans = [np.asarray(image[box_slices(region)], dtype=np.float32)[None]]
    if cfg.get("priors"):
        arr, pbox = load_prior(fold, case_id)
        shift = case_shift(case_id, image.shape) if cfg.get("prior_shuffle") else None
        pri = region_from_box(np.asarray(arr), pbox, region, full_shape=image.shape, shift=shift)
        idx = [PRIOR_NAMES.index(p) for p in cfg["priors"]]
        chans.append(pri[idx].astype(np.float32) / 255.0)
    return np.concatenate(chans, axis=0)


def predict_stage2(model, case_id, fold, cfg, device):
    """返回全体积 Y 二值预测、可选可见度标签及是否使用 ROI。"""
    tr = cfg["train"]
    patch = tuple(tr["patch"])
    image = load_image(case_id)
    shape = image.shape
    boxes = []
    if cfg.get("roi", {}).get("enabled"):
        boxes = roi_boxes(load_roi_record(fold, case_id), cfg["roi"]["margin_mm"])
    region = inference_region(boxes, shape, patch) if boxes else [[0, n] for n in shape]
    x = stage2_input(case_id, fold, cfg, region, image)
    logits = sliding_window_logits(model, x, patch, tr["step_size"], device)
    prob_y = torch.softmax(logits[:2], dim=0)[1]
    pred = (prob_y > 0.5).cpu().numpy()
    vis = None
    if logits.shape[0] > 2:
        vis = torch.argmax(logits[2:], dim=0).cpu().numpy().astype(np.uint8)
    if boxes:
        m = boxes_mask(boxes, region)
        pred &= m
        if vis is not None:
            vis[~m] = 0
    full = np.zeros(shape, dtype=bool)
    full[box_slices(region)] = pred
    vfull = None
    if vis is not None:
        vfull = np.zeros(shape, dtype=np.uint8)
        vfull[box_slices(region)] = vis
    del logits, prob_y
    return full, vfull, bool(boxes)


def predict_stage1(model, case_id, patch, step_size, device):
    """全体积三类 softmax 转为类别 1 与类别 4 的 uint8 概率。"""
    image = load_image(case_id)
    logits = sliding_window_logits(model, np.asarray(image, dtype=np.float32)[None], patch, step_size, device)
    prob = torch.softmax(logits, dim=0)
    p = (prob[1:] * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    del logits, prob
    return p[0], p[1]


def save_label_nrrd(label, path):
    """以 P0/SOTA 一致的头信息保存 int16 gzip NRRD。"""
    import nrrd
    header = {"type": "short", "dimension": 3, "sizes": list(label.shape),
              "kinds": ["domain", "domain", "domain"], "encoding": "gzip", "endian": "little"}
    nrrd.write(str(path), np.asarray(label).astype(np.int16), header)


__all__ = ["AMP_DTYPE", "sliding_window_logits", "inference_region", "stage2_input", "predict_stage2", "predict_stage1", "save_label_nrrd"]
