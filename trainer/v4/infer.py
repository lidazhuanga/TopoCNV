"""Full-volume / ROI sliding-window inference for v4 (same window settings as the SOTA validator:
128^3, overlap 1 - step_size, gaussian weights with sigma_scale 0.125, sw_batch_size 2)."""
import numpy as np
import torch

from .common import box_shape, box_slices, boxes_mask, case_shift, load_image, region_from_box, union_box
from .roi import roi_boxes

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16
else:
    AMP_DTYPE = None


def sliding_window_logits(model, x, patch, step_size, device):
    """x: float32 [C, D, H, W] numpy -> logits float32 torch on device [K, D, H, W]."""
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
    """Union of the ROI boxes grown by half a patch on each side (training patches centred in the ROI
    see that much context), at least one patch large, clipped to the volume."""
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
    """[C, *region] input: CT plus the configured prior channels."""
    from .common import PRIOR_NAMES
    from .data import load_prior
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
    """Returns (pred_y bool [D,H,W], vis uint8 [D,H,W] or None, roi_used bool)."""
    from .data import load_roi_record
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
    """Stage-1 softmax over (bg, class 1, class 4) on the whole volume -> uint8 (p1, p4)."""
    image = load_image(case_id)
    logits = sliding_window_logits(model, np.asarray(image, dtype=np.float32)[None], patch, step_size, device)
    prob = torch.softmax(logits, dim=0)
    p = (prob[1:] * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    del logits, prob
    return p[0], p[1]


def save_label_nrrd(label, path):
    """int16 gzip NRRD, same header as the P0 / SOTA OOF labels."""
    import nrrd
    header = {"type": "short", "dimension": 3, "sizes": list(label.shape),
              "kinds": ["domain", "domain", "domain"], "encoding": "gzip", "endian": "little"}
    nrrd.write(str(path), np.asarray(label).astype(np.int16), header)
