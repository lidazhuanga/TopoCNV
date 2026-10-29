"""Shared helpers for the mechanism analysis (inference only, no training).

Imports the Stage-2 trainer and the protocol metrics read-only; nothing in
trainer/ or MIA/scripts/ is modified. Paths follow env_cnv.sh.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "MIA" / "scripts", REPO / "trainer", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("CNV_DATA_ROOT", str(REPO / "npc_data" / "train"))
os.environ.setdefault("CNV_RESULTS_ROOT", str(REPO / "results"))

import topocnv.metrics.protocol as mp  # noqa: E402

DATA_ROOT = Path(os.environ["CNV_DATA_ROOT"])
RESULTS = Path(os.environ["CNV_RESULTS_ROOT"])
SPACING = (2.5, 1.1328120231628418, 1.1328120231628418)
CACHE = RESULTS / "preprocess_cache" / "sp_2.500_1.133_1.133"
S1_DIR = RESULTS / ("npc_vesselFM3D_cascade_MIA_v3_vit_memory_stage1"
                    "_s1_6class_protocol_v03_20261016_cv_predictions")
S2_TEMPLATE = "npc_vesselFM3D_cascade_MIA_v3_vit_memory_stage2"

# Arms analysed here (flags as in MIA/scripts/ablation_configs.py).
ARMS = {
    "Full": dict(suffix="_p0_binary_y_full", mb=True, gate=True, alpha=None),
    "A_mem": dict(suffix="_p0_binary_y_a_mem", mb=False, gate=False, alpha=None),
    "A_gate": dict(suffix="_p0_binary_y_a_gate", mb=True, gate=False, alpha=0.5),
}

_T = None


def trainer():
    """The Stage-2 training driver, imported as a module (main() is not run)."""
    global _T
    if _T is None:
        path = REPO / "trainer" / "npc_vesselFM3D_cascade_MIA_v3_vit_memory.py"
        spec = importlib.util.spec_from_file_location("cnv_s2_trainer", path)
        _T = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_T)
    return _T


def fold_cases(fold):
    T = trainer()
    cases = T._get_all_cases(DATA_ROOT)
    ids = sorted(set(c for c, _, _ in cases))
    folds = T._create_folds(ids, 5, seed=42)
    return sorted(folds[fold])


def all_cases():
    T = trainer()
    return sorted(set(c for c, _, _ in T._get_all_cases(DATA_ROOT)))


def ckpt_path(arm, fold):
    return RESULTS / f"{S2_TEMPLATE}{ARMS[arm]['suffix']}" / f"fold{fold}" / "best_model.pth"


def fold_completed(arm, fold):
    return (ckpt_path(arm, fold).parent / "COMPLETED").exists()


def build_model(arm, fold, device):
    """BinaryYStage2Model for one arm/fold with the P0 training flags, best checkpoint loaded."""
    import torch
    T = trainer()
    cfg = ARMS[arm]
    T.GATE_ALPHA_FIXED = cfg["alpha"]
    args = argparse.Namespace(
        num_prototypes_per_class=T.NUM_PROTOTYPES_PER_CLASS,
        memory_momentum=T.MEMORY_MOMENTUM, memory_temperature=T.MEMORY_TEMPERATURE,
        memory_top_k=T.MEMORY_TOP_K, retrieval_mode="cross_attention",
        memory_type="anatomical_v4", backbone="nnunet_mla", use_mla=1,
        dual_path_stages=[0, 1], residual_conv_stages=[0],
        bone_norm_thr=T.DEFAULT_BONE_NORM_THR, evidence_temp=T.DEFAULT_EVIDENCE_TEMP)
    model = T.build_binary_y_stage2_model(args, cfg["mb"], use_gate=cfg["gate"], in_channels=3)
    ck = torch.load(ckpt_path(arm, fold), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])          # strict
    mb = getattr(model.net, "memory_bank", None)
    if mb is not None and not bool(mb.enabled):
        raise RuntimeError(f"{arm} fold{fold}: memory bank disabled in checkpoint")
    model.to(device).eval()
    return model, int(ck.get("best_epoch", -1))


def load_case(cid, with_hu=True):
    """Normalized CT, labels, cascade prior and (optionally) raw HU on the cache grid."""
    T = trainer()
    img, lbl = T._load_cached_case(cid, CACHE, mmap=True)
    img = np.array(img, dtype=np.float32)
    lbl = np.array(lbl, dtype=np.int16)
    prior = np.array(T._load_cascade_prior(S1_DIR, cid), dtype=np.float32)
    hu = None
    if with_hu:
        hu, _ = mp.load_raw_hu_dhw(cid, DATA_ROOT, target_spacing_zyx=SPACING,
                                   reference_shape_dhw=img.shape)
    return img, lbl, prior, hu


def dist_to(mask, r_max_mm):
    """Euclidean distance (mm) to ``mask``, exact up to r_max_mm (inf beyond the crop)."""
    from scipy.ndimage import distance_transform_edt
    pad = [int(np.ceil(r_max_mm / s)) + 2 for s in SPACING]
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + np.array(pad) + 1, mask.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    out = np.full(mask.shape, np.inf, dtype=np.float32)
    out[sl] = distance_transform_edt(~mask[sl], sampling=SPACING)
    return out


def regions(lbl, hu):
    """Protocol masks (Y, H, L) plus background shells around Y, all boolean.

    YH = Y∩H, L = Y\\H (protocol L1), ring5 = 0-5 mm outside Y, bg20 = 5-20 mm outside Y.
    """
    Y, H, L = mp.build_L1(lbl, hu, SPACING)
    d = dist_to(Y, 20.0)
    return dict(Y=Y, H=H, L=L, YH=Y & H, ring5=(d <= 5.0) & ~Y,
                bg20=(d > 5.0) & (d <= 20.0), roi20=d <= 20.0, dist=d)


def evidence_map(img, prior):
    """g(x) on the whole volume with the training-time definition (CPU)."""
    import torch
    from cnv_protocol_s2 import evidence_map_g1
    x = torch.from_numpy(np.concatenate([img[None], prior], 0)[None])
    with torch.no_grad():
        g = evidence_map_g1(x)[0, 0].numpy()
    return g


def bbox(mask, pad, shape=None):
    shape = shape or mask.shape
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
