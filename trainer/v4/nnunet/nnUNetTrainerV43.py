"""nnU-Net v2 trainers for V4.3, route B: CT-MRI coverage-aware fusion (docs/V4_PLAN.md §12).

Installed next to nnUNetTrainerV4.py in nnunetv2/training/nnUNetTrainer/variants/zz_cnv_v4/
(tools/v4_mm43.py install43). Top-level imports are nnunetv2/torch/numpy and the V4 module only.

Input (Dataset557/558): channel 0 CT, 1 MRI ([0, 1], 0 outside the field of view), 2 MRI valid mask (0/1).
Everything else is nnUNetTrainerV4 with no priors, no visibility head and no Skeleton Recall (binary Y,
DC_and_CE with deep supervision, Y foreground oversampling, 300 epochs), so on a CT-only input it equals R0u.
  * Intensity DA acts on CT and MRI, never on the mask; spatial DA is shared by all three channels.
  * Every batch, after DA and before the network (train, validation, inference): mask <- mask > 0.5,
    MRI <- MRI * mask. In training only, then per sample (§12.4):
      case  with P_CASE: MRI and mask of the sample set to 0 (no truncation or slab then);
      trunc with P_TRUNC: tilted plane through z0 ~ U[0, Z) (axis 0), normal tilted from axis 0 by U(0, 20) deg
            with a uniform azimuth, distances in mm; the cranial (+axis 0) or caudal side (0.5 each) set to 0;
      slab  with P_SLAB (independent of trunc): 1 or 2 slabs of 4-10 slices along axis 0, set to 0.
  * perform_actual_validation writes validation_v4/ (each case's own MRI) and validation_v4_nomri/
    (channels 1-2 zeroed for every case), both from the same network state.
"""
import os
import shutil

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.variants.zz_cnv_v4.nnUNetTrainerV4 import (INTENSITY_DA, nnUNetTrainerV4)

N_IN = 3
MRI, MASK = 1, 2
TILT_MAX_DEG = 20.0
SLAB_SLICES = (4, 10)
DROP_SEED = 20261028
CANDIDATES = {"MMd1": (0.2, 0.3, 0.2), "MMd2": (0.4, 0.5, 0.3)}   # (p_case, p_trunc, p_slab), V4_PLAN §12.4


class FirstChannels:
    """Wrap a stock intensity transform so it only sees channels [0, n); the rest (the mask) passes through.
    Same calling convention as nnUNetTrainerV4.Channel0Only (per sample, image[C, X, Y, Z])."""

    def __init__(self, inner, n):
        self.inner, self.n = inner, n

    def __call__(self, **data_dict):
        img = data_dict.get("image")
        if img is None or img.shape[0] <= self.n:
            return self.inner(**data_dict)
        np_in = not isinstance(img, torch.Tensor)
        data_dict["image"] = img[:self.n]
        data_dict = self.inner(**data_dict)
        out = data_dict["image"]
        if np_in:
            data_dict["image"] = np.concatenate([out, img[self.n:]], axis=0)
        else:
            data_dict["image"] = torch.cat([out, img[self.n:].to(out.dtype)], dim=0)
        return data_dict


def draw_drop(rng, shape, p_case, p_trunc, p_slab):
    """Dropout parameters for one sample of patch shape (Z, A, B). Pure function of rng (np.random.Generator)."""
    d = {"case": bool(rng.random() < p_case)}
    if d["case"]:
        return d
    if rng.random() < p_trunc:
        tilt = np.deg2rad(rng.uniform(0.0, TILT_MAX_DEG))
        az = rng.uniform(0.0, 2 * np.pi)
        d["trunc"] = {"z0": float(rng.uniform(0.0, shape[0])), "tilt": float(tilt), "az": float(az),
                      "side": "cranial" if rng.random() < 0.5 else "caudal"}
    if rng.random() < p_slab:
        n = 1 if rng.random() < 0.5 else 2
        slabs = []
        for _ in range(n):
            t = int(rng.integers(SLAB_SLICES[0], SLAB_SLICES[1] + 1))
            t = min(t, shape[0])
            s = int(rng.integers(0, shape[0] - t + 1))
            slabs.append((s, s + t))
        d["slabs"] = slabs
    return d


def keep_mask(d, shape, spacing, device="cpu"):
    """1 where MRI and mask are kept, 0 where dropped, as a float tensor of patch shape (Z, A, B)."""
    keep = torch.ones(tuple(shape), device=device)
    if d.get("case"):
        return keep * 0
    if "trunc" in d:
        t = d["trunc"]
        n0 = np.cos(t["tilt"])
        n1 = np.sin(t["tilt"]) * np.cos(t["az"])
        n2 = np.sin(t["tilt"]) * np.sin(t["az"])
        z = (torch.arange(shape[0], device=device, dtype=torch.float32) - t["z0"]) * spacing[0]
        a = (torch.arange(shape[1], device=device, dtype=torch.float32) - (shape[1] - 1) / 2) * spacing[1]
        b = (torch.arange(shape[2], device=device, dtype=torch.float32) - (shape[2] - 1) / 2) * spacing[2]
        dist = z[:, None, None] * n0 + a[None, :, None] * n1 + b[None, None, :] * n2
        drop = dist > 0 if t["side"] == "cranial" else dist < 0
        keep = keep * (~drop).float()
    for s, e in d.get("slabs", ()):
        keep[s:e] = 0
    return keep


class nnUNetTrainerV43(nnUNetTrainerV4):
    P_CASE = P_TRUNC = P_SLAB = 0.0

    @staticmethod
    def wrap_intensity_ct_only(tr_t):
        """V4.3: intensity DA sees CT and MRI (channels 0-1); the valid mask keeps its 0/1 values.
        (The name is the hook nnUNetTrainerV4.get_dataloaders calls.)"""
        out = []
        for t in tr_t.transforms:
            name = type(t).__name__
            if name == "RandomTransform" and type(getattr(t, "transform", None)).__name__ in INTENSITY_DA:
                t.transform = FirstChannels(t.transform, 2)
            elif name in INTENSITY_DA:
                t = FirstChannels(t, 2)
            out.append(t)
        tr_t.transforms = out
        return tr_t

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.zero_channels = []           # no prior channels: MRI and mask are kept
        tf = list(self.plans_manager.transpose_forward)
        assert tf == [0, 1, 2], f"V4.3 dropout assumes patch axis 0 = cache axis 0 (cranial +), plans {tf}"
        self.drop_spacing = [float(s) for s in self.configuration_manager.spacing]
        self.drop_rng = np.random.default_rng(DROP_SEED + 1000 * int(fold) if str(fold).isdigit() else DROP_SEED)
        self._dropping = False
        self._no_mri = False
        self.drop_counts = {"samples": 0, "case": 0, "trunc": 0, "slab": 0}

    def _mask(self, data):
        assert data.shape[1] == N_IN, f"V4.3 needs CT + MRI + mask, got {data.shape[1]} channels"
        data = data.clone()
        m = (data[:, MASK] > 0.5).to(data.dtype)
        data[:, MASK] = m
        data[:, MRI] = data[:, MRI] * m
        if self._no_mri:
            data[:, MRI:] = 0
        elif self._dropping and (self.P_CASE or self.P_TRUNC or self.P_SLAB):
            shape = tuple(data.shape[2:])
            for i in range(data.shape[0]):
                d = draw_drop(self.drop_rng, shape, self.P_CASE, self.P_TRUNC, self.P_SLAB)
                self.drop_counts["samples"] += 1
                self.drop_counts["case"] += int(d.get("case", False))
                self.drop_counts["trunc"] += int("trunc" in d)
                self.drop_counts["slab"] += int("slabs" in d)
                k = keep_mask(d, shape, self.drop_spacing, data.device).to(data.dtype)
                data[i, MRI:] = data[i, MRI:] * k
        return data

    def train_step(self, batch):
        self._dropping = True
        try:
            return super().train_step(batch)
        finally:
            self._dropping = False

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.P_CASE or self.P_TRUNC or self.P_SLAB:
            self.print_to_log_file(f"V4.3 dropout so far: {self.drop_counts}")

    def perform_actual_validation(self, save_probabilities: bool = False):
        """validation_v4_nomri/ (no MRI for any case), then validation_v4/ (each case's MRI), same weights."""
        out, nomri = (os.path.join(self.output_folder, n) for n in ("validation_v4", "validation_v4_nomri"))
        self._no_mri = True
        try:
            super().perform_actual_validation(save_probabilities)
        finally:
            self._no_mri = False
        if os.path.isdir(nomri):
            shutil.rmtree(nomri)
        os.replace(out, nomri)
        super().perform_actual_validation(save_probabilities)


def _arm43(name, p_case=0.0, p_trunc=0.0, p_slab=0.0, smoke=False):
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        assert num_input_channels == N_IN, f"{name} needs CT + MRI + mask, the dataset has {num_input_channels}"
        return nnUNetTrainerV4._build(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                      num_input_channels, 2, enable_deep_supervision)

    attrs = {"P_CASE": float(p_case), "P_TRUNC": float(p_trunc), "P_SLAB": float(p_slab),
             "build_network_architecture": staticmethod(build_network_architecture)}
    if smoke:
        # exact base signature (see nnUNetTrainerV4._arm): nnU-Net rebuilds init kwargs from the frame
        def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                     device=torch.device("cuda")):
            nnUNetTrainerV43.__init__(self, plans, configuration, fold, dataset_json, unpack_dataset, device)
            self.num_epochs, self.num_iterations_per_epoch, self.num_val_iterations_per_epoch = 2, 5, 2
        attrs["__init__"] = __init__
    cls = type(name, (nnUNetTrainerV43,), attrs)
    globals()[name] = cls
    return cls


_arm43("nnUNetTrainerV43_MM")
for _n, (_pc, _pt, _ps) in CANDIDATES.items():
    _arm43(f"nnUNetTrainerV43_{_n}", _pc, _pt, _ps)
    _arm43(f"nnUNetTrainerV43_{_n.replace('MMd', 'MMdcase')}", _pc)      # MMd-case: whole-case dropout only
# smoke only (2 epochs x 5 iterations, strong dropout so every branch runs): tests/v4_nnunet_smoke43.sh
_arm43("nnUNetTrainerV43_SMOKE_MMd", 0.3, 0.6, 0.6, smoke=True)
