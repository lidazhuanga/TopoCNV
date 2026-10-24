"""nnU-Net v2 trainers for the v4 arms (protocol v0.4-draft §3–§4, docs/V4_PLAN.md §4).

Installed into the nnU-Net venv as a symlink nnunetv2/training/nnUNetTrainer/variants/zz_cnv_v4/
(tools/v4_nnunet.py install). The folder name sorts after every stock variant folder, so nnU-Net's
trainer lookup for other trainers never imports this module. Top-level imports are nnunetv2/torch/numpy
only, for the same reason.

Common to every arm (identical to the baseline nnUNetTrainer_300epochs unless listed):
  * 300 epochs, plans copied from Dataset501 (patch, network, batch, optimiser, LR schedule, DA unchanged).
  * Data: Dataset5xx_CNVv4* (4 channels: CT + priors a, b, c; labels = original 0..5, ROI-cropped).
  * Foreground oversampling picks Y voxels (classes 1-4 merged), as the binary-Y baseline does; class 5
    is never a sampling target.
  * Main output: 2 channels (bg, Y), nnU-Net's DC_and_CE loss with deep supervision, Y = classes 1-4.
  * Prior channels not used by an arm are set to 0 after augmentation, in training and inference.
Per arm: PRIORS (kept prior channels), VIS (5-class visibility head, class 5 'background' | 'ignore',
weight 0.5), SKEL_W (Skeleton Recall Loss weight, Kirchhoff et al. ECCV 2024: tubed skeleton of the
augmented Y patch, soft recall, all deep-supervision levels).

perform_actual_validation writes validation_v4/{case}.npz (Y and visibility labels on the ROI-crop grid,
nnU-Net's nonzero crop reverted); tools/v4_nnunet.py export pastes them into the full volume.
"""
import os
from typing import List

import numpy as np
import torch
from torch import autocast, nn

from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context, empty_cache

Y_CLASSES = (1, 2, 3, 4)
PRIOR_CHANNELS = {"a": 1, "b": 2, "c": 3}
VIS_WEIGHT = 0.5

# Stock nnU-Net intensity transforms would corrupt the prior channels (probability / atlas-support
# maps): Gamma with p_invert_image=1 even flips them, so training would never see the clean priors
# that inference gets. Spatial transforms stay shared by all channels so CT, priors and labels
# remain aligned. User decision 2026-10-24: intensity DA acts on the CT channel only.
# Names as instantiated by nnUNetTrainer.get_training_transforms (batchgeneratorsv2 0.3.6);
# pinned by tests/test_v4_nnunet.py::test_intensity_list_covers_installed_pipeline.
INTENSITY_DA = ("GammaTransform", "MultiplicativeBrightnessTransform", "ContrastTransform",
                "GaussianBlurTransform", "SimulateLowResolutionTransform", "GaussianNoiseTransform")


class Channel0Only:
    """Wrap a stock intensity transform so it only sees channel 0 (CT); channels 1..3 pass through.

    batchgeneratorsv2 calls each transform per sample as transforms(**{'image': t[C, X, Y, Z],
    'segmentation': s}) — no batch axis, channel axis 0 (nnUNetDataLoader3D.generate_train_batch)."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, **data_dict):
        img = data_dict.get("image")
        if img is None or img.shape[0] < 2:
            return self.inner(**data_dict)
        np_in = not isinstance(img, torch.Tensor)
        data_dict["image"] = img[:1]
        data_dict = self.inner(**data_dict)
        out = data_dict["image"]
        if np_in:
            data_dict["image"] = np.concatenate([out[:1], img[1:]], axis=0)
        else:
            data_dict["image"] = torch.cat([out[:1], img[1:].to(out.dtype)], dim=0)
        return data_dict


# ---------------- data ----------------

def merge_y_locations(class_locations):
    """nnU-Net class_locations {label: (N, 4)} -> {1: locations of classes 1-4}; class 5 dropped."""
    if class_locations is None:
        return None
    locs = [class_locations[c] for c in Y_CLASSES if c in class_locations and len(class_locations[c])]
    return {1: np.concatenate(locs, 0)} if locs else {1: np.zeros((0, 4), dtype=np.int64)}


class YSamplingDataLoader3D(nnUNetDataLoader3D):
    """Foreground oversampling on Y = classes 1-4 merged (the binary-Y baseline's single foreground class)."""

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None, verbose=False):
        return super().get_bbox(data_shape, force_fg, merge_y_locations(class_locations), overwrite_class, verbose)


def tubed_skeleton(y: np.ndarray) -> np.ndarray:
    """3D skeleton of a binary mask, dilated twice (skimage default footprint), restricted to the mask —
    the 'tube' variant of the official Skeleton Recall nnU-Net implementation."""
    from skimage.morphology import dilation, skeletonize
    out = np.zeros(y.shape, dtype=np.int16)
    idx = np.nonzero(y)
    if len(idx[0]) == 0:
        return out
    lo = [max(int(i.min()) - 3, 0) for i in idx]
    hi = [min(int(i.max()) + 4, n) for i, n in zip(idx, y.shape)]
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    sub = y[sl]
    sk = (skeletonize(sub) > 0).astype(np.int16)
    sk = dilation(dilation(sk))
    out[sl] = sk * sub
    return out


def make_skeleton_transform():
    from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

    class SkeletonTransform(BasicTransform):
        """Appends the tubed skeleton of Y (classes 1-4 of segmentation channel 0) as segmentation channel 1."""

        def apply(self, data_dict, **params):
            seg = data_dict["segmentation"]
            y = ((seg[0] >= 1) & (seg[0] <= 4)).numpy()
            sk = torch.from_numpy(tubed_skeleton(y)).to(seg.dtype)
            data_dict["segmentation"] = torch.cat([seg, sk[None]], 0)
            return data_dict

    return SkeletonTransform()


# ---------------- loss ----------------

class SoftSkeletonRecall(nn.Module):
    """1 - soft recall of the tubed skeleton (batch-level sums, smooth 1e-5)."""

    def forward(self, prob_y, skel):
        s = skel.float()
        return 1.0 - ((prob_y * s).sum() + 1e-5) / (s.sum() + 1e-5)


class V4Loss(nn.Module):
    def __init__(self, ds_weights, batch_dice, vis=None, skel_w=0.0):
        super().__init__()
        self.w = [float(x) for x in ds_weights]
        kw = {"batch_dice": batch_dice, "smooth": 1e-5, "do_bg": False, "ddp": False}
        self.main = DC_and_CE_loss(kw, {}, weight_ce=1, weight_dice=1, ignore_label=None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        self.vis = vis
        if vis is not None:
            self.vis_loss = DC_and_CE_loss(kw, {}, weight_ce=1, weight_dice=1,
                                           ignore_label=5 if vis == "ignore" else None,
                                           dice_class=MemoryEfficientSoftDiceLoss)
        self.skel_w = float(skel_w)
        self.skel = SoftSkeletonRecall()

    def forward(self, outputs: List[torch.Tensor], targets: List[torch.Tensor]):
        total = 0.0
        for w, out, tgt in zip(self.w, outputs, targets):
            if w == 0:
                continue
            raw = tgt[:, :1]
            y = ((raw >= 1) & (raw <= 4)).to(raw.dtype)
            loss = self.main(out[:, :2], y)
            if self.vis is not None:
                tv = torch.where((raw >= 1) & (raw <= 4), raw, torch.zeros_like(raw))
                if self.vis == "ignore":
                    tv = torch.where(raw == 5, torch.full_like(raw, 5), tv)
                loss = loss + VIS_WEIGHT * self.vis_loss(out[:, 2:7], tv)
            if self.skel_w > 0:
                p = torch.softmax(out[:, :2].float(), 1)[:, 1]
                loss = loss + self.skel_w * self.skel(p, tgt[:, 1])
            total = total + w * loss
        return total


# ---------------- trainer ----------------

class nnUNetTrainerV4(nnUNetTrainer):
    PRIORS = ()          # kept prior channels, subset of ('a', 'b', 'c')
    VIS = None           # None | 'background' | 'ignore'
    SKEL_W = 0.0
    N_OUT = 2

    @staticmethod
    def wrap_intensity_ct_only(tr_t):
        """Intensity DA sees channel 0 (CT) only; priors keep their clean values.
        batchgeneratorsv2 hides most augs inside RandomTransform (probability wrapper),
        so both the top level and the wrapped transform are matched by class name."""
        out = []
        for t in tr_t.transforms:
            name = type(t).__name__
            if name == "RandomTransform" and type(getattr(t, "transform", None)).__name__ in INTENSITY_DA:
                t.transform = Channel0Only(t.transform)
            elif name in INTENSITY_DA:
                t = Channel0Only(t)
            out.append(t)
        tr_t.transforms = out
        return tr_t

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
        keep = {PRIOR_CHANNELS[p] for p in self.PRIORS}
        self.zero_channels = [c for c in PRIOR_CHANNELS.values() if c not in keep]

    @classmethod
    def _n_out(cls):
        return 2 + (5 if cls.VIS else 0)

    @staticmethod
    def _build(arch_name, kw, kw_imp, n_in, n_out, ds):
        return get_network_from_plans(arch_name, kw, kw_imp, n_in, n_out, allow_init=True, deep_supervision=ds)

    def _mask(self, data):
        if self.zero_channels and data.shape[1] > max(self.zero_channels):
            data = data.clone()
            data[:, self.zero_channels] = 0
        return data

    def _build_loss(self):
        scales = self._get_deep_supervision_scales()
        w = np.array([1 / (2 ** i) for i in range(len(scales))])
        w[-1] = 0
        w = w / w.sum()
        return V4Loss(w, self.configuration_manager.batch_dice, self.VIS, self.SKEL_W)

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        ds_scales = self._get_deep_supervision_scales()
        rot, dummy2d, init_patch, mirror_axes = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        tr_t = nnUNetTrainer.get_training_transforms(
            patch_size, rot, ds_scales, mirror_axes, dummy2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm, is_cascaded=False,
            foreground_labels=self.label_manager.foreground_labels, regions=None, ignore_label=None)
        self.wrap_intensity_ct_only(tr_t)
        if self.SKEL_W > 0:
            ts = tr_t.transforms
            i = len(ts) - 1 if ds_scales is not None else len(ts)  # before DownsampleSegForDSTransform
            ts.insert(i, make_skeleton_transform())
        val_t = nnUNetTrainer.get_validation_transforms(ds_scales, is_cascaded=False,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=None, ignore_label=None)
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = YSamplingDataLoader3D(dataset_tr, self.batch_size, init_patch, patch_size, self.label_manager,
                                      oversample_foreground_percent=self.oversample_foreground_percent,
                                      sampling_probabilities=None, pad_sides=None, transforms=tr_t)
        dl_val = YSamplingDataLoader3D(dataset_val, self.batch_size, patch_size, patch_size, self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=val_t)
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
        n = get_allowed_n_proc_DA()
        if n == 0:
            mt_tr, mt_val = SingleThreadedAugmenter(dl_tr, None), SingleThreadedAugmenter(dl_val, None)
        else:
            mt_tr = NonDetMultiThreadedAugmenter(dl_tr, None, n, max(6, n // 2), None,
                                                 pin_memory=self.device.type == "cuda", wait_time=0.002)
            mt_val = NonDetMultiThreadedAugmenter(dl_val, None, max(1, n // 2), max(3, n // 4), None,
                                                  pin_memory=self.device.type == "cuda", wait_time=0.002)
        _ = next(mt_tr)
        _ = next(mt_val)
        return mt_tr, mt_val

    def train_step(self, batch):
        data = self._mask(batch["data"].to(self.device, non_blocking=True))
        target = [t.to(self.device, non_blocking=True) for t in batch["target"]]
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            out = self.network(data)
            loss = self.loss(out, target)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch):
        data = self._mask(batch["data"].to(self.device, non_blocking=True))
        target = [t.to(self.device, non_blocking=True) for t in batch["target"]]
        if self.SKEL_W > 0:  # validation targets carry no skeleton channel
            target = [torch.cat([t, torch.zeros_like(t)], 1) for t in target]
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            out = self.network(data)
            loss = self.loss(out, target)
        o, t = out[0][:, :2], target[0][:, :1]
        y = ((t >= 1) & (t <= 4)).long()
        pred = torch.zeros(o.shape, device=o.device, dtype=torch.float32)
        pred.scatter_(1, o.argmax(1, keepdim=True), 1)
        tp, fp, fn, _ = get_tp_fp_fn_tn(pred, y, axes=[0] + list(range(2, o.ndim)))
        return {"loss": loss.detach().cpu().numpy(), "tp_hard": tp.detach().cpu().numpy()[1:],
                "fp_hard": fp.detach().cpu().numpy()[1:], "fn_hard": fn.detach().cpu().numpy()[1:]}

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Y and visibility labels of the split's validation cases -> validation_v4/{case}.npz."""
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)
        out_dir = os.path.join(self.output_folder, "validation_v4")
        os.makedirs(out_dir, exist_ok=True)
        _, dataset_val = self.get_tr_and_val_datasets()
        for k in dataset_val.keys():
            data, _, props = dataset_val.load_case(k)
            x = self._mask(torch.from_numpy(np.asarray(data, dtype=np.float32))[None])[0]
            with torch.no_grad():
                logits = predictor.predict_sliding_window_return_logits(x).float()
            y = (torch.softmax(logits[:2], 0)[1] > 0.5).cpu().numpy().astype(np.uint8)
            vis = torch.argmax(logits[2:], 0).cpu().numpy().astype(np.uint8) if logits.shape[0] > 2 else None
            full_shape = props["shape_before_cropping"]
            bbox = props["bbox_used_for_cropping"]
            # the v4 datasets are on the plans' spacing already, so nnU-Net never resamples them
            assert tuple(y.shape) == tuple(b - a for a, b in bbox), (k, y.shape, bbox)
            sl = tuple(slice(a, b) for a, b in bbox)
            Y = np.zeros(full_shape, np.uint8)
            Y[sl] = y
            save = {"Y": Y}
            if vis is not None:
                V = np.zeros(full_shape, np.uint8)
                V[sl] = vis
                save["vis"] = V
            np.savez_compressed(os.path.join(out_dir, f"{k}.npz"), **save)
            self.print_to_log_file(f"validation_v4 {k}: Y voxels {int(Y.sum())}")
            del logits
            empty_cache(self.device)
        with open(os.path.join(out_dir, "DONE"), "w") as f:
            f.write(f"{len(dataset_val.keys())} cases\n")
        self.set_deep_supervision_enabled(True)


def _arm(name, priors=(), vis=None, skel_w=0.0, smoke=False):
    n_out = 2 + (5 if vis else 0)

    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        return nnUNetTrainerV4._build(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                      num_input_channels, n_out, enable_deep_supervision)

    attrs = {"PRIORS": tuple(priors), "VIS": vis, "SKEL_W": float(skel_w), "N_OUT": n_out,
             "build_network_architecture": staticmethod(build_network_architecture)}
    if smoke:
        def __init__(self, *a, **kw):
            nnUNetTrainerV4.__init__(self, *a, **kw)
            self.num_epochs, self.num_iterations_per_epoch, self.num_val_iterations_per_epoch = 2, 5, 2
        attrs["__init__"] = __init__
    cls = type(name, (nnUNetTrainerV4,), attrs)
    globals()[name] = cls
    return cls


# R1: ROI only. R1+P: priors. R1+P+V: + visibility head. Full: + Skeleton Recall. Names encode the settings;
# the development candidates and the frozen arms are all listed here so nnU-Net can find them by name.
_arm("nnUNetTrainerV4_R1")
for _combo in ("abc", "bc", "ac", "ab", "a", "b", "c"):
    _arm(f"nnUNetTrainerV4_P{_combo}", priors=tuple(_combo))
    for _c5, _t in (("background", "bg"), ("ignore", "ign")):
        _arm(f"nnUNetTrainerV4_P{_combo}_V{_t}", priors=tuple(_combo), vis=_c5)
        for _w, _wt in ((0.5, "05"), (1.0, "10")):
            _arm(f"nnUNetTrainerV4_P{_combo}_V{_t}_S{_wt}", priors=tuple(_combo), vis=_c5, skel_w=_w)
for _c5, _t in (("background", "bg"), ("ignore", "ign")):
    for _w, _wt in ((0.5, "05"), (1.0, "10")):
        _arm(f"nnUNetTrainerV4_noP_V{_t}_S{_wt}", priors=(), vis=_c5, skel_w=_w)
# smoke test only (2 epochs x 5 iterations): tests/v4_nnunet_smoke.sh
_arm("nnUNetTrainerV4_SMOKE_Pabc_Vign_S10", priors=tuple("abc"), vis="ignore", skel_w=1.0, smoke=True)
_arm("nnUNetTrainerV4_SMOKE_R1", smoke=True)
