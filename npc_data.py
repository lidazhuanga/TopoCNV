import os
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

try:
    import nrrd as _nrrd
except ImportError:
    _nrrd = None

import npc_preprocess as pp


DATA_ROOT_DEFAULT = os.environ.get(
    "CNV_DATA_ROOT", str(Path(__file__).resolve().parent / "npc_data" / "train"))


class NPCDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        label_paths: List[Path],
        patch_size: Tuple[int, int, int],
        training: bool = True,
        target_spacing: Tuple[float, float, float] = None,
        ensure_foreground: bool = True,
        clip_lower: float = -200.0,
        clip_upper: float = 300.0,
    ) -> None:
        assert len(image_paths) == len(label_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.patch_size = patch_size
        self.training = training
        self.target_spacing = target_spacing
        self.ensure_foreground = ensure_foreground
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image_nii = nib.load(str(image_path))
        image = image_nii.get_fdata()

        # Support both NIfTI (.nii.gz) and NRRD (.nrrd) label formats
        if str(label_path).endswith('.nrrd') and _nrrd is not None:
            label, _ = _nrrd.read(str(label_path))
            # nrrd returns (x, y, z) or (z, y, x) depending on header
            # Ensure 3D array
            if label.ndim == 4:
                label = label[0]  # Take first channel if multi-channel
        else:
            label_nii = nib.load(str(label_path))
            label = label_nii.get_fdata()

        image = np.asarray(image, dtype=np.float32)
        label = np.asarray(label, dtype=np.int16)

        if image.ndim != 3:
            raise ValueError(f"Expect 3D image, got shape {image.shape} from {image_path}")
        if label.shape != image.shape:
            raise ValueError(f"Image and label shape mismatch: {image.shape} vs {label.shape} for {image_path}")

        # 获取原始 spacing (从 NIfTI header)
        # NIfTI spacing 格式为 (x, y, z)，需要转换为 (z, y, x)
        original_spacing = None
        if self.target_spacing is not None:
            nii_spacing = image_nii.header.get_zooms()[:3]  # (x, y, z)
            original_spacing = (float(nii_spacing[2]), float(nii_spacing[1]), float(nii_spacing[0]))  # (z, y, x)

        image = np.transpose(image, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        if self.training:
            image_t, label_t = pp.preprocess_training(
                image, label, self.patch_size,
                original_spacing=original_spacing,
                target_spacing=self.target_spacing,
                ensure_foreground=self.ensure_foreground,
                clip_lower=self.clip_lower,
                clip_upper=self.clip_upper,
            )
        else:
            image_t, label_t = pp.preprocess_validation(
                image, label, self.patch_size,
                original_spacing=original_spacing,
                target_spacing=self.target_spacing,
                clip_lower=self.clip_lower,
                clip_upper=self.clip_upper,
            )

        return image_t, label_t


def _get_all_cases(data_root: str) -> List[Tuple[str, Path, Path]]:
    """
    Get all cases from the data directory.
    
    Supports both NIfTI (.nii.gz) and NRRD (.nrrd) label formats.
    Prefers .nrrd if available (multi-channel, preserves class overlap).
    
    Returns:
        List of (patient_id, image_path, label_path) tuples
    """
    data_root_path = Path(data_root)
    image_dir = data_root_path / "image"
    label_dir = data_root_path / "label"

    image_files = sorted(image_dir.glob("*.nii.gz"))
    cases: List[Tuple[str, Path, Path]] = []
    for img_path in image_files:
        # img_path.stem 对 "xxx.nii.gz" 会得到 "xxx.nii"，这里额外去掉尾部的 ".nii"，得到真正的患者 ID
        pid = img_path.stem
        if pid.endswith(".nii"):
            pid = pid[:-4]
        
        # Check for NRRD first (multi-channel, preserves overlap), then NIfTI
        nrrd_path = label_dir / f"{pid}.nrrd"
        nii_path = label_dir / f"{pid}.nii.gz"
        
        if nrrd_path.exists():
            label_path = nrrd_path
        elif nii_path.exists():
            label_path = nii_path
        else:
            raise FileNotFoundError(
                f"Label file not found for {pid}: checked {nrrd_path} and {nii_path}"
            )
        
        cases.append((pid, img_path, label_path))
    return cases


def _create_folds(patient_ids: List[str], num_folds: int = 5, seed: int = 42) -> List[List[str]]:
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(patient_ids))
    idx = np.arange(len(ids))
    rng.shuffle(idx)
    splits = np.array_split(idx, num_folds)
    folds: List[List[str]] = []
    for s in splits:
        folds.append(ids[s].tolist())
    return folds


def get_train_val_datasets(
    data_root: str,
    patch_size: Tuple[int, int, int],
    fold: int,
    num_folds: int = 5,
    seed: int = 42,
    target_spacing: Tuple[float, float, float] = None,
    ensure_foreground: bool = True,
    clip_lower: float = -200.0,
    clip_upper: float = 300.0,
) -> Tuple[NPCDataset, NPCDataset, List[List[str]]]:
    cases = _get_all_cases(data_root)
    patient_ids = [c[0] for c in cases]
    folds = _create_folds(patient_ids, num_folds=num_folds, seed=seed)

    if fold < 0 or fold >= num_folds:
        raise ValueError(f"fold must be in [0, {num_folds - 1}], got {fold}")

    val_ids = set(folds[fold])

    train_image_paths: List[Path] = []
    train_label_paths: List[Path] = []
    val_image_paths: List[Path] = []
    val_label_paths: List[Path] = []

    for pid, img_path, label_path in cases:
        if pid in val_ids:
            val_image_paths.append(img_path)
            val_label_paths.append(label_path)
        else:
            train_image_paths.append(img_path)
            train_label_paths.append(label_path)

    train_dataset = NPCDataset(
        train_image_paths, train_label_paths, patch_size,
        training=True, target_spacing=target_spacing,
        ensure_foreground=ensure_foreground,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )
    val_dataset = NPCDataset(
        val_image_paths, val_label_paths, patch_size,
        training=False, target_spacing=target_spacing,
        ensure_foreground=False,  # 验证时不需要确保前景
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )

    return train_dataset, val_dataset, folds
