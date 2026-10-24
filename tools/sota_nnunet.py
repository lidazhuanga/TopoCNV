#!/usr/bin/env python3
"""
Standard nnU-Net v2 baseline on the binary Y endpoint (protocol v0.3 §14.4), same data as P0.

Input is the P0 preprocess cache (`results/preprocess_cache/sp_2.500_1.133_1.133`): the same
resampled grid (2.5 x 1.133 x 1.133 mm, z/y/x) and the same normalised CT every other method
trains on, so nnU-Net's predictions land on the grid eval_protocol_metrics.py scores on.
Labels: Y = 1∪2∪3∪4 (class5_bypass -> background). Folds: extras/splits.json, written as
nnU-Net's splits_final.json. Everything else is nnU-Net default: 3d_fullres, default plans, final checkpoint, nnU-Net's own validation pass.
Training length is capped at 300 epochs (nnUNetTrainer_300epochs, tools/nnUNetTrainer_300epochs.py,
copied into the venv), the same cap as the registry SOTA networks. Intensities are not re-normalised
(channel 'nonorm'): the cache is already percentile-normalised, as for every other method.

Runs in the separate venv ~/cnv/envs/nnunetv2_cnv (vanilla nnunetv2 2.5.2 from PyPI), data
under ~/cnv/nnunet_cnv (outside the repo):

  PY=~/cnv/envs/nnunetv2_cnv/bin/python
  $PY tools/sota_nnunet.py prep            # cache -> nnUNet_raw, plan + preprocess, splits (CPU)
  $PY tools/sota_nnunet.py train --fold 0  # train (continues from checkpoint_latest if present),
                                           # then export OOF labels + COMPLETED

OOF labels: results/sota_binary_y_nnunetv2_label_predictions/<case>_stage2_label.nrrd (P0 format).
Done marker: results/sota_binary_y_nnunetv2/fold<k>/COMPLETED.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "preprocess_cache" / "sp_2.500_1.133_1.133"
SPACING_ZYX = (2.5, 1.1328120231628418, 1.1328120231628418)
NNUNET_HOME = Path(os.environ.get("CNV_NNUNET_HOME", Path.home() / "cnv" / "nnunet_cnv"))
DATASET_ID = 501
DATASET = f"Dataset{DATASET_ID}_CNVbinaryY"
CONFIG, TRAINER, PLANS = "3d_fullres", "nnUNetTrainer_300epochs", "nnUNetPlans"
OUT_EXP = ROOT / "results" / "sota_binary_y_nnunetv2"
OUT_LABELS = ROOT / "results" / "sota_binary_y_nnunetv2_label_predictions"


def nnunet_env():
    env = dict(os.environ,
               nnUNet_raw=str(NNUNET_HOME / "nnUNet_raw"),
               nnUNet_preprocessed=str(NNUNET_HOME / "nnUNet_preprocessed"),
               nnUNet_results=str(NNUNET_HOME / "nnUNet_results"),
               PYTHONUNBUFFERED="1")
    env.setdefault("nnUNet_compile", "f")       # torch.compile off: same maths, fewer failure modes
    env.setdefault("nnUNet_n_proc_DA", "8")     # shared host; P0 also needs CPU
    return env


def bin_cmd(name):
    return str(Path(sys.executable).parent / name)


def _write_nifti(arr_zyx, path, dtype):
    """Cache array (z, y, x) -> NIfTI whose NibabelIO read gives back the same (z, y, x) array."""
    import nibabel as nib
    affine = np.diag([SPACING_ZYX[2], SPACING_ZYX[1], SPACING_ZYX[0], 1.0])
    nib.save(nib.Nifti1Image(np.ascontiguousarray(arr_zyx.transpose(2, 1, 0)).astype(dtype), affine), str(path))


def prep(args):
    raw = NNUNET_HOME / "nnUNet_raw" / DATASET
    (raw / "imagesTr").mkdir(parents=True, exist_ok=True)
    (raw / "labelsTr").mkdir(parents=True, exist_ok=True)
    splits = json.loads((ROOT / "extras" / "splits.json").read_text())
    cases = sorted(c for fold in splits["folds"] for c in fold)
    assert len(cases) == len(set(cases)) == 154, len(cases)
    for i, case in enumerate(cases):
        img_p, lab_p = raw / "imagesTr" / f"{case}_0000.nii.gz", raw / "labelsTr" / f"{case}.nii.gz"
        if img_p.exists() and lab_p.exists():
            continue
        img = np.load(CACHE / f"{case}_image.npy")
        lab = np.load(CACHE / f"{case}_label.npy")
        y = ((lab >= 1) & (lab <= 4)).astype(np.uint8)
        _write_nifti(img, img_p, np.float32)
        _write_nifti(y, lab_p, np.uint8)
        print(f"[{i + 1}/{len(cases)}] {case} {img.shape} Y={int(y.sum())}", flush=True)
    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "nonorm"},
        "labels": {"background": 0, "Y": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIO",
        "description": "CN V binary Y = 1∪2∪3∪4 on the TopoCNV preprocess cache grid "
                       "(sp_2.500_1.133_1.133, percentile-normalised CT)",
    }, indent=2))
    env = nnunet_env()
    subprocess.run([bin_cmd("nnUNetv2_plan_and_preprocess"), "-d", str(DATASET_ID), "-c", CONFIG,
                    "-np", str(args.np), "--verify_dataset_integrity"], env=env, check=True)
    pre = NNUNET_HOME / "nnUNet_preprocessed" / DATASET
    (pre / "splits_final.json").write_text(json.dumps(
        [{"train": sorted(c for j, f in enumerate(splits["folds"]) if j != k for c in f),
          "val": sorted(splits["folds"][k])} for k in range(len(splits["folds"]))], indent=1))
    print(f"splits_final.json written from extras/splits.json -> {pre}")


def fold_dir(fold):
    return NNUNET_HOME / "nnUNet_results" / DATASET / f"{TRAINER}__{PLANS}__{CONFIG}" / f"fold_{fold}"


def export(fold):
    """nnU-Net validation NIfTIs -> cache-space (z, y, x) NRRD labels in the P0 format."""
    import nibabel as nib
    import nrrd
    splits = json.loads((ROOT / "extras" / "splits.json").read_text())
    OUT_LABELS.mkdir(parents=True, exist_ok=True)
    for case in sorted(splits["folds"][fold]):
        pred = nib.load(str(fold_dir(fold) / "validation" / f"{case}.nii.gz")).get_fdata()
        pred = np.ascontiguousarray(pred.transpose(2, 1, 0)).astype(np.int16)
        gt_shape = np.load(CACHE / f"{case}_label.npy", mmap_mode="r").shape
        assert pred.shape == gt_shape, (case, pred.shape, gt_shape)
        nrrd.write(str(OUT_LABELS / f"{case}_stage2_label.nrrd"), pred,
                   {"type": "short", "dimension": 3, "sizes": list(pred.shape),
                    "kinds": ["domain"] * 3, "encoding": "gzip", "endian": "little"})
    print(f"exported fold {fold} OOF labels -> {OUT_LABELS}")


def train(args):
    env = nnunet_env()
    cmd = [bin_cmd("nnUNetv2_train"), str(DATASET_ID), CONFIG, str(args.fold), "-tr", TRAINER, "-p", PLANS]
    fd = fold_dir(args.fold)
    if (fd / "checkpoint_final.pth").exists():
        if not (fd / "validation" / "summary.json").exists():
            cmd.append("--val")                 # training done, validation pass missing
        else:
            cmd = None
    elif (fd / "checkpoint_latest.pth").exists():
        cmd.append("--c")
    if cmd:
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)
    export(args.fold)
    out = OUT_EXP / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((fd / "validation" / "summary.json").read_text())
    (out / "nnunet_validation_summary.json").write_text(json.dumps(summary["foreground_mean"], indent=2))
    (out / "COMPLETED").write_text(f"nnU-Net v2 {TRAINER} {CONFIG} fold {args.fold} completed "
                                   f"{datetime.now().isoformat()}\nnnU-Net own val Dice: "
                                   f"{summary['foreground_mean'].get('Dice')}\n")
    print(f"COMPLETED {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--np", type=int, default=8)
    t = sub.add_parser("train")
    t.add_argument("--fold", type=int, required=True)
    e = sub.add_parser("export")
    e.add_argument("--fold", type=int, required=True)
    args = ap.parse_args()
    if args.cmd == "prep":
        prep(args)
    elif args.cmd == "train":
        train(args)
    else:
        export(args.fold)


if __name__ == "__main__":
    main()
