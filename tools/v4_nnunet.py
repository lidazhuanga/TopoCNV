#!/usr/bin/env python3
"""nnU-Net v2 side of the v4 main line (docs/V4_PLAN.md §1–§4). Runs under the nnU-Net venv:

  PY=~/cnv/envs/nnunetv2_cnv/bin/python
  $PY tools/v4_nnunet.py install                                    # trainer -> venv (symlink)
  $PY tools/v4_nnunet.py prep --fold 0 --variant dev --margin 25    # Dataset520 (CPU)
  $PY tools/v4_nnunet.py prep --fold 0 --variant report --margin 25 [--shuffle]
  $PY tools/v4_nnunet.py prep_r0dev                                 # Dataset519 = Dataset501 data, dev splits
  $PY tools/v4_nnunet.py train --fold 0 --variant dev --trainer nnUNetTrainerV4_Pabc --name r1p_abc
  $PY tools/v4_nnunet.py train --fold 0 --variant dev --r0 --name r0_dev

Datasets (Dataset501 and its results are never written):
  51k  report fold k: atlas from T_k, train = T_k, val = report fold k
  52k  dev fold k:    atlas from T_k - V_k, train = T_k - V_k, val = V_k
  53k  report fold k, shuffled priors (Full-v4 + shuffled prior arm)
  519  R0-dev: Dataset501's preprocessed data (symlinked, read only), splits = dev splits of folds 0 and 1
Channels: 0 CT ('nonorm', the cache), 1-3 priors a/b/c ('nonorm', [0, 1]); labels = original 0..5.
Each case is cropped to its ROI (union of the per-side atlas boxes at margin m); outside the side boxes
predictions are set to 0 on export. Plans: Dataset501's nnUNetPlans with the channel fields widened to 4.

Outputs: dev    -> results/v4_dev_preds/<name>/fold{k}/{case}_stage2_label.nrrd
         report -> results/v4_<arm>_label_predictions/{case}_stage2_label.nrrd
Arms with the visibility head also write {case}_vis.nrrd (0..4) next to it, where
eval_protocol_metrics.py picks it up for Dice_tier1..4.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainer.v4.common import (CACHE_DIR, PRIORS_DIR, RESULTS, SPACING, boxes_mask, box_slices,  # noqa: E402
                               case_shift, region_from_box, union_box)
from trainer.v4.splits import all_cases, internal_split, report_cases, train_cases  # noqa: E402

NNUNET_HOME = Path(os.environ.get("CNV_NNUNET_HOME", Path.home() / "cnv" / "nnunet_cnv"))
SRC_DATASET = "Dataset501_CNVbinaryY"
R0DEV = (519, "Dataset519_CNVv4R0dev")
CONFIG, PLANS = "3d_fullres", "nnUNetPlans"
N_CH = 4


def env():
    e = dict(os.environ, nnUNet_raw=str(NNUNET_HOME / "nnUNet_raw"),
             nnUNet_preprocessed=str(NNUNET_HOME / "nnUNet_preprocessed"),
             nnUNet_results=str(NNUNET_HOME / "nnUNet_results"), PYTHONUNBUFFERED="1")
    e.setdefault("nnUNet_compile", "f")   # as the baseline (tools/sota_nnunet.py)
    e.setdefault("nnUNet_n_proc_DA", "8")
    return e


def bin_cmd(name):
    return str(Path(sys.executable).parent / name)


def dataset_id(fold, variant, shuffle=False):
    if variant == "dev":
        assert not shuffle
        return 520 + fold
    return (530 if shuffle else 510) + fold


def dataset_name(fold, variant, shuffle=False):
    return f"Dataset{dataset_id(fold, variant, shuffle)}_CNVv4{variant}F{fold}{'shuf' if shuffle else ''}"


def write_nifti(arr_zyx, path, dtype):
    """Same convention as tools/sota_nnunet.py: NibabelIO reads back the cache (z, y, x) array."""
    import nibabel as nib
    affine = np.diag([SPACING[2], SPACING[1], SPACING[0], 1.0])
    nib.save(nib.Nifti1Image(np.ascontiguousarray(arr_zyx.transpose(2, 1, 0)).astype(dtype), affine), str(path))


def splits_for(fold, variant, cases=None):
    if variant == "dev":
        entries = [{"train": list(internal_split(fold)[0]), "val": list(internal_split(fold)[1])}]
    else:
        entries = [{"train": list(train_cases(fold)), "val": list(report_cases(fold))}]
    if cases:
        keep = set(cases)
        entries = [{"train": [c for c in e["train"] if c in keep],
                    "val": [c for c in e["val"] if c in keep]} for e in entries]
        assert all(e["train"] and e["val"] for e in entries), "case subset emptied a split"
    return entries


def widen_plans(src, dst_name):
    p = json.loads(src.read_text())
    p["dataset_name"] = dst_name
    for c in p["configurations"].values():
        if "normalization_schemes" in c:
            c["normalization_schemes"] = ["NoNormalization"] * N_CH
        if "use_mask_for_norm" in c:
            c["use_mask_for_norm"] = [False] * N_CH
    fip = p.get("foreground_intensity_properties_per_channel", {})
    if "0" in fip:
        p["foreground_intensity_properties_per_channel"] = {str(i): fip["0"] for i in range(N_CH)}
    return p


# ---------------- commands ----------------

def install(_):
    import nnunetv2
    dst = Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer" / "variants" / "zz_cnv_v4"
    dst.mkdir(exist_ok=True)
    (dst / "__init__.py").touch()
    link = dst / "nnUNetTrainerV4.py"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(ROOT / "trainer" / "v4" / "nnunet" / "nnUNetTrainerV4.py")
    print(f"installed {link} -> {link.resolve()}")
    import shutil
    dst300 = dst / "nnUNetTrainer_300epochs.py"
    if dst300.exists() or (dst300.is_symlink() and dst300.is_file()):
        dst300.unlink()
    shutil.copy(ROOT / "tools" / "nnUNetTrainer_300epochs.py", dst300)
    print(f"installed {dst300} (copy)")


def prep(args):
    k, variant = args.fold, args.variant
    name = dataset_name(k, variant, args.shuffle)
    raw = NNUNET_HOME / "nnUNet_raw" / name
    (raw / "imagesTr").mkdir(parents=True, exist_ok=True)
    (raw / "labelsTr").mkdir(parents=True, exist_ok=True)
    pdir = PRIORS_DIR / f"fold{k}_{variant}"
    meta = {"fold": k, "variant": variant, "margin_mm": args.margin, "shuffle": args.shuffle,
            "priors_dir": str(pdir), "created": datetime.now().isoformat(), "cases": {}}
    cases = train_cases(k) if variant == "dev" else all_cases()   # dev: T_k only (V4_PLAN §6)
    if args.cases:
        want = set(args.cases.split(","))
        unknown = want - set(cases)
        assert not unknown, f"cases not in this variant's set: {sorted(unknown)}"
        cases = [c for c in cases if c in want]
        extra = [p.name for p in (raw / "imagesTr").glob("*.nii.gz")
                 if p.name.split("_")[0] not in want]
        assert not extra, f"--cases requires an empty/singleton raw dir; found foreign files: {extra[:5]}"
    for i, c in enumerate(cases):
        rec = json.loads((pdir / f"{c}_roi.json").read_text())
        shape = tuple(rec["shape"])
        boxes = [sd["boxes"][str(args.margin)] for sd in rec["sides"] if sd["boxes"][str(args.margin)] is not None]
        crop = union_box(boxes) if boxes else [[0, n] for n in shape]
        meta["cases"][c] = {"shape": list(shape), "crop": crop, "boxes": boxes}
        if (raw / "labelsTr" / f"{c}.nii.gz").exists() and not args.overwrite:
            continue
        img = np.load(CACHE_DIR / f"{c}_image.npy", mmap_mode="r")
        lab = np.load(CACHE_DIR / f"{c}_label.npy", mmap_mode="r")
        sl = box_slices(crop)
        pri = np.load(pdir / f"{c}_prior.npy", mmap_mode="r")
        shift = case_shift(c, shape) if args.shuffle else None
        pr = region_from_box(np.asarray(pri), rec["prior_box"], crop, full_shape=shape, shift=shift)
        write_nifti(np.asarray(img[sl], np.float32), raw / "imagesTr" / f"{c}_0000.nii.gz", np.float32)
        for ch in range(3):
            write_nifti(pr[ch].astype(np.float32) / 255.0, raw / "imagesTr" / f"{c}_{ch + 1:04d}.nii.gz", np.float32)
        write_nifti(np.asarray(lab[sl], np.uint8), raw / "labelsTr" / f"{c}.nii.gz", np.uint8)
        print(f"[{i + 1}/{len(cases)}] {c} crop {[b - a for a, b in crop]}", flush=True)
    (raw / "v4_meta.json").write_text(json.dumps(meta))
    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "nonorm", "1": "nonorm", "2": "nonorm", "3": "nonorm"},
        "labels": {"background": 0, "c1": 1, "c2": 2, "c3": 3, "c4": 4, "c5": 5},
        "numTraining": len(cases), "file_ending": ".nii.gz", "overwrite_image_reader_writer": "NibabelIO",
        "description": f"CN V v4 fold {k} ({variant}{', shuffled priors' if args.shuffle else ''}): ROI-cropped "
                       f"cache CT + priors a/b/c, original labels 0..5, margin {args.margin} mm, "
                       f"{len(cases)} cases"}, indent=2))
    pre = NNUNET_HOME / "nnUNet_preprocessed" / name
    pre.mkdir(parents=True, exist_ok=True)
    src = NNUNET_HOME / "nnUNet_preprocessed" / SRC_DATASET / f"{PLANS}.json"
    (pre / f"{PLANS}.json").write_text(json.dumps(widen_plans(src, name), indent=2))
    shutil.copy(raw / "dataset.json", pre / "dataset.json")
    # validate splits before the expensive preprocessing (P2-6)
    (pre / "splits_final.json").write_text(json.dumps(
        splits_for(k, variant, args.cases.split(",") if args.cases else None), indent=1))
    did = dataset_id(k, variant, args.shuffle)
    subprocess.run([bin_cmd("nnUNetv2_preprocess"), "-d", str(did),
                    "-plans_name", PLANS, "-c", CONFIG, "-np", str(args.np)], env=env(), check=True)
    # fingerprint: nnUNetTrainer.on_train_start copies it unconditionally; prep must provide it
    subprocess.run([bin_cmd("nnUNetv2_extract_fingerprint"), "-d", str(did), "-np", str(args.np)],
                   env=env(), check=True)
    if not args.cases:                      # smoke subsets must not mark the production queue done
        done = ROOT / "log" / "v4" / f"nnprep_{did}.done"
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(f"{datetime.now().isoformat()} {name} margin {args.margin}\n")
    print(f"prepared {name}")


def prep_r0dev(_):
    """Dataset519: Dataset501's preprocessed files symlinked (read only) with the dev splits of folds 0, 1."""
    src = NNUNET_HOME / "nnUNet_preprocessed" / SRC_DATASET
    pre = NNUNET_HOME / "nnUNet_preprocessed" / R0DEV[1]
    pre.mkdir(parents=True, exist_ok=True)
    for item in ("nnUNetPlans_3d_fullres", "gt_segmentations", "dataset.json", "dataset_fingerprint.json"):
        if (src / item).exists() and not (pre / item).exists():
            (pre / item).symlink_to(src / item)
    p = json.loads((src / f"{PLANS}.json").read_text())
    p["dataset_name"] = R0DEV[1]
    (pre / f"{PLANS}.json").write_text(json.dumps(p, indent=2))
    splits = []
    for k in range(5):                       # nested CV needs R0-dev on every fold (V4_PLAN §6)
        tr, iv = internal_split(k)
        splits.append({"train": tr, "val": iv})
    (pre / "splits_final.json").write_text(json.dumps(splits, indent=1))
    print(f"prepared {pre}")


def results_dir(did, dname, trainer, split):
    return NNUNET_HOME / "nnUNet_results" / dname / f"{trainer}__{PLANS}__{CONFIG}" / f"fold_{split}"


def gate_report(args):
    """Report training must follow the per-fold frozen selection (V4_PLAN §7): all 5 frozen files
    exist, and the requested trainer/margin match fold k's chosen_config."""
    fz = ROOT / "configs" / "v4_frozen"
    missing = [k for k in range(5) if not (fz / f"fold{k}.json").exists()]
    if missing:
        raise SystemExit(f"report phase needs all 5 frozen configs (configs/v4_frozen/fold*.json); "
                         f"missing folds {missing}")
    cfg = json.loads((fz / f"fold{args.fold}.json").read_text()).get("chosen_config", {})
    if "margin" in cfg and int(cfg["margin"]) != int(args.margin):
        raise SystemExit(f"fold {args.fold}: --margin {args.margin} != frozen margin {cfg['margin']}")
    trainers = cfg.get("trainers") or ([cfg["trainer"]] if "trainer" in cfg else [])
    if trainers and args.trainer not in trainers:
        raise SystemExit(f"fold {args.fold}: trainer {args.trainer!r} not among frozen {trainers}")


def train(args):
    if args.variant == "report":
        gate_report(args)
    if args.r0 and not (0 <= args.fold < 5):
        raise SystemExit("--r0 (Dataset519) has dev splits for all 5 folds (V4_PLAN §6)")
    if args.r0:
        did, dname, trainer, split = R0DEV[0], R0DEV[1], "nnUNetTrainer_300epochs", args.fold
        assert args.variant == "dev"
    else:
        did, dname = dataset_id(args.fold, args.variant, args.shuffle), dataset_name(args.fold, args.variant, args.shuffle)
        trainer, split = args.trainer, 0
    fd = results_dir(did, dname, trainer, split)
    cmd = [bin_cmd("nnUNetv2_train"), str(did), CONFIG, str(split), "-tr", trainer, "-p", PLANS]
    if (fd / "checkpoint_final.pth").exists():
        done = (fd / "validation" / "summary.json").exists() if args.r0 else (fd / "validation_v4" / "DONE").exists()
        cmd = None if done else cmd + ["--val"]
    elif (fd / "checkpoint_latest.pth").exists():
        cmd.append("--c")
    if cmd:
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env(), check=True)
    export(args, fd)


def export(args, fd):
    import nrrd
    hdr = lambda a: {"type": "short", "dimension": 3, "sizes": list(a.shape), "kinds": ["domain"] * 3,  # noqa: E731
                     "encoding": "gzip", "endian": "little"}
    k = args.fold
    cases = internal_split(k)[1] if args.variant == "dev" else report_cases(k)
    if args.cases:
        keep = set(args.cases.split(","))
        cases = [c for c in cases if c in keep]
        assert cases, "no requested case is a validation case of this fold/variant"
    if args.variant == "dev":
        out = RESULTS / "v4_dev_preds" / args.name / f"fold{k}"
    else:
        out = RESULTS / f"v4_{args.name}_label_predictions"
    out.mkdir(parents=True, exist_ok=True)
    if args.r0:
        import nibabel as nib
        for c in cases:
            pred = nib.load(str(fd / "validation" / f"{c}.nii.gz")).get_fdata()
            pred = np.ascontiguousarray(pred.transpose(2, 1, 0)).astype(np.int16)
            nrrd.write(str(out / f"{c}_stage2_label.nrrd"), pred, hdr(pred))
    else:
        dname = dataset_name(k, args.variant, args.shuffle)
        meta = json.loads((NNUNET_HOME / "nnUNet_raw" / dname / "v4_meta.json").read_text())["cases"]
        for c in cases:
            m = meta[c]
            shape, crop = tuple(m["shape"]), m["crop"]
            d = np.load(fd / "validation_v4" / f"{c}.npz")
            roi = boxes_mask(m["boxes"], crop) if m["boxes"] else np.ones(d["Y"].shape, bool)
            assert d["Y"].shape == roi.shape, (c, d["Y"].shape, roi.shape)
            full = np.zeros(shape, np.int16)
            full[box_slices(crop)] = (d["Y"] > 0) & roi
            nrrd.write(str(out / f"{c}_stage2_label.nrrd"), full, hdr(full))
            if "vis" in d.files:
                v = np.zeros(shape, np.int16)
                v[box_slices(crop)] = d["vis"] * roi
                nrrd.write(str(out / f"{c}_vis.nrrd"), v, hdr(v))
    marker = out / ("EXPORTED" if args.variant == "dev" else f"EXPORTED_fold{k}")
    marker.write_text(f"{datetime.now().isoformat()} {len(cases)} cases from {fd}\n")
    print(f"exported {len(cases)} cases -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("install")
    p = sub.add_parser("prep")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--variant", choices=["dev", "report"], required=True)
    p.add_argument("--margin", type=int, required=True)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--np", type=int, default=8)
    p.add_argument("--cases", default="", help="comma list (smoke tests): tiny dataset of these cases only")
    sub.add_parser("prep_r0dev")
    t = sub.add_parser("train")
    t.add_argument("--fold", type=int, required=True)
    t.add_argument("--variant", choices=["dev", "report"], required=True)
    t.add_argument("--trainer", default=None)
    t.add_argument("--r0", action="store_true")
    t.add_argument("--shuffle", action="store_true")
    t.add_argument("--name", required=True, help="output name (dev candidate or report arm)")
    t.add_argument("--cases", default="", help="comma list (smoke tests): export only these cases")
    a = ap.parse_args()
    {"install": install, "prep": prep, "prep_r0dev": prep_r0dev, "train": train}[a.cmd](a)


if __name__ == "__main__":
    main()
