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

V4.2 (V4_PLAN §11): full image, labels from the frozen harmonised cache results/v4_harm_v1. Commands:
  $PY tools/v4_nnunet.py prep42 --what binY     # Dataset540: CT + binary Y (R0u, B, D4')
  $PY tools/v4_nnunet.py prep42 --what s1       # Dataset541: CT + classes 0..4 (Stage 1); links 540's CT
  $PY tools/v4_nnunet.py train42 --dataset binY --fold 0 --variant dev --name v42_r0u     # stock R0u
  $PY tools/v4_nnunet.py train42 --dataset binY --fold 0 --variant dev --trainer nnUNetTrainerV4_noP_S05 --name v42_s05
  $PY tools/v4_nnunet.py train42 --dataset s1 --fold 0 --variant dev --inner 0            # Stage 1 inner model
  bash tests/v4_nnunet_smoke42.sh <gpu>         # GPU smoke on tiny copies in a scratch nnU-Net home
Checks: prep42 verifies every harmonised label against the manifest and records the label digest in
v42_meta.json and the done marker; train42 refuses fold/inner indices outside the split table and prior arms
on Dataset540, logs each launch to <fold dir>/v42_runs.jsonl, and marks a Stage 1 run done only when every
held-out inner case has its probabilities (validation/*.npz, nnU-Net axis order = cache order).
Splits:
  Dataset540  entries 0-4 = report fold k (train T_k, val fold k); entries 5-9 = dev fold k (train T_k - V_k, val V_k).
  Dataset541  entry 3k + j = dev-tier inner model j of fold k; entry 15 + 3k + j = report tier (trainer/v4/splits.py).
IDs 542-556 are reserved for the 4-channel prior datasets, which are built after Stage 1.
Plans: Dataset501's, with only dataset_name changed. Dev outputs use names with the v42_ prefix.
"""
import argparse
import hashlib
import json
import os
import re
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
from trainer.v4.splits import (N_INNER, all_cases, inner_folds, internal_split, report_cases,  # noqa: E402
                                stage1_train_cases, train_cases)

NNUNET_HOME = Path(os.environ.get("CNV_NNUNET_HOME", Path.home() / "cnv" / "nnunet_cnv"))
SRC_DATASET = "Dataset501_CNVbinaryY"
R0DEV = (519, "Dataset519_CNVv4R0dev")
CONFIG, PLANS = "3d_fullres", "nnUNetPlans"
N_CH = 4
HARM_DIR = RESULTS / "v4_harm_v1"
HARM_CFG = ROOT / "configs" / "v4_harm" / "harm_v1.json"
V42 = {"binY": (540, "Dataset540_CNVv42binY"), "s1": (541, "Dataset541_CNVv42s1")}
R0_TRAINER = "nnUNetTrainer_300epochs"
SMOKE_STOCK = "nnUNetTrainer_SMOKE_2x5"   # stock trainer, 2 epochs x 5 iterations (tests/v4_nnunet_smoke42.sh)


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
    qc_path = ROOT / "configs" / "v4_qc" / f"fold{k}.json"
    if qc_path.exists():
        qc = json.loads(qc_path.read_text())
        assert int(args.margin) == int(qc["margin"]), \
            f"fold {k}: --margin {args.margin} != frozen D1-AM1 margin {qc['margin']}"
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
        if qc_path.exists():
            q = rec.get("qc", {})
            assert q.get("amendment") == "D1-AM1" and "fallback" in q, \
                f"{c}: roi record has no D1-AM1 qc block — priors were built without --qc"
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
    # pre-unpack once here (serialized by the queue): without it, two trainers starting on the
    # same dataset race inside on_train_start's unpack and truncate each other's .npy files
    pre_folder = pre / f"{PLANS}_{CONFIG}"
    subprocess.run([sys.executable, "-c",
                    "from nnunetv2.training.dataloading.utils import unpack_dataset; "
                    f"unpack_dataset('{pre_folder}', unpack_segmentation=True, "
                    f"overwrite_existing=False, num_processes={args.np}, verify_npy=True)"],
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
    qc = ROOT / "configs" / "v4_qc" / f"fold{args.fold}.json"
    if not qc.exists():
        raise SystemExit(f"missing {qc}: the D1-AM1 gate must be frozen before report training")
    qcfg = json.loads(qc.read_text())
    if qcfg.get("fold") != args.fold or qcfg.get("amendment") != "D1-AM1":
        raise SystemExit(f"{qc}: bad fold/amendment ({qcfg.get('fold')}/{qcfg.get('amendment')})")
    cfg = json.loads((fz / f"fold{args.fold}.json").read_text()).get("chosen_config", {})
    if "margin" in cfg and int(cfg["margin"]) != int(qcfg["margin"]):
        raise SystemExit(f"fold {args.fold}: frozen margin {cfg['margin']} != D1-AM1 margin "
                         f"{qcfg['margin']}")
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
    if not args.r0:      # Dataset519 is nnU-Net stock and already unpacked; v4 datasets need the guard
        import fcntl
        pre_folder = NNUNET_HOME / "nnUNet_preprocessed" / dname / f"{PLANS}_{CONFIG}"
        lock = pre_folder / ".unpack_lock"
        with open(lock, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)      # serialize the first unpack across arms
            subprocess.run([sys.executable, "-c",
                            "from nnunetv2.training.dataloading.utils import unpack_dataset; "
                            f"unpack_dataset('{pre_folder}', unpack_segmentation=True, "
                            "overwrite_existing=False, num_processes=8, verify_npy=True)"],
                           env=env(), check=True)
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


# ---------------- V4.2 (V4_PLAN §11): full image, harmonised labels ----------------

def nrrd_header(a):
    return {"type": "short", "dimension": 3, "sizes": list(a.shape), "kinds": ["domain"] * 3,
            "encoding": "gzip", "endian": "little"}


def v42_label(lab, what):
    """binY: classes 1-4 -> 1. s1: class 5 -> 0, classes 1-4 kept (V4_PLAN §10.2)."""
    lab = np.asarray(lab)
    assert int(lab.max()) <= 5, "labels must be 0..5"
    if what == "binY":
        return ((lab >= 1) & (lab <= 4)).astype(np.uint8)
    return np.where(lab == 5, 0, lab).astype(np.uint8)


def v42_splits(what):
    if what == "binY":
        return ([{"train": train_cases(k), "val": report_cases(k)} for k in range(5)]
                + [{"train": internal_split(k)[0], "val": internal_split(k)[1]} for k in range(5)])
    return [{"train": stage1_train_cases(k, j, tier), "val": inner_folds(k, tier=tier)[j]}
            for tier in ("dev", "report") for k in range(5) for j in range(N_INNER)]


def v42_split_index(what, fold, variant, inner=None):
    # an index past splits_final.json makes nnU-Net fall back to a random 80:20 split of all cases
    assert 0 <= fold < 5 and variant in ("dev", "report"), f"fold 0..4 and dev/report only ({fold}, {variant})"
    if what == "binY":
        assert inner is None, "binY has no inner folds"
        return fold if variant == "report" else 5 + fold
    assert inner is not None and 0 <= inner < N_INNER, "Stage 1 needs --inner 0..2"
    return (0 if variant == "dev" else 5 * N_INNER) + N_INNER * fold + inner


def _label_sha(a):
    c, cache = a
    lab = np.load(Path(cache) / f"{c}_label.npy")
    return c, hashlib.sha256(np.ascontiguousarray(lab).tobytes()).hexdigest()   # = arr_sha of the build


def harm_cache(verify=True):
    """The frozen harmonised label cache (tools/v4_harmonize_labels.py); refuses an unbuilt or mismatched one.

    Checks harm_v1.json against the manifest's config sha256, trainer/v4/harmonize.py against the sha256
    recorded at calibration, the manifest's case list, and (verify) every label array against its
    manifest sha256. Returns (cache dir, config sha256, label digest); the digest = sha256 of the sorted
    'case:sha256_label' lines, recorded in v42_meta.json, the done markers and every run record."""
    cache = HARM_DIR / CACHE_DIR.name
    if not (cache / "DONE").exists():
        raise SystemExit(f"{cache} not built: run tools/v4_harmonize_labels.py calibrate / build / qa first")
    cfg = HARM_CFG.read_bytes()
    sha = hashlib.sha256(cfg).hexdigest()
    man = json.loads((cache / "manifest.json").read_text())
    assert man["config_sha256"] == sha, "harmonised cache was built from a different harm_v1.json"
    rule = ROOT / "trainer" / "v4" / "harmonize.py"
    assert (hashlib.sha256(rule.read_bytes()).hexdigest()
            == json.loads(cfg)["code_sha256"]["trainer/v4/harmonize.py"]), f"{rule} changed since calibration"
    assert sorted(man["cases"]) == sorted(all_cases()), "manifest cases != the study cases"
    if verify:
        from multiprocessing import Pool
        with Pool(8) as p:
            got = dict(p.imap_unordered(_label_sha, [(c, str(cache)) for c in man["cases"]]))
        bad = sorted(c for c in man["cases"] if got[c] != man["cases"][c]["sha256_label"])
        assert not bad, f"harmonised labels differ from the manifest: {bad[:5]}"
    digest = hashlib.sha256("".join(f"{c}:{man['cases'][c]['sha256_label']}\n"
                                    for c in sorted(man["cases"])).encode()).hexdigest()
    return cache, sha, digest


def _atomic_nifti(arr, path, dtype):
    """write_nifti through raw/.tmp42 + rename, so an interrupted write never leaves a partial file in place."""
    tmp = path.parent.parent / ".tmp42" / path.name
    tmp.parent.mkdir(exist_ok=True)
    write_nifti(arr, tmp, dtype)
    os.replace(tmp, path)


def _write42(a):
    c, what, cache, raw, overwrite = a
    import nibabel as nib
    raw = Path(raw)
    lab = v42_label(np.load(Path(cache) / f"{c}_label.npy"), what)
    img = raw / "imagesTr" / f"{c}_0000.nii.gz"
    if what == "s1":   # same CT as Dataset540: link its raw image instead of writing a second copy
        src = NNUNET_HOME / "nnUNet_raw" / V42["binY"][1] / "imagesTr" / f"{c}_0000.nii.gz"
        assert src.exists(), f"{src} missing: run prep42 --what binY first"
        if img.is_symlink() or img.exists():
            img.unlink()
        img.symlink_to(src)
    elif overwrite or not img.exists():   # the harm cache's image is a symlink to the original cache CT
        _atomic_nifti(np.asarray(np.load(Path(cache) / f"{c}_image.npy", mmap_mode="r"), np.float32), img, np.float32)
    out = raw / "labelsTr" / f"{c}.nii.gz"
    _atomic_nifti(lab, out, np.uint8)     # always rewritten: labels follow the verified cache, never a stale file
    back = np.asarray(nib.load(str(out)).dataobj).transpose(2, 1, 0)
    assert np.array_equal(back, lab), f"{c}: label read-back mismatch"
    return c, [int((lab == v).sum()) for v in range(5)]


def _versions():
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for pkg in ("nnunetv2", "torch", "numpy", "scipy", "scikit-image", "nibabel"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def prep42(args):
    from multiprocessing import Pool
    what = args.what
    did, name = V42[what]
    raw = NNUNET_HOME / "nnUNet_raw" / name
    (raw / "imagesTr").mkdir(parents=True, exist_ok=True)
    (raw / "labelsTr").mkdir(parents=True, exist_ok=True)
    cases = list(all_cases())
    if args.cases:
        want = set(args.cases.split(","))
        assert not want - set(cases), f"unknown cases: {sorted(want - set(cases))}"
        cases = [c for c in cases if c in want]
        # a smoke subset would rewrite the full dataset's dataset.json / splits_final.json in place
        extra = sorted({p.name.split("_")[0] for p in (raw / "imagesTr").glob("*.nii.gz")} - want)
        assert not extra, f"--cases requires an empty/subset raw dir; {name} holds other cases: {extra[:5]}"
    elif what == "s1":
        assert (ROOT / "log" / "v4" / "nnprep_540.done").exists(), "Dataset540 is not complete: prep42 --what binY first"
    cache, harm_sha, digest = harm_cache()
    with Pool(args.np) as p:
        jobs = [(c, what, str(cache), str(raw), args.overwrite) for c in cases]
        for i, (c, counts) in enumerate(p.imap_unordered(_write42, jobs)):
            print(f"[{i + 1}/{len(cases)}] {c} voxels per label value {counts}", flush=True)
    labels = ({"background": 0, "Y": 1} if what == "binY"
              else {"background": 0, "c1": 1, "c2": 2, "c3": 3, "c4": 4})
    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "nonorm"}, "labels": labels, "numTraining": len(cases), "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIO",
        "description": f"CN V V4.2 ({what}): full-volume cache CT, labels from {HARM_DIR.name} "
                       f"({'binary Y' if what == 'binY' else 'classes 0..4, class 5 -> 0'}), {len(cases)} cases"},
        indent=2))
    (raw / "v42_meta.json").write_text(json.dumps({
        "what": what, "harm_cache": str(cache), "harm_config_sha256": harm_sha, "harm_label_digest": digest,
        "harm_built": (cache / "DONE").read_text().strip(),
        "tool_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(), "versions": _versions(),
        "created": datetime.now().isoformat(), "cases": cases}, indent=1))
    pre = NNUNET_HOME / "nnUNet_preprocessed" / name
    pre.mkdir(parents=True, exist_ok=True)
    plans = json.loads((NNUNET_HOME / "nnUNet_preprocessed" / SRC_DATASET / f"{PLANS}.json").read_text())
    plans["dataset_name"] = name      # everything else as Dataset501 (patch, network, batch, spacing, norm)
    (pre / f"{PLANS}.json").write_text(json.dumps(plans, indent=2))
    shutil.copy(raw / "dataset.json", pre / "dataset.json")
    splits = v42_splits(what)
    if args.cases:                    # smoke subsets: keep the indexing, drop the other cases
        keep = set(cases)
        splits = [{"train": [c for c in e["train"] if c in keep], "val": [c for c in e["val"] if c in keep]}
                  for e in splits]
    (pre / "splits_final.json").write_text(json.dumps(splits, indent=1))
    subprocess.run([bin_cmd("nnUNetv2_preprocess"), "-d", str(did), "-plans_name", PLANS, "-c", CONFIG,
                    "-np", str(args.np)], env=env(), check=True)
    subprocess.run([bin_cmd("nnUNetv2_extract_fingerprint"), "-d", str(did), "-np", str(args.np)],
                   env=env(), check=True)
    pre_folder = pre / f"{PLANS}_{CONFIG}"
    subprocess.run([sys.executable, "-c",
                    "from nnunetv2.training.dataloading.utils import unpack_dataset; "
                    f"unpack_dataset('{pre_folder}', unpack_segmentation=True, "
                    f"overwrite_existing=False, num_processes={args.np}, verify_npy=True)"],
                   env=env(), check=True)
    if not args.cases:
        done = ROOT / "log" / "v4" / f"nnprep_{did}.done"
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(f"{datetime.now().isoformat()} {name} {HARM_DIR.name} config {harm_sha[:12]} "
                        f"labels {digest[:12]}\n")
    print(f"prepared {name}")


def _log_run(fd, name, trainer, cmd):
    """Append what this launch ran with (code, trainer files as installed, dataset labels) to fd/v42_runs.jsonl."""
    import nnunetv2
    zz = Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer" / "variants" / "zz_cnv_v4"
    meta = json.loads((NNUNET_HOME / "nnUNet_raw" / name / "v42_meta.json").read_text())
    rec = {"time": datetime.now().isoformat(), "cmd": cmd, "trainer": trainer,
           "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                      text=True).stdout.strip(),
           "trainer_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(zz.glob("*.py"))},
           "dataset": {k: meta.get(k) for k in ("harm_config_sha256", "harm_label_digest", "created")},
           "versions": _versions()}
    fd.mkdir(parents=True, exist_ok=True)
    with open(fd / "v42_runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def train42(args):
    what = args.dataset
    did, name = V42[what]
    if what == "binY":
        trainer = args.trainer or R0_TRAINER
        if args.variant == "report" and trainer != R0_TRAINER:
            raise SystemExit("V4.2 report runs other than R0u wait for the per-fold V4.2 freeze (V4_PLAN §11.6)")
        if re.match(r"nnUNetTrainerV4_(SMOKE_)?P[abc]", trainer):
            raise SystemExit(f"{trainer} uses priors; Dataset540 is CT only (prior datasets 542-556 follow Stage 1)")
        if not args.name:
            raise SystemExit("--name is required for binY runs")
        if args.inner is not None:
            raise SystemExit("--inner is for Stage 1 (--dataset s1) only")
        split = v42_split_index(what, args.fold, args.variant)
    else:
        if args.trainer not in (None, R0_TRAINER, SMOKE_STOCK):
            raise SystemExit("Stage 1 is the stock nnU-Net trainer (V4_PLAN §10.3)")
        trainer = args.trainer or R0_TRAINER
        split = v42_split_index(what, args.fold, args.variant, args.inner)
    stock = not trainer.startswith("nnUNetTrainerV4")     # nnU-Net's own validation/ output (R0u, Stage 1, smoke)
    splits = json.loads((NNUNET_HOME / "nnUNet_preprocessed" / name / "splits_final.json").read_text())
    assert split < len(splits) and splits[split]["train"] and splits[split]["val"], \
        f"{name}: splits_final.json has no usable entry {split}"
    val = splits[split]["val"]
    fd = results_dir(did, name, trainer, split)

    def missing_npz():
        return [c for c in val if not (fd / "validation" / f"{c}.npz").exists()]

    import fcntl
    pre_folder = NNUNET_HOME / "nnUNet_preprocessed" / name / f"{PLANS}_{CONFIG}"
    with open(pre_folder / ".unpack_lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)      # serialize the first unpack across runs on the same dataset
        subprocess.run([sys.executable, "-c",
                        "from nnunetv2.training.dataloading.utils import unpack_dataset; "
                        f"unpack_dataset('{pre_folder}', unpack_segmentation=True, "
                        "overwrite_existing=False, num_processes=8, verify_npy=True)"],
                       env=env(), check=True)
    cmd = [bin_cmd("nnUNetv2_train"), str(did), CONFIG, str(split), "-tr", trainer, "-p", PLANS]
    if what == "s1":
        cmd.append("--npz")             # inner-OOF probabilities of the held-out inner fold (priors a, b, ICP)
    if (fd / "checkpoint_final.pth").exists():
        done = (fd / "validation" / "summary.json").exists() if stock else (fd / "validation_v4" / "DONE").exists()
        if what == "s1" and missing_npz():
            done = False                    # validated without --npz (or files lost): validate again with --npz
        cmd = None if done else cmd + ["--val"]
    elif (fd / "checkpoint_latest.pth").exists():
        cmd.append("--c")
    if cmd:
        _log_run(fd, name, trainer, cmd)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env(), check=True)
    if what == "binY":
        export42(args, fd, stock)
        return
    miss = missing_npz()
    assert not miss, f"{fd}: no Stage 1 probabilities (validation/*.npz) for {miss[:5]}"
    if trainer != R0_TRAINER:               # smoke runs never mark the production queue done
        print(f"Stage 1 smoke ({trainer}) done, no queue marker: {fd}")
        return
    mark = ROOT / "log" / "v4" / f"s142_{args.variant}_f{args.fold}_i{args.inner}.done"
    mark.parent.mkdir(parents=True, exist_ok=True)
    mark.write_text(f"{datetime.now().isoformat()} {fd} npz {len(val)}\n")
    print(f"Stage 1 {args.variant} fold {args.fold} inner {args.inner} done: {fd}")


def export42(args, fd, stock):
    import nrrd
    k = args.fold
    cases = internal_split(k)[1] if args.variant == "dev" else report_cases(k)
    if args.cases:
        keep = set(args.cases.split(","))
        cases = [c for c in cases if c in keep]
        assert cases, "no requested case is a validation case of this fold/variant"
    out = (RESULTS / "v4_dev_preds" / args.name / f"fold{k}" if args.variant == "dev"
           else RESULTS / f"v4_{args.name}_label_predictions")
    out.mkdir(parents=True, exist_ok=True)
    for c in cases:
        if stock:
            import nibabel as nib
            pred = np.asarray(nib.load(str(fd / "validation" / f"{c}.nii.gz")).dataobj).transpose(2, 1, 0)
        else:
            pred = np.load(fd / "validation_v4" / f"{c}.npz")["Y"]
        pred = np.ascontiguousarray(pred).astype(np.int16)
        shape = np.load(CACHE_DIR / f"{c}_label.npy", mmap_mode="r").shape
        assert pred.shape == shape, (c, pred.shape, shape)
        nrrd.write(str(out / f"{c}_stage2_label.nrrd"), pred, nrrd_header(pred))
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
    p42 = sub.add_parser("prep42")
    p42.add_argument("--what", choices=sorted(V42), required=True)
    p42.add_argument("--overwrite", action="store_true")
    p42.add_argument("--np", type=int, default=8)
    p42.add_argument("--cases", default="", help="comma list (smoke tests): tiny dataset of these cases only")
    t42 = sub.add_parser("train42")
    t42.add_argument("--dataset", choices=sorted(V42), required=True)
    t42.add_argument("--fold", type=int, required=True)
    t42.add_argument("--variant", choices=["dev", "report"], required=True)
    t42.add_argument("--inner", type=int, default=None, help="Stage 1 (s1): inner model 0..2")
    t42.add_argument("--trainer", default=None, help="binY: default = stock R0u trainer")
    t42.add_argument("--name", default=None, help="binY: output name (v42_* for dev candidates)")
    t42.add_argument("--cases", default="", help="comma list (smoke tests): export only these cases")
    a = ap.parse_args()
    {"install": install, "prep": prep, "prep_r0dev": prep_r0dev, "train": train,
     "prep42": prep42, "train42": train42}[a.cmd](a)


if __name__ == "__main__":
    main()
