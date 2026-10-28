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
Checks: prep42 verifies every harmonised label against the manifest and records the label and CT digests in
v42_meta.json and the done marker; train42 starts only with the user's direction record
(configs/v4_harm/harm_v1_direction.json) and a dataset built from that cache, refuses fold/inner indices
outside the split table and prior arms on Dataset540, takes stock vs V4 output paths from the class nnU-Net
resolves, logs each launch to <fold dir>/v42_runs.jsonl, and marks a Stage 1 run done only when every
held-out inner case has its probabilities (validation/*.npz, (5, z, LR, AP) = cache axis order).
Exports of case subsets or smoke trainers write EXPORTED_PARTIAL, never the marker the queue waits for.
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
HARM_DIRECTION = ROOT / "configs" / "v4_harm" / "harm_v1_direction.json"   # user confirmation, V4_PLAN §11.9


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

    Checks harm_v1.json against the manifest's config sha256, the rule and build code against the sha256s
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
    for rel, want in json.loads(cfg)["code_sha256"].items():     # rule and build code, as calibrated
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == want, f"{rel} changed since calibration"
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


def _atomic_text(path, text):
    tmp = path.with_name(f".{path.name}.tmp42")
    tmp.write_text(text)
    os.replace(tmp, path)


def _arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _digest(shas):
    """sha256 of the sorted 'case:sha256' lines (labels: the manifest's sha256_label; images: the CT arrays)."""
    return hashlib.sha256("".join(f"{c}:{shas[c]}\n" for c in sorted(shas)).encode()).hexdigest()


def _write42(a):
    c, what, cache, raw, overwrite = a
    import nibabel as nib
    raw = Path(raw)
    lab = v42_label(np.load(Path(cache) / f"{c}_label.npy"), what)
    ct = np.asarray(np.load(Path(cache) / f"{c}_image.npy", mmap_mode="r"), np.float32)  # links to the cache CT
    img = raw / "imagesTr" / f"{c}_0000.nii.gz"
    if what == "s1":   # same CT as Dataset540: link its raw image instead of writing a second copy
        src = NNUNET_HOME / "nnUNet_raw" / V42["binY"][1] / "imagesTr" / f"{c}_0000.nii.gz"
        assert src.exists(), f"{src} missing: run prep42 --what binY first"
        tmp = img.with_name(f".{img.name}.tmp42")
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(src)
        os.replace(tmp, img)
    elif (overwrite or not img.exists()
          or not np.array_equal(np.asarray(nib.load(str(img)).dataobj).transpose(2, 1, 0), ct)):
        _atomic_nifti(ct, img, np.float32)   # missing, asked for, or not this cache's CT
    out = raw / "labelsTr" / f"{c}.nii.gz"
    _atomic_nifti(lab, out, np.uint8)     # always rewritten: labels follow the verified cache, never a stale file
    back = np.asarray(nib.load(str(out)).dataobj).transpose(2, 1, 0)
    assert np.array_equal(back, lab), f"{c}: label read-back mismatch"
    return c, [int((lab == v).sum()) for v in range(5)], _arr_sha(ct)


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
    done = ROOT / "log" / "v4" / f"nnprep_{did}.done"
    if args.cases:
        want = set(args.cases.split(","))
        assert not want - set(cases), f"unknown cases: {sorted(want - set(cases))}"
        cases = [c for c in cases if c in want]
        # a smoke subset would rewrite the full dataset's dataset.json / splits_final.json in place
        extra = sorted({p.name.split("_")[0] for p in (raw / "imagesTr").glob("*.nii.gz")} - want)
        assert not extra, f"--cases requires an empty/subset raw dir; {name} holds other cases: {extra[:5]}"
    else:
        if what == "s1":
            assert (ROOT / "log" / "v4" / "nnprep_540.done").exists(), \
                "Dataset540 is not complete: prep42 --what binY first"
        done.unlink(missing_ok=True)      # a rebuild is not done (for the queue) until it has finished
    cache, harm_sha, digest = harm_cache()
    img_sha = {}
    with Pool(args.np) as p:
        jobs = [(c, what, str(cache), str(raw), args.overwrite) for c in cases]
        for i, (c, counts, s) in enumerate(p.imap_unordered(_write42, jobs)):
            img_sha[c] = s
            print(f"[{i + 1}/{len(cases)}] {c} voxels per label value {counts}", flush=True)
    labels = ({"background": 0, "Y": 1} if what == "binY"
              else {"background": 0, "c1": 1, "c2": 2, "c3": 3, "c4": 4})
    _atomic_text(raw / "dataset.json", json.dumps({
        "channel_names": {"0": "nonorm"}, "labels": labels, "numTraining": len(cases), "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIO",
        "description": f"CN V V4.2 ({what}): full-volume cache CT, labels from {HARM_DIR.name} "
                       f"({'binary Y' if what == 'binY' else 'classes 0..4, class 5 -> 0'}), {len(cases)} cases"},
        indent=2))
    _atomic_text(raw / "v42_meta.json", json.dumps({
        "what": what, "harm_cache": str(cache), "harm_config_sha256": harm_sha, "harm_label_digest": digest,
        "image_digest": _digest(img_sha), "harm_built": (cache / "DONE").read_text().strip(),
        "tool_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(), "versions": _versions(),
        "created": datetime.now().isoformat(), "cases": cases}, indent=1))
    pre = NNUNET_HOME / "nnUNet_preprocessed" / name
    pre.mkdir(parents=True, exist_ok=True)
    plans = json.loads((NNUNET_HOME / "nnUNet_preprocessed" / SRC_DATASET / f"{PLANS}.json").read_text())
    plans["dataset_name"] = name      # everything else as Dataset501 (patch, network, batch, spacing, norm)
    _atomic_text(pre / f"{PLANS}.json", json.dumps(plans, indent=2))
    _atomic_text(pre / "dataset.json", (raw / "dataset.json").read_text())
    splits = v42_splits(what)
    if args.cases:                    # smoke subsets: keep the indexing, drop the other cases
        keep = set(cases)
        splits = [{"train": [c for c in e["train"] if c in keep], "val": [c for c in e["val"] if c in keep]}
                  for e in splits]
    _atomic_text(pre / "splits_final.json", json.dumps(splits, indent=1))
    subprocess.run([bin_cmd("nnUNetv2_preprocess"), "-d", str(did), "-plans_name", PLANS, "-c", CONFIG,
                    "-np", str(args.np)], env=env(), check=True)
    subprocess.run([bin_cmd("nnUNetv2_extract_fingerprint"), "-d", str(did), "-np", str(args.np), "--clean"],
                   env=env(), check=True)   # --clean: a kept fingerprint would describe the previous labels
    pre_folder = pre / f"{PLANS}_{CONFIG}"
    subprocess.run([sys.executable, "-c",
                    "from nnunetv2.training.dataloading.utils import unpack_dataset; "
                    f"unpack_dataset('{pre_folder}', unpack_segmentation=True, "
                    f"overwrite_existing=False, num_processes={args.np}, verify_npy=True)"],
                   env=env(), check=True)
    if not args.cases:
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(f"{datetime.now().isoformat()} {name} {HARM_DIR.name} config {harm_sha[:12]} "
                        f"labels {digest[:12]} images {_digest(img_sha)[:12]}\n")
    print(f"prepared {name}")


def check_direction():
    """The user's confirmation of the harm_v1 direction names exactly the current cache (V4_PLAN §11.3, §11.9).
    Returns (config sha256, label digest)."""
    _, sha, digest = harm_cache(verify=False)
    if not HARM_DIRECTION.exists():
        raise SystemExit(f"{HARM_DIRECTION} missing: the harm_v1 direction is not confirmed (V4_PLAN §11.9)")
    d = json.loads(HARM_DIRECTION.read_text())
    assert d.get("status") == "confirmed" and (d.get("harm_config_sha256"), d.get("harm_label_digest")) == (sha, digest), \
        f"{HARM_DIRECTION} does not confirm the current harm_v1 cache"
    return sha, digest


def check_dataset42(name):
    """Before any V4.2 training: the harm_v1 direction is confirmed (check_direction) and the dataset was built
    from that cache. Returns its v42_meta.json."""
    sha, digest = check_direction()
    meta = json.loads((NNUNET_HOME / "nnUNet_raw" / name / "v42_meta.json").read_text())
    assert (meta.get("harm_config_sha256"), meta.get("harm_label_digest")) == (sha, digest), \
        f"{name} was built from other labels: rebuild it with prep42"
    return meta


def trainer_class(trainer):
    """The class nnU-Net itself resolves for this name: output paths and run records follow it, not the name."""
    import nnunetv2
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    cls = recursive_find_python_class(str(Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer"),
                                      trainer, "nnunetv2.training.nnUNetTrainer")
    if cls is None:
        raise SystemExit(f"nnU-Net finds no trainer class named {trainer}")
    return cls


def is_v4(cls):
    return any(k.__name__ == "nnUNetTrainerV4" for k in cls.__mro__)


def _npz_shape(p, key):
    """Shape in the header of array `key` of npz file p, or None if the file or the array is unreadable."""
    import zipfile
    try:
        with zipfile.ZipFile(p) as zf, zf.open(f"{key}.npy") as f:
            major, _ = np.lib.format.read_magic(f)
            read = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
            return tuple(read(f)[0])
    except Exception:
        return None


def _npz_ok(p, shape, n_cls):
    """validation/<case>.npz is a readable npz whose 'probabilities' header says (n_cls, *shape)."""
    return _npz_shape(p, "probabilities") == (n_cls,) + tuple(shape)


def _log_run(fd, trainer, cls, meta, cmd, extra=None):
    """Append what this launch ran with to fd/v42_runs.jsonl: code, the trainer class's files, dataset labels."""
    import inspect
    files = sorted({inspect.getfile(k) for k in cls.__mro__ if k.__module__.startswith("nnunetv2")})
    rec = {"time": datetime.now().isoformat(), "cmd": cmd, "trainer": trainer,
           "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                      text=True).stdout.strip(),
           "trainer_files_sha256": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
           "dataset": {k: meta.get(k) for k in ("harm_config_sha256", "harm_label_digest", "image_digest", "created",
                                                "priors_digest", "qc42_sha256") if k in meta},
           "versions": _versions(), **(extra or {})}
    fd.mkdir(parents=True, exist_ok=True)
    with open(fd / "v42_runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def train42(args):
    what = args.dataset
    if what == "prior":
        return train42p(args)
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
    meta = check_dataset42(name)
    cls = trainer_class(trainer)
    stock = not is_v4(cls)             # nnU-Net's own validation/ output (R0u, Stage 1, stock smoke)
    splits = json.loads((NNUNET_HOME / "nnUNet_preprocessed" / name / "splits_final.json").read_text())
    assert split < len(splits) and splits[split]["train"] and splits[split]["val"], \
        f"{name}: splits_final.json has no usable entry {split}"
    val = splits[split]["val"]
    fd = results_dir(did, name, trainer, split)

    def missing_npz():
        return [c for c in val if not _npz_ok(fd / "validation" / f"{c}.npz",
                                              np.load(CACHE_DIR / f"{c}_label.npy", mmap_mode="r").shape, 5)]

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
        _log_run(fd, trainer, cls, meta, cmd)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env(), check=True)
    if what == "binY":
        export42(args, fd, stock, trainer, meta["harm_label_digest"])
        return
    miss = missing_npz()
    assert not miss, f"{fd}: no usable Stage 1 probabilities (validation/*.npz) for {miss[:5]}"
    if trainer != R0_TRAINER:               # smoke runs never mark the production queue done
        print(f"Stage 1 smoke ({trainer}) done, no queue marker: {fd}")
        return
    mark = ROOT / "log" / "v4" / f"s142_{args.variant}_f{args.fold}_i{args.inner}.done"
    mark.parent.mkdir(parents=True, exist_ok=True)
    mark.write_text(f"{datetime.now().isoformat()} {fd} npz {len(val)} labels {meta['harm_label_digest'][:12]}\n")
    print(f"Stage 1 {args.variant} fold {args.fold} inner {args.inner} done: {fd}")


def export42(args, fd, stock, trainer="", digest=""):
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
    # case subsets and smoke trainers never write the marker the queue and the dev summary wait for
    tag = "EXPORTED_PARTIAL" if (args.cases or "SMOKE" in trainer) else "EXPORTED"
    marker = out / (tag if args.variant == "dev" else f"{tag}_fold{k}")
    marker.write_text(f"{datetime.now().isoformat()} {len(cases)} cases from {fd} trainer {trainer} "
                      f"labels {digest[:12]}\n")
    print(f"exported {len(cases)} cases -> {out} ({marker.name})")


# ---------------- V4.2 prior datasets 542-556 and D2' (V4_PLAN §11.4, §11.6.1-11.6.2) ----------------

PRIORS42 = Path(os.environ.get("V42_PRIORS_DIR", str(RESULTS / "v4_priors42")))   # override: smoke tests
QC42 = Path(os.environ.get("V42_QC_DIR", str(ROOT / "configs" / "v4_qc42")))   # D1-AM2 freeze, write-once
SELECT42 = RESULTS / "v4_select42"              # D4' / D2' manifests (tools/v4_select42.py)
COMBOS = ("ab", "ac", "bc", "abc")
W_SUFFIX = {"S0": "", "S05": "_S05", "S10": "_S10"}   # D4' choice -> prior trainer name suffix


def v42p(fold, variant, shuffle=False):
    """(id, name) of the 4-channel full-image dataset: 542+k dev, 547+k report, 552+k report with shuffled priors."""
    assert 0 <= fold < 5 and variant in ("dev", "report") and not (shuffle and variant == "dev"), (fold, variant, shuffle)
    did = (542 if variant == "dev" else 552 if shuffle else 547) + fold
    return did, f"Dataset{did}_CNVv42{variant}F{fold}{'shuf' if shuffle else ''}"


def copy_id(case, j):
    """Prior copy j of a held-out case: priors from Stage 1 inner model j (V_PLAN §11.4 point 3)."""
    return f"{case}_j{j}"


def split_copy(i):
    case, j = i.rsplit("_j", 1)
    return case, int(j)


def holdout_cases(fold, variant):
    """Cases no Stage 1 inner model of the tier saw: V_k (dev) or report fold k (report)."""
    return internal_split(fold)[1] if variant == "dev" else report_cases(fold)


def stage2_train_cases(fold, variant):
    return internal_split(fold)[0] if variant == "dev" else train_cases(fold)


def s1_dir(fold, variant, inner, trainer=R0_TRAINER):
    did, name = V42["s1"]
    split = v42_split_index("s1", fold, variant, inner)
    return results_dir(did, name, trainer, split), split


def prior_sha(prior, bbox):
    """Content hash of one prior file (array bytes + bbox), independent of the npz container's timestamps."""
    h = hashlib.sha256(np.ascontiguousarray(prior, np.uint8).tobytes())
    h.update(json.dumps([list(prior.shape), np.asarray(bbox).astype(int).tolist()]).encode())
    return h.hexdigest()


def _file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _save_npz(path, **arrays):
    """np.savez_compressed through a temporary name + rename (no partial file under the final name)."""
    tmp = path.with_name(f".{path.name}.tmp42")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def _predict42(fd, split, pre_name, ids, out_dir, kind, device="cuda"):
    """Sliding-window inference of fd's checkpoint_final on the preprocessed cases `ids`, as nnU-Net's own
    validation does it (tile step 0.5, Gaussian, mirroring). kind 's1': 'probabilities' (n_cls, *cache shape)
    through nnU-Net's own convert_predicted_logits_to_segmentation_with_correct_shape, i.e. the format of
    validation/*.npz written with --npz. kind 'pY': V4 arm, softmax of the Y head on the full volume, after
    zeroing the prior channels the arm does not use, exactly as nnUNetTrainerV4.perform_actual_validation."""
    os.environ.setdefault("nnUNet_compile", "f")
    import torch
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
    dev = torch.device(device)
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                           perform_everything_on_device=dev.type == "cuda", device=dev, verbose=False,
                           verbose_preprocessing=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(str(fd.parent), use_folds=(split,), checkpoint_name="checkpoint_final.pth")
    cls = trainer_class(pred.trainer_name)
    assert list(pred.plans_manager.transpose_forward) == [0, 1, 2], "plans transpose the axes: export would differ"
    zero = []
    if is_v4(cls):
        assert kind == "pY", f"{cls.__name__} is a V4 arm: use kind 'pY'"
        mod = sys.modules[cls.__module__]
        pc = mod.PRIOR_CHANNELS
        zero = sorted(c for c in pc.values() if c not in {pc[p] for p in cls.PRIORS})
        # the sliding-window buffer follows label_manager: give it the arm's head count, as its validation does
        pred.label_manager = mod._NOutLabelManager(pred.label_manager, cls._n_out())
    else:
        assert kind == "s1", f"{cls.__name__} is stock: use kind 's1'"
    pre_folder = NNUNET_HOME / "nnUNet_preprocessed" / pre_name / f"{PLANS}_{CONFIG}"
    ds = nnUNetDataset(str(pre_folder), case_identifiers=list(ids), num_images_properties_loading_threshold=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = "probabilities" if kind == "s1" else "pY"
    # outputs are reused on resume only if they come from this very checkpoint and this tool
    source = {"checkpoint_sha256": _file_sha(fd / "checkpoint_final.pth"), "kind": kind, "split": split,
              "tool_sha256": _file_sha(Path(__file__).resolve())}
    src_file = out_dir / "SOURCE.json"
    if not src_file.exists() or json.loads(src_file.read_text()) != source:
        for old in out_dir.glob("*.npz"):
            old.unlink()
        _atomic_text(src_file, json.dumps(source, indent=1))
    for n, i in enumerate(ids):
        out = out_dir / f"{i}.npz"
        data, _, props = ds.load_case(i)
        full = tuple(int(s) for s in props["shape_before_cropping"])
        want = ((pred.label_manager.num_segmentation_heads,) if kind == "s1" else ()) + full
        if _npz_shape(out, key) == want:
            continue                                 # resumed: this case is already written
        x = torch.from_numpy(np.asarray(data, dtype=np.float32))
        if zero and x.shape[0] > max(zero):
            x = x.clone()
            x[zero] = 0
        logits = pred.predict_logits_from_preprocessed_data(x)
        if kind == "s1":
            _, probs = convert_predicted_logits_to_segmentation_with_correct_shape(
                logits, pred.plans_manager, pred.configuration_manager, pred.label_manager, props,
                return_probabilities=True)
            _save_npz(out, probabilities=probs)
        else:
            y = torch.softmax(logits.float()[:2], 0)[1].numpy()
            bbox = props["bbox_used_for_cropping"]
            assert tuple(y.shape) == tuple(b - a for a, b in bbox), (i, y.shape, bbox)   # no resampling
            p = np.zeros(full, np.float32)
            p[tuple(slice(a, b) for a, b in bbox)] = y
            _save_npz(out, pY=p)
        print(f"[{n + 1}/{len(ids)}] {i} -> {out}", flush=True)
        del logits
    return out_dir


def predict42(args):
    """Stage 1 inner model j of fold k on the cases it never saw (V_k or report fold k) -> <fold dir>/holdout42/."""
    k, v, j = args.fold, args.variant, args.inner
    if args.dataset != "s1" or j is None:
        raise SystemExit("predict42 is for Stage 1 (--dataset s1 --inner 0..2)")
    trainer = args.trainer or R0_TRAINER
    if trainer not in (R0_TRAINER, SMOKE_STOCK):
        raise SystemExit("Stage 1 is the stock nnU-Net trainer (V4_PLAN §10.3)")
    name = V42["s1"][1]
    meta = check_dataset42(name)
    fd, split = s1_dir(k, v, j, trainer)
    ckpt = fd / "checkpoint_final.pth"
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} missing: Stage 1 {v} fold {k} inner {j} has not finished")
    cases = list(holdout_cases(k, v))
    held = set(json.loads((NNUNET_HOME / "nnUNet_preprocessed" / name / "splits_final.json").read_text())[split]["train"])
    assert not held & set(cases), "a holdout case is in this Stage 1 model's training set"
    if args.cases:
        keep = set(args.cases.split(","))
        cases = [c for c in cases if c in keep]
        assert cases, "no requested case is a holdout case of this fold/variant"
    out = _predict42(fd, split, name, cases, fd / "holdout42", "s1", args.device)
    bad = [c for c in cases if not _npz_ok(out / f"{c}.npz", np.load(CACHE_DIR / f"{c}_label.npy", mmap_mode="r").shape, 5)]
    assert not bad, f"unusable holdout probabilities: {bad[:5]}"
    if args.cases or trainer != R0_TRAINER:
        print(f"Stage 1 holdout prediction ({trainer}, {len(cases)} cases) done, no queue marker: {out}")
        return
    mark = ROOT / "log" / "v4" / f"s1p42_{v}_f{k}_i{j}.done"
    mark.write_text(f"{datetime.now().isoformat()} {out} npz {len(cases)} checkpoint {_file_sha(ckpt)[:12]} "
                    f"labels {meta['harm_label_digest'][:12]}\n")
    print(f"Stage 1 {v} fold {k} inner {j}: {len(cases)} holdout cases -> {out}")


def _atomic_link(src, dst):
    tmp = dst.with_name(f".{dst.name}.tmp42")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(src)
    os.replace(tmp, dst)


def _write42p(a):
    i, case, pdir, raw, shuffle, want_sha = a
    import nibabel as nib
    raw = Path(raw)
    src = NNUNET_HOME / "nnUNet_raw" / V42["binY"][1]
    for s, d in ((src / "imagesTr" / f"{case}_0000.nii.gz", raw / "imagesTr" / f"{i}_0000.nii.gz"),
                 (src / "labelsTr" / f"{case}.nii.gz", raw / "labelsTr" / f"{i}.nii.gz")):
        assert s.exists(), f"{s} missing: Dataset540 is not built"
        _atomic_link(s, d)                       # CT and binary Y exactly as Dataset540
    with np.load(Path(pdir) / f"{i}_prior.npz") as d:
        pri, bbox = d["prior"], d["bbox"]
    assert prior_sha(pri, bbox) == want_sha, f"{i}: prior file differs from the priors DONE record"
    shape = tuple(np.load(CACHE_DIR / f"{case}_label.npy", mmap_mode="r").shape)
    box = [[int(lo), int(hi)] for lo, hi in bbox]
    if pri.size == 0:
        chans = np.zeros((3,) + shape, np.uint8)          # fallback / no support: all priors zero
    else:
        chans = region_from_box(pri, box, [[0, n] for n in shape], full_shape=shape,
                                shift=case_shift(case, shape) if shuffle else None)
    for ch in range(3):
        arr = chans[ch].astype(np.float32) / np.float32(255.0)
        path = raw / "imagesTr" / f"{i}_{ch + 1:04d}.nii.gz"
        _atomic_nifti(arr, path, np.float32)
        back = np.asarray(nib.load(str(path)).dataobj).transpose(2, 1, 0)
        assert np.array_equal(back, arr), f"{path}: read-back mismatch"
    return i, [int((chans[ch] > 0).sum()) for ch in range(3)]


def prep42p(args):
    """4-channel full-image dataset (CT + priors a/b/c in [0, 1]) from results/v4_priors42/fold{k}_{variant}."""
    from multiprocessing import Pool
    k, variant, shuffle = args.fold, args.variant, args.shuffle
    if variant != "dev":
        raise SystemExit("report-tier prior datasets wait for the per-fold V4.2 freeze (V4_PLAN §11.6.1)")
    did, name = v42p(k, variant, shuffle)
    pdir = PRIORS42 / f"fold{k}_{variant}"
    if not (pdir / "DONE").exists():
        raise SystemExit(f"{pdir}/DONE missing: build the priors first (tools/v4_build_priors42.py)")
    rec = json.loads((pdir / "DONE").read_text())
    sha, digest = check_direction()
    assert (rec["harm_config_sha256"], rec["harm_label_digest"]) == (sha, digest), "priors were built from other labels"
    assert rec["fold"] == k and rec["variant"] == variant, (rec["fold"], rec["variant"])
    qc = QC42 / f"fold{k}.json"
    assert _file_sha(qc) == rec["qc42_sha256"], f"{qc} changed since the priors were built"
    m540 = check_dataset42(V42["binY"][1])
    ids = rec["ids"]                               # {id: {"case": ..., "sha": ...}}
    train = list(stage2_train_cases(k, variant))
    val = [copy_id(c, j) for c in holdout_cases(k, variant) for j in range(N_INNER)]
    raw = NNUNET_HOME / "nnUNet_raw" / name
    order = train + val
    done = ROOT / "log" / "v4" / f"nnprep_{did}.done"
    if args.cases:                                 # smoke subsets: these cases (and their copies) only
        want = set(args.cases.split(","))
        order = [i for i in order if (split_copy(i)[0] if i in val else i) in want]
        assert order and all(i in ids for i in order), "case subset is empty or has no priors"
    else:
        assert sorted(ids) == sorted(order), "priors DONE does not list exactly the training cases + holdout copies"
        done.unlink(missing_ok=True)
    (raw / "imagesTr").mkdir(parents=True, exist_ok=True)
    (raw / "labelsTr").mkdir(parents=True, exist_ok=True)
    have = ({p.name[:-len("_0000.nii.gz")] for p in (raw / "imagesTr").glob("*_000[0-9].nii.gz")}
            | {p.name[:-len(".nii.gz")] for p in (raw / "labelsTr").glob("*.nii.gz")})
    stale = sorted(have - set(order))
    assert not stale, f"{raw} holds cases outside this build: {stale[:5]}"
    nz = {}
    with Pool(args.np) as p:
        jobs = [(i, ids[i]["case"], str(pdir), str(raw), shuffle, ids[i]["sha"]) for i in order]
        for n, (i, cnt) in enumerate(p.imap_unordered(_write42p, jobs)):
            nz[i] = cnt
            print(f"[{n + 1}/{len(order)}] {i} non-zero prior voxels a/b/c {cnt}", flush=True)
    _atomic_text(raw / "dataset.json", json.dumps({
        "channel_names": {"0": "nonorm", "1": "nonorm", "2": "nonorm", "3": "nonorm"},
        "labels": {"background": 0, "Y": 1}, "numTraining": len(order), "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIO",
        "description": f"CN V V4.2 fold {k} {variant}{' shuffled priors' if shuffle else ''}: full-volume cache CT + "
                       f"priors a/b/c (D1-AM2), binary Y from {HARM_DIR.name}; holdout cases as 3 prior copies"},
        indent=2))
    _atomic_text(raw / "v42_meta.json", json.dumps({
        "what": "prior", "fold": k, "variant": variant, "shuffle": shuffle,
        "harm_cache": m540["harm_cache"], "harm_config_sha256": sha, "harm_label_digest": digest,
        "image_digest": m540["image_digest"], "priors_dir": str(pdir), "priors_digest": rec["digest"],
        "qc42_sha256": rec["qc42_sha256"], "margin": rec["margin"],
        "tool_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(), "versions": _versions(),
        "created": datetime.now().isoformat(), "ids": order, "nonzero_prior_voxels": nz}, indent=1))
    pre = NNUNET_HOME / "nnUNet_preprocessed" / name
    pre.mkdir(parents=True, exist_ok=True)
    plans = widen_plans(NNUNET_HOME / "nnUNet_preprocessed" / SRC_DATASET / f"{PLANS}.json", name)
    _atomic_text(pre / f"{PLANS}.json", json.dumps(plans, indent=2))
    _atomic_text(pre / "dataset.json", (raw / "dataset.json").read_text())
    keep = set(order)
    splits = [{"train": [i for i in train if i in keep], "val": [i for i in val if i in keep]}]
    assert splits[0]["train"] and splits[0]["val"], "empty split"
    _atomic_text(pre / "splits_final.json", json.dumps(splits, indent=1))
    subprocess.run([bin_cmd("nnUNetv2_preprocess"), "-d", str(did), "-plans_name", PLANS, "-c", CONFIG,
                    "-np", str(args.np)], env=env(), check=True)
    subprocess.run([bin_cmd("nnUNetv2_extract_fingerprint"), "-d", str(did), "-np", str(args.np), "--clean"],
                   env=env(), check=True)
    pre_folder = pre / f"{PLANS}_{CONFIG}"
    subprocess.run([sys.executable, "-c",
                    "from nnunetv2.training.dataloading.utils import unpack_dataset; "
                    f"unpack_dataset('{pre_folder}', unpack_segmentation=True, "
                    f"overwrite_existing=False, num_processes={args.np}, verify_npy=True)"],
                   env=env(), check=True)
    if not args.cases:
        done.write_text(f"{datetime.now().isoformat()} {name} priors {rec['digest'][:12]} "
                        f"labels {digest[:12]} m {rec['margin']}\n")
    print(f"prepared {name}")


def check_dataset42p(name):
    """check_dataset42, plus: the priors and the D1-AM2 config are still the ones the dataset was built from."""
    meta = check_dataset42(name)
    assert meta.get("what") == "prior", f"{name} is not a prior dataset"
    rec = json.loads((Path(meta["priors_dir"]) / "DONE").read_text())
    assert rec["digest"] == meta["priors_digest"], f"{name}: priors changed since the dataset was built"
    assert _file_sha(QC42 / f"fold{meta['fold']}.json") == meta["qc42_sha256"], f"{name}: D1-AM2 config changed"
    return meta


def d4_choice(fold):
    """(w*_k, sha256 of d4.json) from fold k's D4' manifest (tools/v4_select42.py d4)."""
    p = SELECT42 / f"fold{fold}" / "d4.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: D2' needs fold {fold}'s D4' selection first (tools/v4_select42.py d4)")
    d = json.loads(p.read_text())
    assert d["step"] == "D4'" and d["fold"] == fold and d["chosen"]["w"] in W_SUFFIX, (d["step"], d["fold"])
    return d["chosen"]["w"], _file_sha(p)


def prior_trainer(combo, w):
    return f"nnUNetTrainerV4_P{combo}{W_SUFFIX[w]}"


def train42p(args):
    """D2' prior arm on Dataset542+k: train, predict the 3 prior copies of every V_k case, average p_Y, export."""
    k = args.fold
    if args.variant != "dev":
        raise SystemExit("report-tier prior runs wait for the per-fold V4.2 freeze (V4_PLAN §11.6.1)")
    if args.priors not in COMBOS:
        raise SystemExit(f"--priors must be one of {COMBOS} (D2', V4_PLAN §11.6.1)")
    if not args.name:
        raise SystemExit("--name is required (v42_d2<combo> for D2')")
    did, name = v42p(k, "dev")
    if args.trainer:
        if "SMOKE" not in args.trainer:
            raise SystemExit("production prior trainers follow d4.json; --trainer is for smoke trainers only")
        trainer, w, d4sha = args.trainer, None, None
    else:
        w, d4sha = d4_choice(k)
        trainer = prior_trainer(args.priors, w)
    meta = check_dataset42p(name)
    cls = trainer_class(trainer)
    assert is_v4(cls) and "".join(sorted(cls.PRIORS)) == args.priors, f"{trainer} does not use priors {args.priors}"
    split = 0
    splits = json.loads((NNUNET_HOME / "nnUNet_preprocessed" / name / "splits_final.json").read_text())
    assert splits[split]["train"] and splits[split]["val"], f"{name}: no usable split entry"
    val = splits[split]["val"]
    fd = results_dir(did, name, trainer, split)
    import fcntl
    pre_folder = NNUNET_HOME / "nnUNet_preprocessed" / name / f"{PLANS}_{CONFIG}"
    with open(pre_folder / ".unpack_lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        subprocess.run([sys.executable, "-c",
                        "from nnunetv2.training.dataloading.utils import unpack_dataset; "
                        f"unpack_dataset('{pre_folder}', unpack_segmentation=True, "
                        "overwrite_existing=False, num_processes=8, verify_npy=True)"],
                       env=env(), check=True)
    cmd = [bin_cmd("nnUNetv2_train"), str(did), CONFIG, str(split), "-tr", trainer, "-p", PLANS]
    if (fd / "checkpoint_final.pth").exists():
        cmd = None if (fd / "validation_v4" / "DONE").exists() else cmd + ["--val"]
    elif (fd / "checkpoint_latest.pth").exists():
        cmd.append("--c")
    if cmd:
        _log_run(fd, trainer, cls, meta, cmd, extra={"d2": {"priors": args.priors, "w": w, "d4_sha256": d4sha}})
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env(), check=True)
    _predict42(fd, split, name, val, fd / "pY42", "pY", args.device)
    export42p(args, fd, trainer, meta, val)


def export42p(args, fd, trainer, meta, val):
    """V_k prediction = mean p_Y over the 3 prior copies > 0.5 (V4_PLAN §11.4 point 3, §11.6.1)."""
    import nrrd
    k = args.fold
    by_case = {}
    for i in val:
        by_case.setdefault(split_copy(i)[0], []).append(i)
    cases = [c for c in holdout_cases(k, "dev") if c in by_case]
    full = not args.cases and "SMOKE" not in trainer
    if full:
        assert sorted(by_case) == sorted(holdout_cases(k, "dev")), "validation copies do not cover V_k"
        assert all(sorted(split_copy(i)[1] for i in v) == list(range(N_INNER)) for v in by_case.values()), \
            "every V_k case needs exactly one copy per Stage 1 inner model"
    out = RESULTS / "v4_dev_preds" / args.name / f"fold{k}"
    out.mkdir(parents=True, exist_ok=True)
    if not full and (out / "EXPORTED").exists():
        raise SystemExit(f"{out} holds a complete export: a partial one must use another --name")
    for m in ("EXPORTED", "EXPORTED_PARTIAL"):
        (out / m).unlink(missing_ok=True)       # no marker while the files change
    for c in cases:
        shape = np.load(CACHE_DIR / f"{c}_label.npy", mmap_mode="r").shape
        acc = np.zeros(shape, np.float32)
        for i in sorted(by_case[c]):
            p = np.load(fd / "pY42" / f"{i}.npz")["pY"]
            assert p.shape == shape, (i, p.shape, shape)
            acc += p
        pred = (acc / np.float32(len(by_case[c])) > 0.5).astype(np.int16)
        nrrd.write(str(out / f"{c}_stage2_label.nrrd"), pred, nrrd_header(pred))
    marker = out / ("EXPORTED" if full else "EXPORTED_PARTIAL")
    _atomic_text(marker, f"{datetime.now().isoformat()} {len(cases)} cases from {fd} trainer {trainer} "
                         f"labels {meta['harm_label_digest'][:12]} priors {meta['priors_digest'][:12]} "
                         f"mean of {N_INNER} prior copies\n")
    print(f"exported {len(cases)} cases -> {out} ({marker.name})")


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
    t42.add_argument("--dataset", choices=sorted(V42) + ["prior"], required=True)
    t42.add_argument("--fold", type=int, required=True)
    t42.add_argument("--variant", choices=["dev", "report"], required=True)
    t42.add_argument("--inner", type=int, default=None, help="Stage 1 (s1): inner model 0..2")
    t42.add_argument("--trainer", default=None, help="binY: default = stock R0u trainer; prior: smoke only")
    t42.add_argument("--name", default=None, help="binY / prior: output name (v42_* for dev candidates)")
    t42.add_argument("--priors", default=None, help="prior: ab, ac, bc or abc (trainer from d4.json)")
    t42.add_argument("--device", default="cuda", help="prior: device of the copy inference")
    t42.add_argument("--cases", default="", help="comma list (smoke tests): export only these cases")
    pr42 = sub.add_parser("predict42")
    pr42.add_argument("--dataset", choices=["s1"], default="s1")
    pr42.add_argument("--fold", type=int, required=True)
    pr42.add_argument("--variant", choices=["dev", "report"], required=True)
    pr42.add_argument("--inner", type=int, required=True)
    pr42.add_argument("--trainer", default=None, help="default stock R0u trainer; smoke: nnUNetTrainer_SMOKE_2x5")
    pr42.add_argument("--device", default="cuda")
    pr42.add_argument("--cases", default="", help="comma list (tests): these holdout cases only, no marker")
    pp42 = sub.add_parser("prep42p")
    pp42.add_argument("--fold", type=int, required=True)
    pp42.add_argument("--variant", choices=["dev", "report"], default="dev")
    pp42.add_argument("--shuffle", action="store_true")
    pp42.add_argument("--np", type=int, default=8)
    pp42.add_argument("--cases", default="", help="comma list (smoke tests): these cases and their copies only")
    a = ap.parse_args()
    {"install": install, "prep": prep, "prep_r0dev": prep_r0dev, "train": train,
     "prep42": prep42, "train42": train42, "predict42": predict42, "prep42p": prep42p}[a.cmd](a)


if __name__ == "__main__":
    main()
