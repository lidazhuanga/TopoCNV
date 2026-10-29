#!/usr/bin/env python3
"""V4.3 route B (docs/V4_PLAN.md §12): CT + MRI + MRI valid mask datasets and runs. Runs under the nnU-Net venv:

  PY=~/cnv/envs/nnunetv2_cnv/bin/python
  $PY tools/v4_mm43.py install43                      # trainer/v4/nnunet/nnUNetTrainerV43.py -> venv (symlink)
  $PY tools/v4_mm43.py prep43 --what mm               # Dataset557: CT + MRI + valid mask, binary Y (harm_v1)
  $PY tools/v4_mm43.py prep43 --what shuf --fold 0 --variant report
  $PY tools/v4_mm43.py train43 --what mm --fold 0 --variant dev --trainer nnUNetTrainerV43_MMd1 --name v43_mmd1

Data: the frozen MRI cache ~/cnv/mri/cache (FROZEN + SHA256SUMS, cnv-mri-prep). prep43 checks every listed
file against SHA256SUMS and records the sha256 of SHA256SUMS itself; train43 refuses a dataset whose record
differs from the cache's current SHA256SUMS. CT and labels are links to Dataset540. MM copies 540's splits;
MM-shuf uses a separate dataset and one split per fold/variant, with donors drawn only from that training pool.
Outputs: validation_v4/ -> results/v4_dev_preds/<name>/fold{k} (each case's MRI) and validation_v4_nomri/ ->
<name>_nomri/fold{k} (no MRI for any case); report tier -> results/v4_<name>[_nomri]_label_predictions.
Queue marker log/v4/nn43_<name>_{variant}_f{k}.done after both exports. Smoke trainers and case subsets write
EXPORTED_PARTIAL and no marker.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import v4_nnunet as N  # noqa: E402
from topocnv.data.utils import CACHE_DIR, RESULTS  # noqa: E402
from topocnv.data.v4_splits import all_cases, internal_split, report_cases  # noqa: E402

PROTOCOL = "V4.3 / V4-AM-20261028-03"
MRI_ROOT = Path(os.environ.get("V43_MRI_ROOT", str(Path.home() / "cnv" / "mri")))   # override: tests
MRI_CACHE = MRI_ROOT / "cache"
V43 = {"mm": (557, "Dataset557_CNVv43mm")}
N_CH = 3
SHUF_SEED = 20261028
SHUF_DATASET_BASE = 559
SHUF_POLICY = "train_pool_v1"
FROZEN43 = ROOT / "configs" / "v4_frozen43"
MRI_TRAINER_PREFIX = "nnUNetTrainerV43_"


def dataset43(what, fold=None, variant=None):
    if what == "mm":
        return V43[what]
    if what != "shuf" or fold is None or variant is None:
        raise SystemExit("MM-shuf requires --fold and --variant; the global Dataset558 is unsupported")
    split = N.v42_split_index("binY", fold, variant)
    did = SHUF_DATASET_BASE + split
    return did, f"Dataset{did}_CNVv43mmshuf_{variant}_f{fold}"


# ---------------- MRI cache ----------------

def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _sum_path(name):
    """SHA256SUMS lists cache files by bare name and qa_summary.csv, which lives one level up."""
    return MRI_ROOT / name if name == "qa_summary.csv" else MRI_CACHE / name


def mri_sums():
    """(sha256 of SHA256SUMS, {file name: sha256}) of the frozen cache; refuses an unfrozen one."""
    if not (MRI_CACHE / "FROZEN").exists():
        raise SystemExit(f"{MRI_CACHE}/FROZEN missing: the MRI cache is not frozen (cnv-mri-prep)")
    sums = {}
    for line in (MRI_CACHE / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            sha, name = line.split(None, 1)
            sums[name.strip().lstrip("*")] = sha
    return _sha_file(MRI_CACHE / "SHA256SUMS"), sums


def _check_one(a):
    name, want = a
    return name, _sha_file(_sum_path(name)) == want


def verify_mri_cache(np_=8):
    """Every file listed in SHA256SUMS still has its sha256; every case has _mri and _mri_valid listed."""
    from multiprocessing import Pool
    sums_sha, sums = mri_sums()
    miss = [c for c in all_cases() if f"{c}_mri.npy" not in sums or f"{c}_mri_valid.npy" not in sums]
    assert not miss, f"SHA256SUMS lacks the MRI or valid mask of {miss[:5]}"
    with Pool(np_) as p:
        bad = sorted(n for n, ok in p.imap_unordered(_check_one, sorted(sums.items())) if not ok)
    assert not bad, f"MRI cache files differ from SHA256SUMS: {bad[:5]}"
    return sums_sha, sums


from functools import lru_cache  # noqa: E402


@lru_cache(maxsize=1)
def reg_ok():
    """{case: 0/1} from qa_summary.csv (cnv-mri-prep's automatic registration rule; coverage never sets 0)."""
    import csv
    with open(MRI_ROOT / "qa_summary.csv", newline="") as f:
        rows = {r["case"]: int(r["reg_ok"]) for r in csv.DictReader(f)}
    miss = sorted(set(all_cases()) - set(rows))
    assert not miss, f"qa_summary.csv lacks {miss[:5]}"
    return {c: rows[c] for c in all_cases()}


def load_mri(case):
    """(MRI float32 [0, 1], valid mask uint8) on the cache grid; for reg_ok = 0 both are all zeros."""
    shape = tuple(np.load(CACHE_DIR / f"{case}_label.npy", mmap_mode="r").shape)
    mri = np.asarray(np.load(MRI_CACHE / f"{case}_mri.npy"), np.float32)
    valid = np.asarray(np.load(MRI_CACHE / f"{case}_mri_valid.npy"), np.uint8)
    assert mri.shape == valid.shape == shape, (case, mri.shape, valid.shape, shape)
    assert np.isfinite(mri).all() and mri.min() >= 0 and mri.max() <= 1, f"{case}: MRI outside [0, 1]"
    assert set(np.unique(valid)) <= {0, 1}, f"{case}: valid mask is not 0/1"
    return mri, valid


# ---------------- MM-shuf (V4_PLAN §12.2 point 3) ----------------

def derangement(cases, seed=SHUF_SEED):
    """{recipient: donor}, no fixed point; the first derangement drawn from default_rng(seed)."""
    cases = sorted(cases)
    if len(cases) < 2 or len(set(cases)) != len(cases):
        raise ValueError("derangement needs at least two distinct cases")
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(len(cases))
        if not (p == np.arange(len(cases))).any():
            return {cases[i]: cases[int(p[i])] for i in range(len(cases))}


def shuffle_assignment(fold, variant, cases=None):
    index = N.v42_split_index("binY", fold, variant)
    split = {key: sorted(value) for key, value in N.v42_splits("binY")[index].items()}
    if cases is not None:
        keep = set(cases)
        extra = keep - set(split["train"]) - set(split["val"])
        if extra:
            raise SystemExit(f"MM-shuf subset contains cases outside {variant} fold {fold}: {sorted(extra)}")
        split = {key: [c for c in value if c in keep] for key, value in split.items()}
    if not split["train"] or not split["val"]:
        raise SystemExit("MM-shuf needs nonempty train and validation partitions")
    ok = reg_ok()
    pool = [c for c in split["train"] if ok[c]]
    val = [c for c in split["val"] if ok[c]]
    seed = SHUF_SEED + index
    donors = {}
    if pool or val:
        if len(pool) < 2:
            raise SystemExit("MM-shuf needs at least two MRI-usable training cases; held-out donors are forbidden")
        donors = derangement(pool, seed)
        rng = np.random.default_rng([seed, 1])
        donors.update(zip(val, map(str, rng.choice(pool, size=len(val), replace=len(val) > len(pool)))))
    return {"policy": SHUF_POLICY, "fold": fold, "variant": variant, "seed": seed,
            "split": split, "donors": donors}


def centroid(mask):
    idx = np.nonzero(mask)
    return np.array([float(i.mean()) for i in idx]) if len(idx[0]) else None


def place(arr, shift, shape):
    """arr moved by the integer shift (dst = src + shift) into a zero array of `shape`; outside parts dropped."""
    out = np.zeros(shape, arr.dtype)
    src, dst = [], []
    for s, n_src, n_dst in zip(shift, arr.shape, shape):
        lo, hi = max(0, -s), min(n_src, n_dst - s)
        if hi <= lo:
            return out
        src.append(slice(lo, hi))
        dst.append(slice(lo + s, hi + s))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def shuffled_mri(recipient, donor):
    """Donor MRI + mask, donor mask centroid moved onto the recipient's own mask centroid (no labels used)."""
    shape = tuple(np.load(CACHE_DIR / f"{recipient}_label.npy", mmap_mode="r").shape)
    _, own = load_mri(recipient)
    mri, valid = load_mri(donor)
    cr, cd = centroid(own), centroid(valid)
    assert cr is not None and cd is not None, (recipient, donor, "empty valid mask")
    shift = [int(round(x)) for x in cr - cd]
    return place(mri, shift, shape), place(valid, shift, shape), shift


# ---------------- datasets 557 / 558 ----------------

def widen_plans3(src, dst_name):
    p = json.loads(Path(src).read_text())
    p["dataset_name"] = dst_name
    for c in p["configurations"].values():
        if "normalization_schemes" in c:
            assert c["normalization_schemes"] == ["NoNormalization"], c["normalization_schemes"]
            c["normalization_schemes"] = ["NoNormalization"] * N_CH
        if "use_mask_for_norm" in c:
            c["use_mask_for_norm"] = [False] * N_CH
    fip = p.get("foreground_intensity_properties_per_channel", {})
    if "0" in fip:
        p["foreground_intensity_properties_per_channel"] = {str(i): fip["0"] for i in range(N_CH)}
    return p


def _write43(a):
    case, what, raw, donor = a
    import nibabel as nib
    raw = Path(raw)
    src = N.NNUNET_HOME / "nnUNet_raw" / N.V42["binY"][1]
    for s, d in ((src / "imagesTr" / f"{case}_0000.nii.gz", raw / "imagesTr" / f"{case}_0000.nii.gz"),
                 (src / "labelsTr" / f"{case}.nii.gz", raw / "labelsTr" / f"{case}.nii.gz")):
        assert s.exists(), f"{s} missing: Dataset540 is not built"
        N._atomic_link(s, d)
    shift = None
    if what == "shuf" and reg_ok()[case] and donor is None:
        raise SystemExit(f"{case}: MM-shuf is missing a training-pool donor")
    if donor is None:
        mri, valid = load_mri(case)
        assert reg_ok()[case] == 1 or (not mri.any() and not valid.any()), f"{case}: reg_ok = 0 but MRI/mask not zero"
    else:
        assert what == "shuf" and reg_ok()[case] and donor != case, f"{case}: invalid MM-shuf donor {donor}"
        mri, valid, shift = shuffled_mri(case, donor)
    for ch, arr, dt in ((1, mri, np.float32), (2, valid, np.uint8)):
        path = raw / "imagesTr" / f"{case}_{ch:04d}.nii.gz"
        N._atomic_nifti(arr, path, dt)
        back = np.asarray(nib.load(str(path)).dataobj).transpose(2, 1, 0)
        assert np.array_equal(back, arr), f"{path}: read-back mismatch"
    return case, int(valid.sum()), N._arr_sha(mri), N._arr_sha(valid), shift


def prep43(args):
    from multiprocessing import Pool
    did, name = dataset43(args.what, getattr(args, "fold", None), getattr(args, "variant", None))
    raw = N.NNUNET_HOME / "nnUNet_raw" / name
    pre = N.NNUNET_HOME / "nnUNet_preprocessed" / name
    if args.what == "shuf":
        for folder in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            existing = sorted((N.NNUNET_HOME / folder).glob(f"Dataset{did}_*"))
            if existing:
                raise SystemExit(f"refusing to overwrite MM-shuf dataset ID {did}: {existing}; inspect it or use a separate CNV_NNUNET_HOME")
    m540 = N.check_dataset42(N.V42["binY"][1])
    if not (ROOT / "log" / "v4" / "nnprep_540.done").exists() and not args.cases:
        raise SystemExit("Dataset540 is not complete")
    sums_sha, _ = verify_mri_cache(args.np) if not args.skip_verify else (mri_sums()[0], None)
    ok = reg_ok()
    shuffle = None
    if args.what == "shuf":
        shuffle = shuffle_assignment(args.fold, args.variant, args.cases.split(",") if args.cases else None)
        cases = sorted(shuffle["split"]["train"] + shuffle["split"]["val"])
    else:
        cases = list(all_cases())
        if args.cases:
            want = set(args.cases.split(","))
            assert not want - set(cases), f"unknown cases: {sorted(want - set(cases))}"
            cases = [c for c in cases if c in want]
    done = ROOT / "log" / "v4" / f"nnprep_{did}.done"
    if not args.cases:
        done.unlink(missing_ok=True)
    donors = shuffle["donors"] if shuffle else {}
    (raw / "imagesTr").mkdir(parents=True, exist_ok=True)
    (raw / "labelsTr").mkdir(parents=True, exist_ok=True)
    have = ({p.name.rsplit("_", 1)[0] for p in (raw / "imagesTr").glob("*.nii.gz")}
            | {p.name[:-len(".nii.gz")] for p in (raw / "labelsTr").glob("*.nii.gz")})
    stale = sorted(have - set(cases))
    assert not stale, f"{raw} holds cases outside this build: {stale[:5]}"
    rec = {}
    with Pool(args.np) as p:
        jobs = [(c, args.what, str(raw), donors.get(c)) for c in cases]
        for n, (c, nvalid, s_mri, s_valid, shift) in enumerate(p.imap_unordered(_write43, jobs)):
            rec[c] = {"valid_voxels": nvalid, "mri_sha": s_mri, "valid_sha": s_valid, "reg_ok": ok[c]}
            if args.what == "shuf":
                rec[c].update(donor=donors.get(c), shift=shift)
            print(f"[{n + 1}/{len(cases)}] {c} valid voxels {nvalid}" + (f" donor {donors.get(c)}" if donors else ""),
                  flush=True)
    N._atomic_text(raw / "dataset.json", json.dumps({
        "channel_names": {"0": "nonorm", "1": "nonorm", "2": "nonorm"},
        "labels": {"background": 0, "Y": 1}, "numTraining": len(cases), "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIO",
        "description": f"CN V V4.3 ({args.what}): cache CT + MRI ([0,1]) + MRI valid mask (fov AND reg_ok)"
                       f"{', MRI and mask of a training-pool donor' if args.what == 'shuf' else ''}, "
                       f"binary Y from {N.HARM_DIR.name}, {len(cases)} cases"}, indent=2))
    N._atomic_text(raw / "v42_meta.json", json.dumps({
        "what": f"mm43_{args.what}", "protocol": PROTOCOL, "harm_cache": m540["harm_cache"],
        "harm_config_sha256": m540["harm_config_sha256"], "harm_label_digest": m540["harm_label_digest"],
        "image_digest": m540["image_digest"], "mri_cache": str(MRI_CACHE), "mri_sums_sha256": sums_sha,
        "mri_frozen": (MRI_CACHE / "FROZEN").read_text().strip(),
        "mri_digest": N._digest({c: rec[c]["mri_sha"] + rec[c]["valid_sha"] for c in rec}),
        "shuffle_seed": shuffle["seed"] if shuffle else None, "shuffle": shuffle, "partial": bool(args.cases),
        "tool_sha256": N._file_sha(Path(__file__).resolve()), "versions": N._versions(),
        "created": datetime.now().isoformat(), "cases": rec}, indent=1))
    pre.mkdir(parents=True, exist_ok=True)
    src_pre = N.NNUNET_HOME / "nnUNet_preprocessed" / N.V42["binY"][1]
    N._atomic_text(pre / f"{N.PLANS}.json", json.dumps(widen_plans3(src_pre / f"{N.PLANS}.json", name), indent=2))
    N._atomic_text(pre / "dataset.json", (raw / "dataset.json").read_text())
    if shuffle:
        splits = [shuffle["split"]]
    else:
        splits = json.loads((src_pre / "splits_final.json").read_text())
        if args.cases:
            keep = set(cases)
            splits = [{"train": [c for c in e["train"] if c in keep], "val": [c for c in e["val"] if c in keep]}
                      for e in splits]
    N._atomic_text(pre / "splits_final.json", json.dumps(splits, indent=1))
    subprocess.run([N.bin_cmd("nnUNetv2_preprocess"), "-d", str(did), "-plans_name", N.PLANS, "-c", N.CONFIG,
                    "-np", str(args.np)], env=N.env(), check=True)
    subprocess.run([N.bin_cmd("nnUNetv2_extract_fingerprint"), "-d", str(did), "-np", str(args.np), "--clean"],
                   env=N.env(), check=True)
    pre_folder = pre / f"{N.PLANS}_{N.CONFIG}"
    subprocess.run([sys.executable, "-c",
                    "from nnunetv2.training.dataloading.utils import unpack_dataset; "
                    f"unpack_dataset('{pre_folder}', unpack_segmentation=True, "
                    f"overwrite_existing=False, num_processes={args.np}, verify_npy=True)"],
                   env=N.env(), check=True)
    if not args.cases:
        done.write_text(f"{datetime.now().isoformat()} {name} mri {sums_sha[:12]} labels "
                        f"{m540['harm_label_digest'][:12]}\n")
    print(f"prepared {name}")


def check_dataset43(name, fold=None, variant=None):
    """Check label/MRI provenance and the exact training-pool donor assignment for MM-shuf."""
    meta = N.check_dataset42(name)
    assert meta.get("protocol") == PROTOCOL and meta.get("what", "").startswith("mm43_"), f"{name} is not a V4.3 dataset"
    sums_sha, _ = mri_sums()
    assert meta["mri_sums_sha256"] == sums_sha, f"{name}: the MRI cache's SHA256SUMS changed since the build"
    if meta["what"] == "mm43_shuf":
        if name != dataset43("shuf", fold, variant)[1]:
            raise SystemExit(f"{name}: MM-shuf dataset does not match the requested fold/variant")
        expected = shuffle_assignment(fold, variant, list(meta["cases"]) if meta.get("partial") else None)
        if meta.get("shuffle") != expected or meta.get("shuffle_seed") != expected["seed"]:
            raise SystemExit(f"{name}: legacy/global or mismatched MM-shuf donors; rebuild a fold-specific dataset")
        assert sorted(meta["cases"]) == sorted(expected["split"]["train"] + expected["split"]["val"]), \
            f"{name}: MM-shuf case set does not match its split"
        ok = reg_ok()
        for case, rec in meta["cases"].items():
            assert rec.get("donor") == expected["donors"].get(case), f"{case}: MM-shuf donor mismatch"
            assert rec["reg_ok"] == ok[case], f"{case}: MRI availability changed"
        splits = json.loads((N.NNUNET_HOME / "nnUNet_preprocessed" / name / "splits_final.json").read_text())
        assert splits == [expected["split"]], f"{name}: MM-shuf requires exactly its own split at index 0"
    elif name != V43["mm"][1] or meta["what"] != "mm43_mm":
        raise SystemExit(f"{name}: dataset identity does not match the V4.3 arm")
    return meta


# ---------------- runs ----------------

def frozen_choice(k):
    p = FROZEN43 / f"fold{k}.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: V4.3 report runs wait for the freeze (V4_PLAN §12.5)")
    return json.loads(p.read_text())["chosen"], N._file_sha(p)


def check_trainer(what, k, variant, trainer):
    if not trainer.startswith(MRI_TRAINER_PREFIX):
        raise SystemExit(f"{trainer}: V4.3 runs use nnUNetTrainerV43_* trainers")
    if "SMOKE" in trainer:
        return None
    if variant == "dev":
        if what != "mm" or trainer not in ("nnUNetTrainerV43_MMd1", "nnUNetTrainerV43_MMd2"):
            raise SystemExit("V4.3 dev runs are MMd-1 / MMd-2 on Dataset557 only (V4_PLAN §12.5)")
        return None
    chosen, sha = frozen_choice(k)
    ok = {("mm", chosen["trainer"]), ("mm", chosen["trainer"].replace("MMd", "MMdcase")),
          ("mm", "nnUNetTrainerV43_MM"), ("shuf", "nnUNetTrainerV43_MM")}
    if (what, trainer) not in ok:
        raise SystemExit(f"report fold {k}: {trainer} on {what} is not one of the frozen arms {sorted(ok)}")
    return sha


def results_dirs(fd):
    return fd / "validation_v4", fd / "validation_v4_nomri"


def train43(args):
    k, variant, trainer = args.fold, args.variant, args.trainer
    did, name = dataset43(args.what, k, variant)
    if not args.name:
        raise SystemExit("--name is required")
    frozen_sha = check_trainer(args.what, k, variant, trainer)
    meta = check_dataset43(name, k, variant)
    if args.what == "shuf" and meta.get("partial") and "SMOKE" not in trainer:
        raise SystemExit("a partial MM-shuf dataset is for smoke trainers only")
    cls = N.trainer_class(trainer)
    assert any(c.__name__ == "nnUNetTrainerV43" for c in cls.__mro__), f"{trainer} is not a V4.3 trainer"
    split = 0 if args.what == "shuf" else N.v42_split_index("binY", k, variant)
    splits = json.loads((N.NNUNET_HOME / "nnUNet_preprocessed" / name / "splits_final.json").read_text())
    assert split < len(splits) and splits[split]["train"] and splits[split]["val"], f"{name}: no split {split}"
    fd = N.results_dir(did, name, trainer, split)
    import fcntl
    pre_folder = N.NNUNET_HOME / "nnUNet_preprocessed" / name / f"{N.PLANS}_{N.CONFIG}"
    with open(pre_folder / ".unpack_lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        subprocess.run([sys.executable, "-c",
                        "from nnunetv2.training.dataloading.utils import unpack_dataset; "
                        f"unpack_dataset('{pre_folder}', unpack_segmentation=True, "
                        "overwrite_existing=False, num_processes=8, verify_npy=True)"],
                       env=N.env(), check=True)
    cmd = [N.bin_cmd("nnUNetv2_train"), str(did), N.CONFIG, str(split), "-tr", trainer, "-p", N.PLANS]
    with_mri, no_mri = results_dirs(fd)
    if (fd / "checkpoint_final.pth").exists():
        ck = N._file_sha(fd / "checkpoint_final.pth")
        done = all((d / "DONE").exists() and (d / "CHECKPOINT").exists() and (d / "CHECKPOINT").read_text().strip() == ck
                   for d in (with_mri, no_mri))
        cmd = None if done else cmd + ["--val"]          # missing, or validated from other weights: validate again
    elif (fd / "checkpoint_latest.pth").exists():
        cmd.append("--c")
    if cmd:
        N._log_run(fd, trainer, cls, meta, cmd, extra={"mm43": {
            "protocol": PROTOCOL, "dataset": name, "mri_sums_sha256": meta["mri_sums_sha256"],
            "mri_digest": meta["mri_digest"], "frozen43_sha256": frozen_sha,
            "shuffle": meta.get("shuffle")}})
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=N.env(), check=True)
    export43(args, fd, trainer, meta)


def export43(args, fd, trainer, meta):
    import nrrd
    k = args.fold
    cases = internal_split(k)[1] if args.variant == "dev" else report_cases(k)
    if args.cases:
        keep = set(args.cases.split(","))
        cases = [c for c in cases if c in keep]
        assert cases, "no requested case is a validation case of this fold/variant"
    partial = bool(args.cases) or "SMOKE" in trainer
    ckpt = N._file_sha(fd / "checkpoint_final.pth")
    mark = ROOT / "log" / "v4" / f"nn43_{args.name}_{args.variant}_f{k}.done"
    mark.unlink(missing_ok=True)
    for sub, name in zip(results_dirs(fd), (args.name, f"{args.name}_nomri")):
        if not (sub / "DONE").exists():
            raise SystemExit(f"{sub}/DONE missing: validation did not finish")
        if not (sub / "CHECKPOINT").exists() or (sub / "CHECKPOINT").read_text().strip() != ckpt:
            raise SystemExit(f"{sub}: validated from other weights than the current checkpoint_final.pth")
        out = (RESULTS / "v4_dev_preds" / name / f"fold{k}" if args.variant == "dev"
               else RESULTS / f"v4_{name}_label_predictions")
        out.mkdir(parents=True, exist_ok=True)
        tags = ("EXPORTED", "EXPORTED_PARTIAL") if args.variant == "dev" else (f"EXPORTED_fold{k}",
                                                                                f"EXPORTED_PARTIAL_fold{k}")
        if partial and (out / tags[0]).exists():
            raise SystemExit(f"{out} holds a complete export: a partial one must use another --name")
        for t in tags:
            (out / t).unlink(missing_ok=True)
        for c in cases:
            pred = np.ascontiguousarray(np.load(sub / f"{c}.npz")["Y"]).astype(np.int16)
            shape = np.load(CACHE_DIR / f"{c}_label.npy", mmap_mode="r").shape
            assert pred.shape == shape, (c, pred.shape, shape)
            nrrd.write(str(out / f"{c}_stage2_label.nrrd"), pred, N.nrrd_header(pred))
        marker = out / tags[1 if partial else 0]
        N._atomic_text(marker, f"{datetime.now().isoformat()} {len(cases)} cases from {fd} trainer {trainer} "
                               f"labels {meta['harm_label_digest'][:12]} mri {meta['mri_sums_sha256'][:12]} "
                               f"condition {'no_mri' if sub.name.endswith('nomri') else 'with_mri'} "
                               f"checkpoint {ckpt}\n")
        print(f"exported {len(cases)} cases -> {out} ({marker.name})")
    if not partial:
        mark.parent.mkdir(parents=True, exist_ok=True)
        N._atomic_text(mark, f"{datetime.now().isoformat()} {fd} checkpoint {ckpt[:12]}\n")


def install43(_):
    import nnunetv2
    dst = Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer" / "variants" / "zz_cnv_v4"
    if not (dst / "nnUNetTrainerV4.py").exists():
        raise SystemExit(f"{dst}/nnUNetTrainerV4.py missing: run tools/v4_nnunet.py install first")
    link = dst / "nnUNetTrainerV43.py"
    target = ROOT / "trainer" / "v4" / "nnunet" / "nnUNetTrainerV43.py"
    if link.exists() and not link.is_symlink():
        raise SystemExit(f"{link} is a regular file, not our symlink: refusing to delete it")
    if link.is_symlink():
        link.unlink()
    link.symlink_to(target)
    print(f"installed {link} -> {link.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("install43")
    p = sub.add_parser("prep43")
    p.add_argument("--what", choices=["mm", "shuf"], required=True)
    p.add_argument("--fold", type=int, choices=range(5), help="required for MM-shuf")
    p.add_argument("--variant", choices=["dev", "report"], help="required for MM-shuf")
    p.add_argument("--np", type=int, default=8)
    p.add_argument("--cases", default="", help="comma list (smoke tests): tiny dataset of these cases only")
    p.add_argument("--skip_verify", action="store_true", help="smoke only: skip re-hashing the MRI cache")
    t = sub.add_parser("train43")
    t.add_argument("--what", choices=["mm", "shuf"], required=True)
    t.add_argument("--fold", type=int, choices=range(5), required=True)
    t.add_argument("--variant", choices=["dev", "report"], required=True)
    t.add_argument("--trainer", required=True)
    t.add_argument("--name", required=True)
    t.add_argument("--cases", default="", help="comma list (smoke tests): export only these cases")
    a = ap.parse_args()
    if a.cmd == "prep43" and a.skip_verify and not a.cases:
        raise SystemExit("--skip_verify is for smoke subsets (--cases) only")
    {"install43": install43, "prep43": prep43, "train43": train43}[a.cmd](a)


if __name__ == "__main__":
    main()
