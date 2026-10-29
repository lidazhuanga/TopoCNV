#!/usr/bin/env python3
"""Thickness-harmonised reference labels for V4.2 (amendment V4-AM-20261027-02). Run with the nnU-Net venv.

  python tools/v4_harmonize_labels.py calibrate   # freeze t from GT geometry -> configs/v4_harm/harm_v1.json
  python tools/v4_harmonize_labels.py build       # results/v4_harm_v1/sp_2.500_1.133_1.133/{case}_label.npy
  python tools/v4_harmonize_labels.py qa          # invariants, geometry tables, montages -> results/v4_harm_v1/qa/
  python tools/v4_harmonize_labels.py sanity      # r0_dev on V_k against original vs harmonised GT (dev only)

The thin rule, sigma and the t grid are fixed in trainer/v4/harmonize.py. calibrate uses GT geometry only: it
picks the t whose median harmonised thin-case thickness is closest to the median thickness of the thick cases
(ties -> larger t, i.e. less growth). calibrate and build are write-once (--force to override). The output
directory is a drop-in cache: harmonised labels are real files, {case}_image.npy are symlinks to the original
cache. The original cache is never written.
"""
import argparse
import csv
import datetime
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from topocnv.data.utils import CACHE_DIR, RESULTS, load_image, load_label  # noqa: E402
from topocnv.utils.box_ops import mask_Y  # noqa: E402
from topocnv.data.harmonize import (PLANE4, SIGMA, THIN_THR, T_GRID, harmonise, run_length,  # noqa: E402
                                  skeleton, thickness)
from topocnv.data.v4_splits import all_cases, internal_split  # noqa: E402

VERSION = "harm_v1"
AMENDMENT = "V4-AM-20261027-02"
CFG = REPO / "configs" / "v4_harm" / f"{VERSION}.json"
OUT = RESULTS / f"v4_{VERSION}" / CACHE_DIR.name
QA = RESULTS / f"v4_{VERSION}" / "qa"
CT_DATES = Path.home() / "cnv" / "mri" / "cases.csv"  # descriptive only (date window of the thin batch)
WORKERS = 8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_cfg():
    if not CFG.exists():
        raise SystemExit(f"{CFG} missing: run calibrate first")
    return json.loads(CFG.read_text())


# ---------------------------------------------------------------- calibrate
def _thick(c):
    return c, thickness(load_label(c, mmap=False))


def _thick_grid(c):
    lab = load_label(c, mmap=False)
    return c, [thickness(harmonise(lab, t)) for t in T_GRID]


def calibrate(args):
    if CFG.exists() and not args.force:
        raise SystemExit(f"{CFG} already frozen (write-once); pass --force to override")
    cases = list(all_cases())
    with Pool(WORKERS) as p:
        th = dict(p.map(_thick, cases))
    thin = sorted(c for c in cases if th[c] < THIN_THR)
    thick = [c for c in cases if th[c] >= THIN_THR]
    target = float(np.median([th[c] for c in thick]))
    with Pool(WORKERS) as p:
        grid = dict(p.map(_thick_grid, thin))
    med = np.median(np.array([grid[c] for c in thin]), axis=0)
    err = np.abs(med - target)
    best = max(i for i in range(len(T_GRID)) if err[i] == err.min())  # ties -> larger t
    t = T_GRID[best]
    window = {}
    if CT_DATES.exists():
        dates = {r["case"]: r["ct_date"][:7] for r in csv.DictReader(open(CT_DATES))}
        inwin = {c: "2024-12" <= dates.get(c, "") <= "2025-09" for c in cases}
        window = {"rule": "CT date in [2024-12, 2025-09] (descriptive, not used for selection)",
                  "agreement_with_thin_rule": float(np.mean([inwin[c] == (c in thin) for c in cases])),
                  "thin_outside_window": sorted(c for c in thin if not inwin[c]),
                  "window_not_thin": sorted(c for c in cases if inwin[c] and c not in thin)}
    cfg = {"amendment": AMENDMENT, "version": VERSION,
           "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "status": "draft: thin -> thick direction pending PI and clinician confirmation",
           "direction": "thin -> thick (majority convention; cases before 2024-12 and after 2025-09 are thick)",
           "rule": {"thickness": "Y voxels / 3-D skeleton voxels on the original label",
                    "thin_if_thickness_below": THIN_THR, "sigma_vox_zyx": list(SIGMA),
                    "growth": "in-plane 4-neighbours of Y with Gaussian(Y) >= t, never class 5; added voxel = "
                              "class of nearest original Y voxel (physical spacing); original voxels unchanged",
                    "calibration": "t on t_grid minimising |median harmonised thin thickness - median thick "
                                   "thickness| (GT geometry only, no predictions); ties -> larger t"},
           "t": t, "t_grid": list(T_GRID),
           "calibration_median_thin_thickness": {str(g): round(float(m), 4) for g, m in zip(T_GRID, med)},
           "target_thick_median": round(target, 4),
           "thin_median_before": round(float(np.median([th[c] for c in thin])), 4),
           "n_cases": len(cases), "n_thin": len(thin), "thin_cases": thin,
           "per_case_thickness": {c: round(th[c], 4) for c in cases},
           "date_window": window, "source_cache": str(CACHE_DIR),
           "code_sha256": {"topocnv/data/harmonize.py": sha(REPO / "topocnv" / "data" / "harmonize.py"),
                           "tools/v4_harmonize_labels.py": sha(Path(__file__).resolve())}}
    CFG.parent.mkdir(parents=True, exist_ok=True)
    CFG.write_text(json.dumps(cfg, indent=1))
    print(f"thin {len(thin)}/{len(cases)}  thin median {cfg['thin_median_before']}  target {target:.3f}")
    for g, m in zip(T_GRID, med):
        print(f"  t {g:.2f}  harmonised thin median {m:.3f}" + ("   <- chosen" if g == t else ""))
    print(f"wrote {CFG}")


# ---------------------------------------------------------------- build
def _build(a):
    c, t, is_thin = a
    lab = load_label(c, mmap=False)
    out = harmonise(lab, t) if is_thin else lab
    np.save(OUT / f"{c}_label.npy", out)
    img = OUT / f"{c}_image.npy"
    if not img.exists():
        img.symlink_to(CACHE_DIR / f"{c}_image.npy")
    added = mask_Y(out) & ~mask_Y(lab)
    return c, {"thin": is_thin, "n_added": int(added.sum()),
               "n_added_by_class": {str(j): int((added & (out == j)).sum()) for j in (1, 2, 3, 4)},
               "sha256_label": arr_sha(out), "sha256_source_label": arr_sha(lab)}


def build(args):
    cfg = load_cfg()
    done = OUT / "DONE"
    if done.exists() and not args.force:
        raise SystemExit(f"{OUT} already built (write-once); pass --force to override")
    OUT.mkdir(parents=True, exist_ok=True)
    thin = set(cfg["thin_cases"])
    jobs = [(c, cfg["t"], c in thin) for c in all_cases()]
    with Pool(WORKERS) as p:
        man = dict(p.map(_build, jobs))
    meta = {"amendment": AMENDMENT, "version": VERSION, "config": str(CFG.relative_to(REPO)),
            "config_sha256": sha(CFG), "t": cfg["t"], "built": datetime.datetime.now().isoformat(timespec="seconds"),
            "cases": man}
    (OUT / "manifest.json").write_text(json.dumps(meta, indent=1))
    done.write_text(meta["built"] + "\n")
    n_add = sum(m["n_added"] for m in man.values())
    print(f"built {len(man)} labels in {OUT} ({len(thin)} harmonised, {n_add} voxels added)")


# ---------------------------------------------------------------- qa
def _geom(lab):
    y = mask_Y(lab)
    sk = skeleton(y)
    area = y.sum(axis=(1, 2))
    from scipy import ndimage as ndi
    return {"thickness": float(y.sum() / max(1, sk.sum())), "skeleton": int(sk.sum()), "Y": int(y.sum()),
            "rl_z": run_length(y, 0), "rl_lr": run_length(y, 1), "rl_ap": run_length(y, 2),
            "n_slices": int((area > 0).sum()), "area_med": float(np.median(area[area > 0])),
            "components": int(ndi.label(y, np.ones((3, 3, 3)))[1]),
            **{f"frac_c{j}": float((lab == j).sum() / max(1, y.sum())) for j in (1, 2, 3, 4)}}


def _qa(a):
    c, is_thin = a
    from scipy import ndimage as ndi
    orig = load_label(c, mmap=False)
    harm = np.load(OUT / f"{c}_label.npy")
    yo, yh = mask_Y(orig), mask_Y(harm)
    added = yh & ~yo
    changed = harm != orig
    inv = {"same_shape": harm.shape == orig.shape,
           "unchanged_if_not_thin": bool(is_thin or not changed.any()),
           "only_added_voxels_changed": bool(not (changed & ~added).any()),
           "no_y_removed": bool(not (yo & ~yh).any()),
           "class5_kept": bool((harm[orig == 5] == 5).all()),
           "growth_in_plane_ring": bool(not (added & ~ndi.binary_dilation(yo, PLANE4)).any()),
           "added_classes_1to4": bool(np.isin(harm[added], (1, 2, 3, 4)).all())}
    go, gh = _geom(orig), _geom(harm)
    row = {"case": c, "thin": is_thin, "n_added": int(added.sum()), **inv,
           **{f"{k}_orig": v for k, v in go.items()}, **{f"{k}_harm": v for k, v in gh.items()}}
    row["z_extent_same"] = go["n_slices"] == gh["n_slices"]
    row["components_not_increased"] = gh["components"] <= go["components"]
    return row


def _montage(c, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    orig = load_label(c, mmap=False)
    harm = np.load(OUT / f"{c}_label.npy")
    img = np.asarray(load_image(c))
    yo, yh = mask_Y(orig), mask_Y(harm)
    added = yh & ~yo
    zs = np.argsort(added.sum(axis=(1, 2)))[::-1][:3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, z in zip(axes, sorted(zs)):
        r, q = np.nonzero(yh[z])
        r0, r1 = max(0, r.min() - 25), min(yh.shape[1], r.max() + 26)
        q0, q1 = max(0, q.min() - 25), min(yh.shape[2], q.max() + 26)
        ax.imshow(img[z, r0:r1, q0:q1].T, cmap="gray", origin="lower", interpolation="nearest")
        ov = np.zeros((r1 - r0, q1 - q0, 4))
        ov[yo[z, r0:r1, q0:q1]] = (0.1, 0.9, 0.2, 0.55)
        ov[added[z, r0:r1, q0:q1]] = (1.0, 0.15, 0.1, 0.75)
        ax.imshow(ov.transpose(1, 0, 2), origin="lower", interpolation="nearest")
        ax.set_title(f"z={z}: original Y {int(yo[z].sum())} vox, added {int(added[z].sum())}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{c} ({VERSION}): green = original label, red = added by harmonisation (axis 1 horizontal)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def qa(args):
    cfg = load_cfg()
    if not (OUT / "DONE").exists():
        raise SystemExit("build first")
    QA.mkdir(parents=True, exist_ok=True)
    thin = set(cfg["thin_cases"])
    with Pool(WORKERS) as p:
        rows = p.map(_qa, [(c, c in thin) for c in all_cases()])
    with open(QA / "qa_per_case.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = ["thickness", "skeleton", "rl_z", "rl_lr", "rl_ap", "area_med", "n_slices", "components",
            "frac_c1", "frac_c2", "frac_c3", "frac_c4"]
    groups = {"thin_orig": [(r, "orig") for r in rows if r["thin"]],
              "thin_harm": [(r, "harm") for r in rows if r["thin"]],
              "thick": [(r, "orig") for r in rows if not r["thin"]]}
    summ = {g: {k: float(np.median([r[f"{k}_{s}"] for r, s in v])) for k in keys} | {"n": len(v)}
            for g, v in groups.items()}
    from scipy.stats import mannwhitneyu
    th_h = [r["thickness_harm"] for r in rows if r["thin"]]
    th_o = [r["thickness_orig"] for r in rows if r["thin"]]
    th_k = [r["thickness_orig"] for r in rows if not r["thin"]]
    inv_keys = ["same_shape", "unchanged_if_not_thin", "only_added_voxels_changed", "no_y_removed", "class5_kept",
                "growth_in_plane_ring", "added_classes_1to4", "z_extent_same", "components_not_increased"]
    fails = {k: [r["case"] for r in rows if not r[k]] for k in inv_keys}
    rel = abs(summ["thin_harm"]["thickness"] - summ["thick"]["thickness"]) / summ["thick"]["thickness"]
    skel_rel = [abs(r["skeleton_harm"] - r["skeleton_orig"]) / max(1, r["skeleton_orig"]) for r in rows if r["thin"]]
    gates = {"invariants_all_cases": all(not v for v in fails.values()),
             "thin_harm_median_thickness_within_10pct_of_thick": rel <= 0.10}
    out = {"version": VERSION, "t": cfg["t"], "groups": summ, "invariant_failures": fails,
           "mannwhitney_p_thickness": {"thin_orig_vs_thick": float(mannwhitneyu(th_o, th_k).pvalue),
                                       "thin_harm_vs_thick": float(mannwhitneyu(th_h, th_k).pvalue)},
           "thin_harm_vs_thick_rel_median_diff": rel,
           "thin_skeleton_rel_change": {"median": float(np.median(skel_rel)), "max": float(np.max(skel_rel))},
           "gates": gates, "pass": all(gates.values())}
    rng = np.random.default_rng(20261027)
    ranked = sorted((r for r in rows if r["thin"]), key=lambda r: -r["n_added"] / max(1, r["Y_orig"]))
    pick = [r["case"] for r in ranked[:4]]
    rest = [r["case"] for r in ranked[4:]]
    pick += list(rng.choice(rest, size=min(4, len(rest)), replace=False))
    for c in pick:
        _montage(c, QA / f"montage_{c}.png")
    out["montages"] = pick
    (QA / "qa_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("groups", "mannwhitney_p_thickness", "thin_skeleton_rel_change",
                                          "invariant_failures", "gates", "pass")}, indent=1))


# ---------------------------------------------------------------- sanity (dev only)
def _sanity(a):
    import nrrd
    k, c = a
    p = nrrd.read(str(RESULTS / "v4_dev_preds" / "r0_dev" / f"fold{k}" / f"{c}_stage2_label.nrrd"))[0] > 0
    yo = mask_Y(load_label(c))
    yh = mask_Y(np.load(OUT / f"{c}_label.npy", mmap_mode="r"))
    d = lambda y: float(2 * (p & y).sum() / max(1, p.sum() + y.sum()))  # noqa: E731
    return {"fold": k, "case": c, "dice_orig": d(yo), "dice_harm": d(yh),
            "p_over_y_orig": float(p.sum() / yo.sum()), "p_over_y_harm": float(p.sum() / yh.sum())}


def sanity(args):
    cfg = load_cfg()
    thin = set(cfg["thin_cases"])
    jobs = [(k, c) for k in range(5) for c in internal_split(k)[1]]
    with Pool(WORKERS) as p:
        rows = p.map(_sanity, jobs)
    out = {"note": "r0_dev (trained on original labels, T_k - V_k) on V_k; descriptive check run after t was frozen"}
    for g, sel in (("thin", lambda r: r["case"] in thin), ("thick", lambda r: r["case"] not in thin),
                   ("all", lambda r: True)):
        rs = [r for r in rows if sel(r)]
        out[g] = {"n": len(rs), "dice_orig": float(np.mean([r["dice_orig"] for r in rs])),
                  "dice_harm": float(np.mean([r["dice_harm"] for r in rs])),
                  "p_over_y_orig_median": float(np.median([r["p_over_y_orig"] for r in rs])),
                  "p_over_y_harm_median": float(np.median([r["p_over_y_harm"] for r in rs]))}
    QA.mkdir(parents=True, exist_ok=True)
    (QA / "sanity_r0dev.json").write_text(json.dumps(out | {"rows": rows}, indent=1))
    print(json.dumps({g: out[g] for g in ("thin", "thick", "all")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["calibrate", "build", "qa", "sanity"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"calibrate": calibrate, "build": build, "qa": qa, "sanity": sanity}[args.what](args)


if __name__ == "__main__":
    main()
