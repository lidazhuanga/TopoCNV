#!/usr/bin/env python3
"""Build ROI records and prior channels for every case of one outer fold (CPU).

Stage-1 input per case, per variant (nested selection, V4_PLAN §5 as revised 2026-10-24):
  --variant dev     Stage-1 runs under results/v4_stage1/fold{k}_dev (inner folds over T_k - V_k):
                    inner-OOF prediction for T_k - V_k, mean of the 3 dev inner models for V_k
                    (V_k is held out of every dev Stage-1 model, exactly like report-fold cases).
  --variant report  Stage-1 runs under results/v4_stage1/fold{k} (inner folds over T_k):
                    inner-OOF prediction for T_k, mean of the 3 inner models for report fold k.

The atlas and its ICP template come only from the current training set (main session, 2026-10-23):
  dev     atlas set = T_k - V_k   (selection: Stage 2 trains on T_k - V_k, scored on V_k)
  report  atlas set = T_k         (report phase: Stage 2 trains on T_k, scored on fold k)
Atlas-set cases get the leave-one-out atlas. The ROI is the atlas support (> 0.02) mapped into the
case by ICP on its predicted class-4 points, per side, + m mm (protocol §4 as revised 2026-10-23);
the landmark boxes are kept in the record for reference.

Outputs results/v4_priors/fold{k}_{variant}/{case}_roi.json and {case}_prior.npy
(uint8 [3, *prior_box]: a, b, c; prior_box = union of the m = 35 mm side boxes), plus atlas_meta.json.

  python tools/v4_build_priors.py --fold 0 --variant dev --workers 24
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topocnv.data.utils import (PRIOR_BOX_MARGIN, PRIORS_DIR, STAGE1_DIR, load_image,  # noqa: E402
                                load_label)
from topocnv.priors import (Atlas, atlas_roi_record, bilateral_c4_distance_mm,  # noqa: E402
                               build_case_priors, choose_template, prior_box_from_rec,
                               side_c4_centroids)
from topocnv.priors.roi import compute_case_roi, head_midline, landmark_mask_from_probs  # noqa: E402
from topocnv.data.v4_splits import N_INNER, all_cases, internal_split, stage1_source, train_cases  # noqa: E402

ATLAS = None
FOLD = None
VARIANT = None
ATLAS_SET = None
S1_ROOT = None
OUT = None
QC = None          # D1-AM1 gate: {"L", "U", "amendment"} or None (no gate)


def load_s1(case_id):
    tier = "dev" if VARIANT == "dev" else "report"
    src, j = stage1_source(FOLD, case_id, tier=tier)
    if src == "inner":
        d = np.load(S1_ROOT / "pred" / f"{case_id}_s1prob.npz")
        return d["p1"] / np.float32(255), d["p4"] / np.float32(255), f"inner{j}"
    holdout_dir = "val_pred" if tier == "dev" else "report_pred"
    p1 = p4 = None
    for jj in range(N_INNER):
        d = np.load(S1_ROOT / f"inner{jj}" / holdout_dir / f"{case_id}_s1prob.npz")
        a, b = d["p1"].astype(np.float32), d["p4"].astype(np.float32)
        p1, p4 = (a, b) if p1 is None else (p1 + a, p4 + b)
    return p1 / (255.0 * N_INNER), p4 / (255.0 * N_INNER), "ensemble"


def stats_one(case_id):
    return case_id, side_c4_centroids(load_label(case_id), load_image(case_id))


def warp_one(case_id):
    return case_id, ATLAS.warp_case(np.asarray(load_label(case_id)), load_image(case_id))


def prior_one(case_id):
    t0 = time.time()
    image = load_image(case_id)
    p1, p4, src = load_s1(case_id)
    lm, is4 = landmark_mask_from_probs(p1, p4)
    rec, _, c4_clean = compute_case_roi(lm, is4, image)
    exclude = case_id if case_id in ATLAS_SET else None
    rec, transforms = atlas_roi_record(rec, ATLAS, c4_clean, image, exclude=exclude)
    # D1-AM1 QC gate (docs/V4_PLAN.md §6.1): GT-free anatomical sanity check on the bilateral
    # clean class-4 centroids; structural failures or D outside the fold's frozen bounds ->
    # full-volume case: empty side boxes, zeroed priors, no ROI zeroing on export.
    d_mm = bilateral_c4_distance_mm(c4_clean, head_midline(image, None))
    # per-side ICP provenance: how == 'side' means own-side ICP (>= MIN_ICP_POINTS); a side that
    # fell back to bilateral ICP or produced no transform is a structural failure (D1-AM1).
    # target_transforms returns (mid, [(T, info), (T, info)]) — unpack before use
    tr_list = (transforms[1] if isinstance(transforms, tuple) else transforms) or []
    tinfo = [t[1] if isinstance(t, (tuple, list)) and len(t) == 2 else {"how": "none"} for t in tr_list]
    hows = [w.get("how") for w in tinfo]
    if QC is not None:
        own_side = len(hows) == 2 and all(h == "side" and t[0] is not None
                                          for h, t in zip(hows, tr_list))
        structural_ok = (own_side and rec["roi_found"]
                         and all(all(v is not None for v in s["boxes"].values()) for s in rec["sides"]))
        reasons = []
        if d_mm is None:
            reasons.append("D_uncomputable")
        if not structural_ok:
            reasons.append("structural:" + "/".join(str(h) for h in hows))
        if d_mm is not None and not (QC["L"] <= d_mm <= QC["U"]):
            reasons.append("D_out_of_bounds")
        fallback = bool(reasons)
        rec["qc"] = {"amendment": QC["amendment"], "fallback": fallback, "D_mm": d_mm,
                     "L": QC["L"], "U": QC["U"], "sides_how": hows,
                     "reason": "+".join(reasons) if fallback else None}
    else:
        fallback = False
        rec["qc"] = {"amendment": "D1-AM1", "fallback": False, "D_mm": d_mm, "applied": False,
                     "sides_how": hows}
    if fallback:
        for s in rec["sides"]:
            s["boxes"] = {m: None for m in s["boxes"]}
        pbox, pri = [[0, 1], [0, 1], [0, 1]], np.zeros((3, 1, 1, 1), np.uint8)
        info = tinfo          # keep the ICP provenance of the failed mapping (P2-4)
    else:
        pbox = prior_box_from_rec(rec, PRIOR_BOX_MARGIN)
        if pbox is None:
            pbox = [[0, 1], [0, 1], [0, 1]]
            pri, info = np.zeros((3, 1, 1, 1), np.uint8), []
        else:
            pri, info = build_case_priors(p4, c4_clean, image, pbox, ATLAS, exclude=exclude,
                                          transforms=transforms)
    rec.update(prior_box=pbox, stage1_source=src, atlas_warp=info, fold=FOLD,
               atlas_loo=exclude is not None, n_c4_clean=int(c4_clean.sum()))
    np.save(OUT / f"{case_id}_prior.npy", pri)
    (OUT / f"{case_id}_roi.json").write_text(json.dumps(rec))
    return (case_id, rec["roi_found"], [s["source"] for s in rec["sides"]], time.time() - t0,
            bool(rec.get("qc", {}).get("fallback")))


def main():
    global ATLAS, FOLD, VARIANT, ATLAS_SET, S1_ROOT, OUT, QC
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--variant", choices=["dev", "report"], required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--cases", default="",
                    help="comma-separated subset (debugging); default: T_k for dev, all 154 for report")
    ap.add_argument("--stage1_root", default="", help="override Stage-1 dir (smoke tests)")
    ap.add_argument("--qc", default="", help="configs/v4_qc/fold{k}.json (D1-AM1 gate; omit = no gate)")
    args = ap.parse_args()
    FOLD, VARIANT = args.fold, args.variant
    QC = None
    if args.qc:
        q = Path(args.qc)
        if not q.is_absolute():
            q = Path(__file__).resolve().parents[1] / q
        QC = json.loads(q.read_text())
        assert {"L", "U", "amendment", "fold"} <= set(QC), "qc json needs L/U/amendment/fold"
        import math
        assert QC["fold"] == FOLD, f"qc config is for fold {QC['fold']}, building fold {FOLD}"
        assert QC["amendment"] == "D1-AM1", f"unknown amendment {QC['amendment']!r}"
        assert math.isfinite(QC["L"]) and math.isfinite(QC["U"]) and QC["L"] < QC["U"], "bad bounds"
    ATLAS_SET = set(internal_split(FOLD)[0] if VARIANT == "dev" else train_cases(FOLD))
    s1_subdir = f"fold{FOLD}" if VARIANT == "report" else f"fold{FOLD}_dev"
    S1_ROOT = Path(args.stage1_root) if args.stage1_root else STAGE1_DIR / s1_subdir
    if not args.stage1_root:
        missing = [j for j in range(N_INNER) if not (S1_ROOT / f"inner{j}" / "COMPLETED").exists()]
        if missing:
            raise SystemExit(f"Stage-1 inner runs not finished for fold {FOLD} ({s1_subdir}): {missing}")
    OUT = out = PRIORS_DIR / f"fold{FOLD}_{VARIANT}"
    out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")
    t0 = time.time()

    tr = sorted(ATLAS_SET)
    with ctx.Pool(args.workers) as p:
        stats = dict(p.map(stats_one, tr))
    template = choose_template(stats)
    ATLAS = Atlas(load_label(template), load_image(template))
    with ctx.Pool(args.workers) as p:
        for cid, ws in p.imap_unordered(warp_one, tr):
            ATLAS.add_warped(cid, ws)
    print(f"atlas: template {template}, {ATLAS.n} cases ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # dev covers T_k only (T_k - V_k inner-OOF + V_k ensemble); the report fold k has no
    # dev-tier Stage-1 predictions and is not part of the dev dataset (V4_PLAN §5-§6).
    default_cases = train_cases(FOLD) if VARIANT == "dev" else all_cases()
    cases = args.cases.split(",") if args.cases else default_cases
    with ctx.Pool(args.workers) as p:
        res = p.map(prior_one, cases)
    no_roi = [c for c, found, _, _, _ in res if not found]
    mirrored = [c for c, _, src, _, _ in res if "mirror" in src]
    fallback = [c for c, _, _, _, fb in res if fb]
    meta = {"fold": FOLD, "variant": VARIANT, "stage1_dir": s1_subdir, "template": template,
            "n_atlas": ATLAS.n, "n_cases": len(res),
            "no_roi": no_roi, "mirrored_side": mirrored,
            "qc_amendment": (QC or {}).get("amendment"), "qc_bounds": (None if QC is None
                                                                else [QC["L"], QC["U"]]),
            "qc_fallback": fallback, "n_qc_fallback": len(fallback),
            "atlas_boxes": ATLAS.boxes, "minutes": (time.time() - t0) / 60}
    (out / "atlas_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "atlas_boxes"}), flush=True)


if __name__ == "__main__":
    main()
