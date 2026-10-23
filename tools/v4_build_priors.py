#!/usr/bin/env python3
"""Build ROI records and prior channels for every case of one outer fold (CPU).

Needs the three Stage-1 inner runs of fold k (results/v4_stage1/fold{k}/inner{j}/COMPLETED).
Stage-1 input per case: inner-OOF prediction for T_k, mean of the 3 inner models for report fold k.

The atlas and its ICP template come only from the current training set (main session, 2026-10-23):
  --variant dev     atlas set = T_k - V_k   (development: Stage 2 trains on T_k - V_k, scored on V_k)
  --variant report  atlas set = T_k         (report phase: Stage 2 trains on T_k, scored on fold k)
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
from trainer.v4.common import (PRIOR_BOX_MARGIN, PRIORS_DIR, STAGE1_DIR, load_image,  # noqa: E402
                               load_label)
from trainer.v4.priors import (Atlas, atlas_roi_record, build_case_priors, choose_template,  # noqa: E402
                               prior_box_from_rec, side_c4_centroids)
from trainer.v4.roi import compute_case_roi, landmark_mask_from_probs  # noqa: E402
from trainer.v4.splits import N_INNER, all_cases, internal_split, stage1_source, train_cases  # noqa: E402

ATLAS = None
FOLD = None
ATLAS_SET = None
S1_ROOT = None
OUT = None


def load_s1(case_id):
    src, j = stage1_source(FOLD, case_id)
    if src == "inner":
        d = np.load(S1_ROOT / "pred" / f"{case_id}_s1prob.npz")
        return d["p1"] / np.float32(255), d["p4"] / np.float32(255), f"inner{j}"
    p1 = p4 = None
    for jj in range(N_INNER):
        d = np.load(S1_ROOT / f"inner{jj}" / "report_pred" / f"{case_id}_s1prob.npz")
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
    pbox = prior_box_from_rec(rec, PRIOR_BOX_MARGIN)
    if pbox is None:
        pbox = [[0, 1], [0, 1], [0, 1]]
        pri, info = np.zeros((3, 1, 1, 1), np.uint8), []
    else:
        pri, info = build_case_priors(p4, c4_clean, image, pbox, ATLAS, exclude=exclude, transforms=transforms)
    rec.update(prior_box=pbox, stage1_source=src, atlas_warp=info, fold=FOLD,
               atlas_loo=exclude is not None, n_c4_clean=int(c4_clean.sum()))
    np.save(OUT / f"{case_id}_prior.npy", pri)
    (OUT / f"{case_id}_roi.json").write_text(json.dumps(rec))
    return case_id, rec["roi_found"], [s["source"] for s in rec["sides"]], time.time() - t0


def main():
    global ATLAS, FOLD, ATLAS_SET, S1_ROOT, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--variant", choices=["dev", "report"], required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--cases", default="", help="comma-separated subset (debugging); default all 154")
    ap.add_argument("--stage1_root", default="", help="override results/v4_stage1/fold{k} (smoke tests)")
    args = ap.parse_args()
    FOLD = args.fold
    ATLAS_SET = set(internal_split(FOLD)[0] if args.variant == "dev" else train_cases(FOLD))
    S1_ROOT = Path(args.stage1_root) if args.stage1_root else STAGE1_DIR / f"fold{FOLD}"
    if not args.stage1_root:
        missing = [j for j in range(N_INNER) if not (S1_ROOT / f"inner{j}" / "COMPLETED").exists()]
        if missing:
            raise SystemExit(f"Stage-1 inner runs not finished for fold {FOLD}: {missing}")
    OUT = out = PRIORS_DIR / f"fold{FOLD}_{args.variant}"
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

    cases = args.cases.split(",") if args.cases else all_cases()
    with ctx.Pool(args.workers) as p:
        res = p.map(prior_one, cases)
    no_roi = [c for c, found, _, _ in res if not found]
    mirrored = [c for c, _, src, _ in res if "mirror" in src]
    meta = {"fold": FOLD, "variant": args.variant, "template": template, "n_atlas": ATLAS.n, "n_cases": len(res),
            "no_roi": no_roi, "mirrored_side": mirrored, "atlas_boxes": ATLAS.boxes,
            "minutes": (time.time() - t0) / 60}
    (out / "atlas_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "atlas_boxes"}), flush=True)


if __name__ == "__main__":
    main()
