#!/usr/bin/env python3
"""Evidence map g(x) vs. the protocol high-evidence region H, all cases (CPU only).

For every case: the HU that the normalized-CT bone threshold 0.97 corresponds to,
voxel agreement between the two bone definitions near the nerve, and the
distribution of g inside Y∩H, L = Y\\H and the background shells around Y.
g uses the Stage-1 out-of-fold cascade prior, exactly as at Stage-2 inference.

    python tools/analysis/gate_stats.py --out results/analysis_gate --workers 8
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import mech_common as mc

BINS = np.linspace(0, 1, 21)
REGS = ("YH", "L", "ring5", "bg20")


def one_case(cid):
    img, lbl, prior, hu = mc.load_case(cid)
    R = mc.regions(lbl, hu)
    g = mc.evidence_map(img, prior)
    out = {"case": cid, "n_Y": int(R["Y"].sum()), "n_L": int(R["L"].sum()),
           "L_over_Y": float(R["L"].sum() / max(R["Y"].sum(), 1))}

    # Linear map normalized CT -> HU (the cache is a per-case affine map of the
    # resampled CT, clipped to [0,1]); fit on unclipped voxels.
    sel = (img > 0.05) & (img < 0.95)
    sub = np.flatnonzero(sel.ravel())[:: max(1, int(sel.sum()) // 200000)]
    a, b = np.polyfit(img.ravel()[sub], hu.ravel()[sub], 1)
    out.update(hu_at_097=float(a * 0.97 + b), hu_at_1=float(a + b),
               frac_ct_saturated_roi=float((img[R["roi20"]] >= 0.999).mean()))

    roi = R["roi20"]
    bone_hu = (hu >= mc.mp.BONE_HU_LOW) & (hu <= mc.mp.BONE_HU_HIGH)
    bone_g = img > mc.trainer().DEFAULT_BONE_NORM_THR      # sigmoid term > 0.5
    c4_gt = mc.mp.dilate_mm(lbl == 4, mc.SPACING, 3.0)
    c4_pred = prior.max(0) >= 0.5                           # prior term > 0.5
    gh = g >= 0.5

    def agree(A, B, m):
        A, B = A[m], B[m]
        inter = (A & B).sum()
        return dict(dice=float(2 * inter / max(A.sum() + B.sum(), 1)),
                    frac_B_covered=float(inter / max(B.sum(), 1)),
                    frac_A_outside_B=float((A & ~B).sum() / max(A.sum(), 1)))

    out["bone_g_vs_bone_hu_roi"] = agree(bone_g, bone_hu, roi)
    out["c4pred_vs_c4gt_roi"] = agree(c4_pred, c4_gt, roi)
    out["g05_vs_H_roi"] = agree(gh, R["H"], roi)
    # Within Y: how the gate's high/low split matches the protocol's H/L split.
    out["Y_g_high_in_YH"] = float(gh[R["YH"]].mean()) if R["YH"].any() else float("nan")
    out["Y_g_high_in_L"] = float(gh[R["L"]].mean()) if R["L"].any() else float("nan")
    out["Y_frac_g_low"] = float((~gh[R["Y"]]).mean())
    # Which g term makes L voxels "high evidence".
    Lm = R["L"]
    out["L_bone_g"] = float(bone_g[Lm].mean()) if Lm.any() else float("nan")
    out["L_c4pred"] = float(c4_pred[Lm].mean()) if Lm.any() else float("nan")
    out["YH_bone_hu"] = float(bone_hu[R["YH"]].mean()) if R["YH"].any() else float("nan")
    out["YH_c4gt"] = float(c4_gt[R["YH"]].mean()) if R["YH"].any() else float("nan")

    for r in REGS:
        v = g[R[r]]
        if v.size == 0:
            continue
        out[f"g_{r}"] = dict(mean=float(v.mean()), median=float(np.median(v)),
                             p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                             frac_low=float((v < 0.5).mean()),
                             hist=np.histogram(v, BINS)[0].tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(mc.RESULTS / "analysis_gate"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="*")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = a.cases or mc.all_cases()
    with Pool(a.workers) as pool:
        rows = []
        for r in pool.imap_unordered(one_case, cases):
            rows.append(r)
            print(f"{r['case']} L/Y={r['L_over_Y']:.3f} HU@0.97={r['hu_at_097']:.0f} "
                  f"g>=.5 in L={r['Y_g_high_in_L']:.3f} in YH={r['Y_g_high_in_YH']:.3f}", flush=True)
    rows.sort(key=lambda r: r["case"])
    (out / "gate_stats.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {out / 'gate_stats.json'} ({len(rows)} cases)")


if __name__ == "__main__":
    main()
