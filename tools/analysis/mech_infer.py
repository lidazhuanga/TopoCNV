#!/usr/bin/env python3
"""Memory reliance and L2 evidence erasure for P0 Stage-2 checkpoints (inference only).

For each (arm, fold) the validation cases are run through the same Gaussian
sliding window as the trainer (patch 40x192x192, step 0.5), under several
conditions:

  orig      unmodified input
  nomem     memory branch removed at inference: f = f_local at every memory
            level (identical to the g = 1 path); Full and A_gate only
  E1        L2 erasure: CT inside E = H ∩ {d(Y) <= 10 mm} replaced by the case's
            soft-tissue intensity plus matched noise; cascade prior unchanged
  E2        E1 and the cascade prior channels set to 0 inside E
  E1_nomem, E2_nomem   erasure with the memory branch removed

Only windows that overlap E differ between orig and E1/E2 (each window is an
independent forward pass), so erasure conditions recompute those windows and
reuse the reference condition elsewhere; the result is identical to a full pass.

A wrapper around DynUNetMLA3D._gate_fuse (set on the instance; the class is not
modified) records, per memory level and token: the gate value, the memory share
||(1-g) f_mem|| / (||g f_loc|| + ||(1-g) f_mem||), the fused-feature shift
||f - f_loc|| / ||f_loc||, the proposed memory change ||f_mem - f_loc|| / ||f_loc||,
and the retrieval mass on the CN V prototype block.

    python tools/analysis/mech_infer.py --runs Full:0 A_gate:0 A_mem:0 --save_maps 4
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def pick_gpu(min_free_mb, wait_s):
    """Index of the GPU with most free memory once it has >= min_free_mb (polls)."""
    while True:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        free = sorted(((int(f), int(i)) for i, f in
                       (l.split(",") for l in q.strip().splitlines())), reverse=True)
        if free and free[0][0] >= min_free_mb:
            return free[0][1], free[0][0]
        print(f"[gpu] no card with >= {min_free_mb} MB free (best {free[0]}); waiting", flush=True)
        time.sleep(wait_s)


ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="ARM:FOLD, e.g. Full:0")
ap.add_argument("--out", default=None)
ap.add_argument("--cases", nargs="*", help="subset of validation cases")
ap.add_argument("--save_maps", type=int, default=4, help="save ROI maps for the first N cases")
ap.add_argument("--erase_mm", type=float, default=10.0)
ap.add_argument("--min_free_mb", type=int, default=40000)
ap.add_argument("--gpu", type=int, default=None)
ap.add_argument("--skip_existing", action="store_true")
A = ap.parse_args()

if A.gpu is None:
    A.gpu, _free = pick_gpu(A.min_free_mb, 120)
    print(f"[gpu] using {A.gpu} ({_free} MB free)", flush=True)
os.environ["CUDA_VISIBLE_DEVICES"] = str(A.gpu)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import mech_common as mc  # noqa: E402

T = mc.trainer()
mp = mc.mp
PATCH = tuple(T.PATCH_SIZE)
LEVEL_OF_DIM = {128: 0, 256: 1, 320: 2}
TOKSTATS = ("gate", "share", "shift", "dmem", "pfg")
OUT = Path(A.out or (mc.RESULTS / "analysis_mech"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gauss_weight():
    from scipy.ndimage import gaussian_filter
    tmp = np.zeros(PATCH, np.float32)
    tmp[tuple(s // 2 for s in PATCH)] = 1.0
    gw = gaussian_filter(tmp, sigma=[s / 8.0 for s in PATCH], mode="constant", cval=0.0)
    return (gw / gw.max()).astype(np.float32)


GW = gauss_weight()


def positions(shape):
    steps = []
    for n, p in zip(shape, PATCH):
        if n <= p:
            steps.append([0])
            continue
        st = max(int(p * 0.5), 1)
        pos = list(range(0, n - p + 1, st))
        if pos[-1] + p < n:
            pos.append(n - p)
        steps.append(pos)
    return list(itertools.product(*steps))


class Recorder:
    """Instance-level wrapper of net._gate_fuse: optional memory removal + token stats."""

    def __init__(self, model):
        self.net = model.net
        self.mb = getattr(self.net, "memory_bank", None)
        self.force_local = False
        self.record = False
        self.stash = {}
        if self.mb is not None:
            self._orig = self.net._gate_fuse
            self.net._gate_fuse = self.fuse
            self.K = self.mb.num_slots_per_class

    def fuse(self, t_loc, t_mem, grid):
        f = t_loc if self.force_local else self._orig(t_loc, t_mem, grid)
        if self.record:
            lv = LEVEL_OF_DIM[t_loc.shape[-1]]
            gate = self.net._gate_src
            if gate is None:
                g = torch.zeros_like(t_loc[..., :1])
            elif isinstance(gate, float):
                g = torch.full_like(t_loc[..., :1], gate)
            else:
                g = F.adaptive_avg_pool3d(gate.float(), grid).flatten(2).transpose(1, 2)
            g = g.float()
            tl, tm, ff = t_loc.float(), t_mem.float(), f.float()
            nl, nm = tl.norm(dim=-1, keepdim=True), tm.norm(dim=-1, keepdim=True)
            share = (1 - g) * nm / (g * nl + (1 - g) * nm + 1e-8)
            shift = (ff - tl).norm(dim=-1, keepdim=True) / (nl + 1e-8)
            dmem = (tm - tl).norm(dim=-1, keepdim=True) / (nl + 1e-8)
            Asg = self.mb.compute_assignments(t_loc, lv)           # [B,N,P]; block 0 = bg
            pfg = Asg[..., self.K:].sum(-1, keepdim=True).float()
            st = torch.cat([g, share, shift, dmem, pfg], -1)        # [B,N,5]
            self.stash[lv] = st[0].transpose(0, 1).reshape(5, *grid)
        return f


TIMES = {"fwd": 0.0, "n": 0}


@torch.no_grad()
def forward_patch(model, rec, x, z, y, xx, vol_shape, force_local, record):
    t_start = time.time()
    try:
        return _forward_patch(model, rec, x, z, y, xx, vol_shape, force_local, record)
    finally:
        TIMES["fwd"] += time.time() - t_start
        TIMES["n"] += 1


def _forward_patch(model, rec, x, z, y, xx, vol_shape, force_local, record):
    """One window; returns GPU tensors p [pd,ph,pw], p_mem or None, token maps [15,...] or None."""
    rec.force_local, rec.record, rec.stash = force_local, record, {}
    t = torch.from_numpy(np.ascontiguousarray(x[None])).float().to(device, non_blocking=True)
    meta = torch.tensor([[z, y, xx, *vol_shape, 0., 0., 0.]], dtype=torch.float32, device=device)
    out = model(t, None, (meta[:, 0:3], meta[:, 3:6], meta[:, 6:9]))
    p = out[0][0, 0].float()
    pm = torch.sigmoid(out[3][0, 0].float()) if out[3] is not None else None
    tok = None
    if record and rec.stash:
        padded = model.net._patch_padded
        maps = []
        for lv in (0, 1, 2):
            m = F.interpolate(rec.stash[lv][None], size=padded, mode="nearest")[0]
            maps.append(m[:, :PATCH[0], :PATCH[1], :PATCH[2]])
        tok = torch.cat(maps, 0)                                     # [15, pd, ph, pw]
    return p, pm, tok


def run_case(model, rec, arm, cid, save_maps):
    img, lbl, prior, hu = mc.load_case(cid)
    R = mc.regions(lbl, hu)
    g_full = mc.evidence_map(img, prior)
    E = R["H"] & (R["dist"] <= A.erase_mm)
    near = (R["dist"] <= A.erase_mm) & ~R["H"] & (hu >= 0) & (hu <= 100)
    mu, sd = float(img[near].mean()), float(img[near].std())
    rng = np.random.RandomState(int(cid) % (2 ** 31))
    img_e = img.copy()
    img_e[E] = np.clip(mu + sd * rng.randn(int(E.sum())), 0, 1).astype(np.float32)
    prior_e = prior.copy()
    prior_e[:, E] = 0.0
    X = {"orig": np.concatenate([img[None], prior], 0),
         "E1": np.concatenate([img_e[None], prior], 0),
         "E2": np.concatenate([img_e[None], prior_e], 0)}
    vol = img.shape
    if any(v < p for v, p in zip(vol, PATCH)):
        raise RuntimeError(f"{cid}: volume {vol} smaller than patch")

    has_mem = rec.mb is not None
    # name -> (input key, force_local, base condition for windows outside E or None)
    conds = {"orig": ("orig", False, None), "E1": ("E1", False, "orig"), "E2": ("E2", False, "orig")}
    if has_mem:
        conds.update({"nomem": ("orig", True, None), "E1_nomem": ("E1", True, "nomem"),
                      "E2_nomem": ("E2", True, "nomem")})
    tok_conds = {"orig", "E1", "E2"} if has_mem else set()
    roi = mc.bbox(R["roi20"], 0)
    r0 = np.array([s.start for s in roi])
    r1 = np.array([s.stop for s in roi])

    z_ = lambda *shape: torch.zeros(shape, dtype=torch.float32, device=device)  # noqa: E731
    P = {c: z_(*vol) for c in conds}
    PM = {c: z_(*vol) for c in tok_conds}
    TK = {c: z_(15, *(r1 - r0)) for c in tok_conds}
    W = z_(*vol)
    gw = torch.from_numpy(GW).to(device)
    pos = positions(vol)
    n_e = 0
    t0 = time.time()
    with torch.no_grad():
        for z, y, x in pos:
            sl = (slice(z, z + PATCH[0]), slice(y, y + PATCH[1]), slice(x, x + PATCH[2]))
            in_e = bool(E[sl].any())
            n_e += in_e
            W[sl] += gw
            lo = np.maximum(r0, (z, y, x))
            hi = np.minimum(r1, np.array((z, y, x)) + PATCH)
            hit_roi = bool(np.all(hi > lo))
            a = lo - np.array((z, y, x))
            b = hi - np.array((z, y, x))
            src = (slice(a[0], b[0]), slice(a[1], b[1]), slice(a[2], b[2]))
            dst = tuple(slice(int(l_ - o), int(h_ - o)) for l_, h_, o in zip(lo, hi, r0))
            res = {}
            for c in ("orig", "nomem", "E1", "E2", "E1_nomem", "E2_nomem"):
                if c not in conds:
                    continue
                key, force_local, base = conds[c]
                if base is not None and not in_e:
                    res[c] = res[base]
                else:
                    res[c] = forward_patch(model, rec, X[key][(slice(None),) + sl], z, y, x, vol,
                                           force_local, record=(c in tok_conds and hit_roi))
                p, pm, tok = res[c]
                P[c][sl] += p * gw
                if c in PM and pm is not None:
                    PM[c][sl] += pm * gw
                if c in TK and tok is not None and hit_roi:
                    TK[c][(slice(None),) + dst] += tok[(slice(None),) + src] * gw[src]
        W = W.clamp_min(1e-8)
        P = {c: (v / W).cpu().numpy() for c, v in P.items()}
        PM = {c: (v / W).cpu().numpy() for c, v in PM.items()}
        Wr = W[roi]
        TK = {c: (v / Wr).cpu().numpy() for c, v in TK.items()}
        del W, Wr
        torch.cuda.empty_cache()
    t_inf = time.time() - t0
    print(f"    {cid}: {TIMES['n']} forwards, {TIMES['fwd']:.0f}s in forward, {t_inf:.0f}s total",
          flush=True)
    TIMES.update(fwd=0.0, n=0)

    # ---------------- metrics ----------------
    Y, L, YH = R["Y"], R["L"], R["YH"]
    YE = Y & E
    G = {"Yglow": Y & (g_full < 0.5), "Yghigh": Y & (g_full >= 0.5)}
    REG = {"YH": YH, "L": L, "YE": YE, "ring5": R["ring5"], "bg20": R["bg20"], **G}

    def seg_metrics(prob):
        pred = prob > 0.5
        crop = mc.bbox(pred | Y, 4) if pred.any() else mc.bbox(Y, 4)
        return dict(Dice_Y=mp._dice(pred, Y), HD95_Y=mp._hd95(pred[crop], Y[crop], mc.SPACING),
                    Dice_L=mp._dice(pred & L, L), Recall_L=mp._recall(pred, L),
                    # protocol Dice_L = 2R/(1+R); this one also counts false positives outside H
                    Dice_LnotH=mp._dice(pred & ~R["H"], L),
                    Recall_YH=mp._recall(pred, YH), Recall_YE=mp._recall(pred, YE),
                    Precision_Y=float((pred & Y).sum() / max(pred.sum(), 1)),
                    n_pred=int(pred.sum()))

    out = {"arm": arm, "case": cid, "n_Y": int(Y.sum()), "n_L": int(L.sum()), "n_YH": int(YH.sum()),
           "n_YE": int(YE.sum()), "n_E": int(E.sum()), "soft_mu": mu, "soft_sd": sd,
           "n_windows": len(pos), "n_windows_E": n_e, "sec": round(t_inf, 1),
           "region_sizes": {k: int(v.sum()) for k, v in REG.items()},
           "g_mean": {k: float(g_full[v].mean()) if v.any() else None for k, v in REG.items()},
           "metrics": {}, "p_mean": {}, "delta": {}, "tok": {}}
    for c in P:
        out["metrics"][c] = seg_metrics(P[c])
        out["p_mean"][c] = {k: float(P[c][v].mean()) if v.any() else None for k, v in REG.items()}
    for c in PM:
        out["metrics"][c + "_memreadout"] = seg_metrics(PM[c])
    pairs = [("orig", "nomem"), ("E1", "E1_nomem"), ("E2", "E2_nomem"), ("orig", "E1"),
             ("orig", "E2"), ("nomem", "E1_nomem"), ("nomem", "E2_nomem")]
    for a_, b_ in pairs:
        if a_ in P and b_ in P:
            d = P[a_] - P[b_]
            fl = (P[a_] > 0.5) != (P[b_] > 0.5)
            out["delta"][f"{a_}-{b_}"] = {
                k: dict(abs=float(np.abs(d[v]).mean()), signed=float(d[v].mean()),
                        flip=float(fl[v].mean())) if v.any() else None for k, v in REG.items()}
    for c in TK:
        out["tok"][c] = {}
        for k, v in REG.items():
            vv = v[roi]
            if not vv.any():
                continue
            out["tok"][c][k] = {f"{s}{lv}": float(TK[c][lv * 5 + j][vv].mean())
                                for lv in range(3) for j, s in enumerate(TOKSTATS)}

    if save_maps:
        f16 = lambda a: a[roi].astype(np.float16)  # noqa: E731
        maps = dict(ct=f16(img), ct_e=f16(img_e), g=f16(g_full), Y=Y[roi], H=R["H"][roi], L=L[roi],
                    E=E[roi], c4=(lbl == 4)[roi], roi_lo=r0, roi_hi=r1)
        for c in P:
            maps[f"p_{c}"] = f16(P[c])
        for c in PM:
            maps[f"pm_{c}"] = f16(PM[c])
        for c in TK:
            maps[f"tok_{c}"] = TK[c].astype(np.float16)
        np.savez_compressed(OUT / "maps" / f"{arm}_{cid}.npz", **maps)
    return out


def main():
    (OUT / "maps").mkdir(parents=True, exist_ok=True)
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    for run in A.runs:
        arm, fold = run.split(":")
        fold = int(fold)
        if not mc.fold_completed(arm, fold):
            print(f"[skip] {arm} fold{fold}: no COMPLETED marker", flush=True)
            continue
        model, best_ep = mc.build_model(arm, fold, device)
        rec = Recorder(model)
        cases = A.cases or mc.fold_cases(fold)
        print(f"[run] {arm} fold{fold} best_epoch={best_ep} cases={len(cases)} gpu={A.gpu}", flush=True)
        for i, cid in enumerate(cases):
            dst = OUT / "cases" / f"{arm}_f{fold}_{cid}.json"
            if A.skip_existing and dst.exists():
                continue
            r = run_case(model, rec, arm, cid, save_maps=i < A.save_maps)
            r.update(fold=fold, best_epoch=best_ep)
            dst.write_text(json.dumps(r))
            m = r["metrics"]
            msg = " ".join(f"{c}:{m[c]['Dice_Y']:.3f}/{m[c]['Recall_L']:.3f}" for c in
                           ("orig", "nomem", "E1", "E2") if c in m)
            print(f"  [{i + 1}/{len(cases)}] {cid} {r['sec']:.0f}s winE={r['n_windows_E']}/"
                  f"{r['n_windows']} DiceY/RecL {msg}", flush=True)
        del model, rec
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
