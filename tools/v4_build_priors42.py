#!/usr/bin/env python3
"""V4.2 prior channels a/b/c for one outer fold, dev tier (V4_PLAN §11.4, §11.6.2; D1-AM2). CPU only.

Stage 1 = Dataset541 (stock nnU-Net, 5 classes, harmonised labels), probabilities (5, z, LR, AP) in cache order:
  training cases T_k - V_k  validation/{case}.npz of the dev inner model that held the case out (inner-OOF)
  V_k                       holdout42/{case}.npz of each of the 3 dev inner models -> prior copies {case}_j{j}
Per prior id: argmax of the 5 classes; landmarks = argmax in {1, 4}, c4 = argmax 4 (as V4.0, trainer/v4/roi.py);
per-side ICP of the clean c4 to the atlas (harmonised Y of T_k - V_k, leave-one-out for its own cases); support
= union of the per-side atlas-support boxes (> 0.02) expanded by m_k. a = p1+p2+p3+p4, b = 1 - min(d_c4, 30 mm)
/ 30 mm, c = atlas, all zero outside the support; a D1-AM2 fallback case gets all three channels zero.

D1-AM2 (configs/v4_qc42/fold{k}.json, write-once; a rerun must reproduce it exactly), from T_k - V_k only:
  m_k  = smallest m in {15, 25, 35} whose minimum coverage of the harmonised GT Y by the (not fallen back) atlas
         ROI is >= 0.99 over the training cases; 35 if none (V4_PLAN §6.1 point 1)
  L, U = Q1 - 3 IQR, Q3 + 3 IQR (linear quantiles) of D, the left-right distance of the clean c4 centroids;
         structural failure or D outside [L, U] -> fallback (§6.1 point 2)

Outputs results/v4_priors42/fold{k}_dev/: {id}_prior.npz (uint8 'prior' (3, *bbox), 'bbox'), {id}_roi.json,
stage1_qc.csv / stage1_qc.json (§10.3 point 6, descriptive, dev cases only), DONE (written last: inputs and their
sha256, Stage 1 checkpoints, D1-AM2 config sha256, one content hash per prior file, digest).

  python tools/v4_build_priors42.py --fold 0 --workers 12
"""
import argparse
import csv
import hashlib
import io
import json
import math
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import v4_nnunet as N  # noqa: E402
from trainer.v4.common import (MARGINS_MM, SPACING, box_slices, boxes_mask, expand_box, mask_Y,  # noqa: E402
                               union_box)
from trainer.v4.priors import (DIST_TRUNC_MM, Atlas, atlas_roi_record, bilateral_c4_distance_mm,  # noqa: E402
                               choose_template, dist_prior, side_c4_centroids, to_uint8)
from trainer.v4.roi import compute_case_roi, coverage, head_midline, margin_vox  # noqa: E402
from trainer.v4.splits import N_INNER, inner_folds, internal_split  # noqa: E402

AMENDMENT = "D1-AM2"
MIN_VALID_D = 90          # as D1-AM1 (tools/v4_qc_d1.py)
COV_TARGET = 0.99

CACHE = None              # harmonised cache (labels; images link to the original cache)
ATLAS = None
ATLAS_SET = None
INPUTS = None             # {prior id: (case, npz path, inner j)}


def load_probs(path, case):
    raw = Path(path).read_bytes()
    with np.load(io.BytesIO(raw)) as d:
        p = np.asarray(d["probabilities"], np.float32)
    shape = tuple(load_label(case).shape)
    assert p.shape == (5,) + shape, f"{path}: probabilities {p.shape}, cache grid {shape}"
    assert np.isfinite(p).all(), f"{path}: non-finite probabilities"
    return p, hashlib.sha256(raw).hexdigest()


def load_image(case):
    return np.load(CACHE / f"{case}_image.npy", mmap_mode="r")


def load_label(case):
    return np.load(CACHE / f"{case}_label.npy", mmap_mode="r")


def landmarks(p):
    am = p.argmax(0)
    return am, (am == 1) | (am == 4), am == 4


def p_y(p, sl=(slice(None),) * 3):
    return np.clip(p[(slice(1, 5),) + tuple(sl)].sum(0), 0.0, 1.0)


def dist_on_box(c4, box):
    """Prior b on `box`, = dist_prior on the full volume cropped to the box: the transform runs on the box + 30 mm
    only, since c4 voxels farther away than 30 mm give 0 after truncation anyway."""
    reg = expand_box(box, margin_vox(DIST_TRUNC_MM, SPACING), c4.shape)
    d = dist_prior(np.asarray(c4[box_slices(reg)]))
    return d[tuple(slice(lo - r0, hi - r0) for (lo, hi), (r0, _) in zip(box, reg))]


def structural(rec, transforms):
    """(ok, per-side ICP provenance): both sides own-side ICP with a transform, ROI found, every box present."""
    tr = transforms[1] or []
    info = [t[1] for t in tr]
    hows = [w.get("how") for w in info]
    own = len(hows) == 2 and all(h == "side" and t[0] is not None for h, t in zip(hows, tr))
    ok = own and rec["roi_found"] and all(all(v is not None for v in s["boxes"].values()) for s in rec["sides"])
    return bool(ok), hows, info


def fallback_reasons(d_mm, structural_ok, hows, L, U):
    """D1-AM1 gate (V4_PLAN §6.1 point 2), unchanged in D1-AM2. [] = no fallback."""
    r = []
    if d_mm is None or not math.isfinite(d_mm):
        r.append("D_uncomputable")
    if not structural_ok:
        r.append("structural:" + "/".join(str(h) for h in hows))
    if d_mm is not None and math.isfinite(d_mm) and not (L <= d_mm <= U):
        r.append("D_out_of_bounds")
    return r


def freeze_rule(per_case_cov, per_case_d):
    """D1-AM2 numbers from the training cases: {case: {m: coverage}}, {case: D or None} -> dict."""
    cases = sorted(per_case_cov)
    m_k, min_cov, feasible = MARGINS_MM[-1], min(per_case_cov[c][MARGINS_MM[-1]] for c in cases), False
    for m in MARGINS_MM:
        mn = min(per_case_cov[c][m] for c in cases)
        if mn >= COV_TARGET:
            m_k, min_cov, feasible = m, mn, True
            break
    valid = [per_case_d[c] for c in cases if per_case_d[c] is not None and math.isfinite(per_case_d[c])]
    assert len(valid) >= MIN_VALID_D, f"only {len(valid)} valid D values (need {MIN_VALID_D})"
    q1, q3 = (float(x) for x in np.quantile(np.array(valid), [0.25, 0.75], method="linear"))
    iqr = q3 - q1
    return {"margin": int(m_k), "min_cov_at_margin": float(min_cov), "rule_feasible": feasible,
            "L": q1 - 3 * iqr, "U": q3 + 3 * iqr, "q1": q1, "q3": q3, "n_train": len(cases),
            "n_valid_d": len(valid),
            "d_invalid": [c for c in cases if per_case_d[c] is None or not math.isfinite(per_case_d[c])]}


def s1_metrics(am, gt):
    """Stage 1 quality numbers of one prediction against the harmonised GT (class 5 = background)."""
    gt = np.where(gt == 5, 0, gt)
    out = {}
    for c in (1, 2, 3, 4):
        p, g = am == c, gt == c
        s = int(p.sum()) + int(g.sum())
        out[f"dice_c{c}"] = 2.0 * int((p & g).sum()) / s if s else float("nan")
        if c in (1, 4):
            out[f"volratio_c{c}"] = int(p.sum()) / int(g.sum()) if g.any() else float("nan")
    zp = np.nonzero(((am >= 1) & (am <= 4)).any(axis=(1, 2)))[0]
    zg = np.nonzero(((gt >= 1) & (gt <= 4)).any(axis=(1, 2)))[0]
    ok = zp.size and zg.size
    out["z_lo_err_mm"] = float((zp.min() - zg.min()) * SPACING[0]) if ok else float("nan")
    out["z_hi_err_mm"] = float((zp.max() - zg.max()) * SPACING[0]) if ok else float("nan")
    return out


def stats_one(case):
    return case, side_c4_centroids(load_label(case), load_image(case))


def warp_one(case):
    return case, ATLAS.warp_case(np.asarray(load_label(case)), load_image(case))


def pass1(i):
    """Everything the D1-AM2 freeze and the gate need, plus the Stage 1 quality numbers."""
    case, path, j = INPUTS[i]
    p, sha = load_probs(path, case)
    image = load_image(case)
    am, lm, is4 = landmarks(p)
    rec, _, c4 = compute_case_roi(lm, is4, image)
    exclude = case if case in ATLAS_SET else None
    rec, transforms = atlas_roi_record(rec, ATLAS, c4, image, exclude=exclude)
    d_mm = bilateral_c4_distance_mm(c4, head_midline(image, None))
    ok, hows, info = structural(rec, transforms)
    gt = np.asarray(load_label(case))
    gy = mask_Y(gt)
    cov = {m: float(coverage(rec, m, gy)[0]) for m in MARGINS_MM}
    q = {"id": i, "case": case, "inner": j, "D_mm": d_mm, "structural_ok": ok, "sides_how": "/".join(map(str, hows)),
         **{f"icp_rmse_s{s}": w.get("rmse_mm") for s, w in enumerate(info)},
         **{f"cov_m{m}": cov[m] for m in MARGINS_MM}, **s1_metrics(am, gt), "n_c4_clean": int(c4.sum())}
    return {"id": i, "sha_in": sha, "rec": rec, "transforms": transforms, "D_mm": d_mm, "structural_ok": ok,
            "hows": hows, "info": info, "cov": cov, "qc": q}


def pass2(task):
    """Priors of one id at the frozen margin; zero channels on fallback."""
    i, rec, transforms, reasons, m, out = task
    case, path, _ = INPUTS[i]
    image = load_image(case)
    boxes = [] if reasons else [s["boxes"][str(m)] for s in rec["sides"] if s["boxes"][str(m)] is not None]
    if not boxes:
        prior, bbox = np.zeros((3, 0, 0, 0), np.uint8), np.zeros((3, 2), np.int64)
    else:
        p, _ = load_probs(path, case)
        _, lm, is4 = landmarks(p)
        _, _, c4 = compute_case_roi(lm, is4, image)
        box = union_box(boxes)
        sl = box_slices(box)
        a = p_y(p, sl)
        b = dist_on_box(c4, box)
        exclude = case if case in ATLAS_SET else None
        c, _ = ATLAS.warp_to_target(c4, image, box, exclude=exclude, transforms=transforms)
        mask = boxes_mask(boxes, box)
        prior = np.stack([to_uint8(a), to_uint8(b), to_uint8(c)]) * mask[None]
        bbox = np.asarray(box, np.int64)
    N._save_npz(Path(out) / f"{i}_prior.npz", prior=prior, bbox=bbox)
    return i, N.prior_sha(prior, bbox), [int((prior[ch] > 0).sum()) for ch in range(3)]


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def write_once(path, cfg, keys):
    """Write the D1-AM2 config, or check that an existing one has exactly these decision numbers."""
    if path.exists():
        have = json.loads(path.read_text())
        diff = [k for k in keys if jsonable(have.get(k)) != jsonable(cfg.get(k))]
        if diff:
            raise SystemExit(f"{path} is frozen and this run differs on {diff}: refusing (write-once)")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    N._atomic_text(path, json.dumps(jsonable(cfg), indent=1))
    return True


def summarise(rows, train):
    def col(key, ids=None):
        v = np.array([r[key] for r in rows if (ids is None or r["id"] in ids) and r.get(key) is not None], float)
        v = v[np.isfinite(v)]
        return {"mean": float(v.mean()), "median": float(np.median(v)), "n": int(v.size)} if v.size else None
    tr = {r["id"] for r in rows if r["case"] in train}
    ho = {r["id"] for r in rows if r["case"] not in train}
    keys = ["dice_c1", "dice_c2", "dice_c3", "dice_c4", "volratio_c1", "volratio_c4", "z_lo_err_mm", "z_hi_err_mm",
            "icp_rmse_s0", "icp_rmse_s1", "D_mm"] + [f"cov_m{m}" for m in MARGINS_MM]
    out = {}
    for name, ids in (("train_inner_oof", tr), ("holdout_copies", ho)):
        sub = [r for r in rows if r["id"] in ids]
        out[name] = {k: col(k, ids) for k in keys}
        out[name].update(n=len(sub), structural_ok_rate=float(np.mean([r["structural_ok"] for r in sub])),
                         own_side_icp_rate=float(np.mean([h == "side" for r in sub for h in r["sides_how"].split("/")])),
                         fallback_rate=float(np.mean([bool(r["fallback"]) for r in sub])),
                         fallback=[r["id"] for r in sub if r["fallback"]])
    return out


def main():
    global CACHE, ATLAS, ATLAS_SET, INPUTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--variant", choices=["dev", "report"], default="dev")
    ap.add_argument("--workers", type=int, default=12)   # about 1 GB per worker (5-class probabilities)
    a = ap.parse_args()
    k, variant = a.fold, a.variant
    if variant != "dev":
        raise SystemExit("report-tier priors wait for the per-fold V4.2 freeze (V4_PLAN §11.6.1)")
    t0 = time.time()
    CACHE, harm_sha, digest = N.harm_cache(verify=True)
    N.check_direction()
    train, hold = internal_split(k)
    ATLAS_SET = set(train)
    INPUTS, ckpts = {}, {}
    for j in range(N_INNER):
        fd, _ = N.s1_dir(k, variant, j)
        for m in (f"s142_{variant}_f{k}_i{j}.done", f"s1p42_{variant}_f{k}_i{j}.done"):
            if not (ROOT / "log" / "v4" / m).exists():
                raise SystemExit(f"log/v4/{m} missing: Stage 1 {variant} fold {k} inner {j} is not complete")
        ckpts[str(fd)] = N._file_sha(fd / "checkpoint_final.pth")
        for c in inner_folds(k, tier=variant)[j]:
            INPUTS[c] = (c, str(fd / "validation" / f"{c}.npz"), j)
        for c in hold:
            INPUTS[N.copy_id(c, j)] = (c, str(fd / "holdout42" / f"{c}.npz"), j)
    assert sorted(i for i in INPUTS if i in ATLAS_SET) == sorted(train), "inner folds do not partition T_k - V_k"
    missing = [i for i, (_, p, _) in INPUTS.items() if not Path(p).exists()]
    assert not missing, f"Stage 1 probabilities missing for {missing[:5]}"
    out = N.PRIORS42 / f"fold{k}_{variant}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "DONE").unlink(missing_ok=True)
    ctx = mp.get_context("fork")

    with ctx.Pool(a.workers) as p:
        stats = dict(p.map(stats_one, train))
    template = choose_template(stats)
    ATLAS = Atlas(load_label(template), load_image(template))
    with ctx.Pool(a.workers) as p:
        for cid, ws in p.imap_unordered(warp_one, train):
            ATLAS.add_warped(cid, ws)
    print(f"atlas: template {template}, {ATLAS.n} cases ({(time.time() - t0) / 60:.1f} min)", flush=True)

    ids = sorted(INPUTS)
    with ctx.Pool(a.workers) as p:
        res = {r["id"]: r for r in p.imap_unordered(pass1, ids)}
    print(f"pass 1: {len(res)} prior ids ({(time.time() - t0) / 60:.1f} min)", flush=True)

    rule = freeze_rule({c: res[c]["cov"] for c in train}, {c: res[c]["D_mm"] for c in train})
    trig = [c for c in train if fallback_reasons(res[c]["D_mm"], res[c]["structural_ok"], res[c]["hows"],
                                                 rule["L"], rule["U"])]
    cfg = {"amendment": AMENDMENT, "fold": k, **rule, "train_triggers": trig,
           "per_case_D": {c: res[c]["D_mm"] for c in train},
           "per_case_cov": {c: res[c]["cov"] for c in train},
           "stage1_inputs_sha256": {c: res[c]["sha_in"] for c in train}, "stage1_checkpoints_sha256": ckpts,
           "atlas_template": template, "harm_config_sha256": harm_sha, "harm_label_digest": digest,
           "computed": datetime.now().isoformat(),
           "source": "tools/v4_build_priors42.py: dev-tier inner-OOF Stage 1 (Dataset541) of T_k - V_k and the "
                     "harmonised GT Y of T_k - V_k only; no V_k, no report fold; never recalibrated"}
    qc_path = N.QC42 / f"fold{k}.json"
    new = write_once(qc_path, cfg, ["amendment", "fold", "margin", "min_cov_at_margin", "rule_feasible", "L", "U",
                                    "per_case_D", "per_case_cov", "stage1_inputs_sha256", "atlas_template",
                                    "stage1_checkpoints_sha256", "train_triggers",
                                    "harm_config_sha256", "harm_label_digest"])
    qc = json.loads(qc_path.read_text())
    m_k, L, U = int(qc["margin"]), float(qc["L"]), float(qc["U"])
    print(f"D1-AM2 fold {k}: m={m_k} (min cov {qc['min_cov_at_margin']:.4f}, feasible {qc['rule_feasible']}) "
          f"L={L:.3f} U={U:.3f} train triggers {trig} ({'written' if new else 'reproduced'} {qc_path})", flush=True)

    tasks, rows = [], []
    for i in ids:
        r = res[i]
        reasons = fallback_reasons(r["D_mm"], r["structural_ok"], r["hows"], L, U)
        r["rec"]["qc"] = {"amendment": AMENDMENT, "fallback": bool(reasons), "D_mm": r["D_mm"], "L": L, "U": U,
                          "margin": m_k, "sides_how": r["hows"], "reason": "+".join(reasons) or None}
        tasks.append((i, r["rec"], r["transforms"], reasons, m_k, str(out)))
        rows.append({**r["qc"], "fallback": "+".join(reasons)})
    shas, nz = {}, {}
    with ctx.Pool(a.workers) as p:
        for i, sha, cnt in p.imap_unordered(pass2, tasks):
            shas[i], nz[i] = sha, cnt
    for i in ids:
        case, path, j = INPUTS[i]
        rec = dict(res[i]["rec"], id=i, case=case, stage1_inner=j, stage1_npz=path, stage1_sha256=res[i]["sha_in"],
                   atlas_warp=res[i]["info"], atlas_loo=case in ATLAS_SET, nonzero_prior_voxels=nz[i])
        N._atomic_text(out / f"{i}_roi.json", json.dumps(jsonable(rec)))

    with open(out / "stage1_qc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["id"]))
    summ = summarise(rows, set(train))
    N._atomic_text(out / "stage1_qc.json", json.dumps(jsonable(summ), indent=1))
    done = {"fold": k, "variant": variant, "amendment": AMENDMENT, "margin": m_k, "L": L, "U": U,
            "qc42_sha256": N._file_sha(qc_path), "harm_config_sha256": harm_sha, "harm_label_digest": digest,
            "atlas_template": template, "n_atlas": ATLAS.n, "stage1_checkpoints_sha256": ckpts,
            "ids": {i: {"case": INPUTS[i][0], "inner": INPUTS[i][2], "stage1_sha256": res[i]["sha_in"],
                        "sha": shas[i], "fallback": bool(tasks[n][3])} for n, i in enumerate(ids)},
            "digest": N._digest(shas),
            "tool_sha256": N._file_sha(Path(__file__).resolve()), "created": datetime.now().isoformat(),
            "minutes": (time.time() - t0) / 60}
    N._atomic_text(out / "DONE", json.dumps(jsonable(done), indent=1))
    print(json.dumps({"fold": k, "n_ids": len(ids), "margin": m_k, "fallback": [i for i in ids if done["ids"][i]["fallback"]],
                      "digest": done["digest"][:12], "minutes": round(done["minutes"], 1)}), flush=True)


if __name__ == "__main__":
    main()
