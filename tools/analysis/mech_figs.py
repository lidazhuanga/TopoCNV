#!/usr/bin/env python3
"""Figures for the mechanism analysis -> results/analysis_figs/.

    python tools/analysis/mech_figs.py

fig_gate_vs_H.png        g distribution in Y∩H, L, and background shells; HU of the 0.97 threshold
fig_case_<id>.png        CT / g / H-L-Y / Full prediction / memory shift / Δp(memory) panels
fig_memory_reliance.png  memory shift and output change by region, Full vs A_gate
fig_L2_erasure.png       drop in Dice_Y and Recall_{Y∩E} after erasure, per arm
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
FIG = RES / "analysis_figs"
FIG.mkdir(parents=True, exist_ok=True)
MECH = Path(os.environ.get("CNV_MECH_DIR", RES / "analysis_mech"))
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"YH": "#3b6fb6", "L": "#d1495b", "ring5": "#8c8c8c", "bg20": "#c7c7c7",
     "Full": "#d1495b", "A_gate": "#3b6fb6", "A_mem": "#7a7a7a"}
NL = chr(10)
REG_TICKS = ["Y∩H", "L", "0–5" + NL + "mm", "5–20" + NL + "mm"]
NAMES = {"YH": "Y∩H (high evidence)", "L": "L = Y\\H (low evidence)",
         "ring5": "0–5 mm outside Y", "bg20": "5–20 mm outside Y"}


def fig_gate():
    d = json.loads((RES / "analysis_gate" / "gate_stats.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    edges = np.linspace(0, 1, 21)
    for r in ("YH", "L", "ring5", "bg20"):
        h = np.sum([c[f"g_{r}"]["hist"] for c in d if f"g_{r}" in c], 0).astype(float)
        ax[0].stairs(h / h.sum(), edges, color=C[r], label=NAMES[r], lw=1.2)
    ax[0].axvline(0.5, ls=":", c="k", lw=0.8)
    ax[0].set(xlabel="evidence map g", ylabel="fraction of voxels", title=f"g by region ({len(d)} cases)")
    ax[0].legend(frameon=False, fontsize=6)
    yh = [c["Y_g_high_in_YH"] for c in d]
    ll = [c["Y_g_high_in_L"] for c in d]
    ax[1].boxplot([yh, ll], widths=0.5, showfliers=True, flierprops=dict(ms=2))
    ax[1].set_xticks([1, 2], ["Y∩H", "L"])
    ax[1].set(ylabel="fraction with g ≥ 0.5", title="gate vs protocol split")
    hu = [c["hu_at_097"] for c in d]
    ax[2].hist(hu, bins=30, color="#7a7a7a")
    ax[2].axvline(200, ls="--", c="k", lw=0.8)
    ax[2].set(xlabel="HU at normalized CT = 0.97", ylabel="cases",
              title=f"bone threshold of g (median {np.median(hu):.0f} HU)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_gate_vs_H.png")
    plt.close(fig)


def _slice_idx(Y, L, axis):
    """Slice through Y with the most L voxels along ``axis``."""
    cnt = L.sum(tuple(i for i in range(3) if i != axis))
    return int(np.argmax(cnt)) if cnt.max() > 0 else int(np.argmax(Y.sum(tuple(i for i in range(3) if i != axis))))


def fig_cases():
    for f in sorted((MECH / "maps").glob("Full_*.npz")):
        cid = f.stem.split("_", 1)[1]
        m = np.load(f)
        Y, H, L = m["Y"], m["H"], m["L"]
        ct, g = m["ct"].astype(float), m["g"].astype(float)
        p, pn = m["p_orig"].astype(float), m["p_nomem"].astype(float)
        tok = m["tok_orig"].astype(float)          # [15,...]: level*5 + (gate, share, shift, dmem, pfg)
        shift0 = tok[0 * 5 + 2]
        pfg0 = tok[0 * 5 + 4]
        rows = []
        # cache axes: D = superior-inferior, H = left-right, W = anterior-posterior
        for axis, name in ((0, "axial"), (2, "coronal")):
            k = _slice_idx(Y, L, axis)
            ys = np.take(Y, k, axis=axis)
            box = np.argwhere(ys)
            lo = np.maximum(box.min(0) - 12, 0)
            hi = np.minimum(box.max(0) + 13, ys.shape)
            crop = (slice(lo[0], hi[0]), slice(lo[1], hi[1]))
            take = (lambda a, k=k, axis=axis, crop=crop: np.take(a, k, axis=axis)[crop])
            rows.append((name, k, take))
        cols = ["CT", "g", "Y∩H (blue), L (red)", "prediction (red)" + NL + "GT (yellow)",
                "memory shift" + NL + "(level 0)", "CN V prototype" + NL + "retrieval mass",
                "Δp: memory" + NL + "on − off"]
        fig, ax = plt.subplots(2, len(cols), figsize=(len(cols) * 1.45, 3.4))
        vmax_shift = max(float(np.percentile(shift0[Y], 99)), 1e-3)
        for i, (name, k, take) in enumerate(rows):
            kw = dict(aspect=1.0 if name == "axial" else 2.5 / 1.133,
                      origin="upper" if name == "axial" else "lower")
            base = take(ct)
            for j in range(len(cols)):
                a = ax[i, j]
                a.axis("off")
                if j == 1:
                    a.imshow(take(g), cmap="viridis", vmin=0, vmax=1, **kw)
                elif j in (4, 5):
                    a.imshow(take(shift0 if j == 4 else pfg0), cmap="magma", vmin=0,
                             vmax=vmax_shift if j == 4 else 1.0, **kw)
                    a.contour(take(Y).astype(float), [0.5], colors="cyan", linewidths=0.4)
                elif j == 6:
                    a.imshow(take(p - pn), cmap="RdBu_r", vmin=-0.5, vmax=0.5, **kw)
                    a.contour(take(Y).astype(float), [0.5], colors="k", linewidths=0.4)
                else:
                    a.imshow(base, cmap="gray", vmin=0.85, vmax=1, **kw)
                if j == 2:
                    ov = np.zeros(base.shape + (4,))
                    ov[take(H & Y)] = (0.23, 0.44, 0.71, 0.9)
                    ov[take(L)] = (0.82, 0.29, 0.36, 0.9)
                    a.imshow(ov, **kw)
                elif j == 3:
                    a.contour(take(Y).astype(float), [0.5], colors="yellow", linewidths=0.5)
                    a.contour(take(p), [0.5], colors="red", linewidths=0.5)
                if i == 0:
                    a.set_title(cols[j], fontsize=6)
            ax[i, 0].text(-0.05, 0.5, f"{name} {k}", transform=ax[i, 0].transAxes, rotation=90,
                          va="center", ha="right", fontsize=6)
        fig.suptitle(f"case {cid} (Full; yellow = GT Y, red = prediction)", fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / f"fig_case_{cid}.png")
        plt.close(fig)


def _load_cases():
    rows = [json.loads(p.read_text()) for p in sorted((MECH / "cases").glob("*.json"))]
    out = {}
    for r in rows:
        out.setdefault(r["arm"], []).append(r)
    return out


def fig_reliance(D):
    regs = ["YH", "L", "ring5", "bg20"]
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.1))
    qs = [("shift", 0, "memory shift, level 0"), ("shift", 1, "memory shift, level 1"),
          ("shift", 2, "memory shift, level 2")]
    w = 0.38
    for j, (s, lv, title) in enumerate(qs):
        for k, arm in enumerate(("Full", "A_gate")):
            R = D.get(arm, [])
            if not R:
                continue
            vals = [[r["tok"]["orig"].get(g, {}).get(f"{s}{lv}", np.nan) for r in R] for g in regs]
            mu = [np.nanmean(v) for v in vals]
            sd = [np.nanstd(v) for v in vals]
            ax[j].bar(np.arange(4) + (k - 0.5) * w, mu, w, yerr=sd, color=C[arm], label=f"{arm} (n={len(R)})",
                      error_kw=dict(lw=0.6))
        ax[j].set_xticks(range(4), REG_TICKS, fontsize=6)
        ax[j].set_title(title)
    for k, arm in enumerate(("Full", "A_gate")):
        R = D.get(arm, [])
        if not R:
            continue
        vals = [[(r["delta"]["orig-nomem"].get(g) or {}).get("abs", np.nan) for r in R] for g in regs]
        ax[3].bar(np.arange(4) + (k - 0.5) * w, [np.nanmean(v) for v in vals], w,
                  yerr=[np.nanstd(v) for v in vals], color=C[arm], error_kw=dict(lw=0.6))
    ax[3].set_xticks(range(4), REG_TICKS, fontsize=6)
    ax[3].set_title("|p(mem on) − p(mem off)|")
    ax[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fig_memory_reliance.png")
    plt.close(fig)


def fig_l2(D):
    mets = [("Dice_Y", "drop in Dice_Y"), ("Recall_YE", "drop in recall on Y∩E"), ("Recall_L", "drop in Recall_L")]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    arms = [a for a in ("Full", "A_gate", "A_mem") if D.get(a)]
    w = 0.38
    for j, (m, title) in enumerate(mets):
        for k, e in enumerate(("E1", "E2")):
            mu, sd = [], []
            for arm in arms:
                d = [r["metrics"]["orig"][m] - r["metrics"][e][m] for r in D[arm] if e in r["metrics"]]
                mu.append(np.nanmean(d))
                sd.append(np.nanstd(d))
            ax[j].bar(np.arange(len(arms)) + (k - 0.5) * w, mu, w, yerr=sd, label=e,
                      color=["#555555", "#aaaaaa"][k], error_kw=dict(lw=0.6))
        ax[j].set_xticks(range(len(arms)), [f"{a}\n(n={len(D[a])})" for a in arms], fontsize=6)
        ax[j].axhline(0, c="k", lw=0.5)
        ax[j].set_title(title)
    ax[0].legend(frameon=False, fontsize=6, title="erasure", title_fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fig_L2_erasure.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_gate()
    if (MECH / "cases").exists():
        D = _load_cases()
        fig_cases()
        fig_reliance(D)
        fig_l2(D)
    print(sorted(p.name for p in FIG.glob("*.png")))
