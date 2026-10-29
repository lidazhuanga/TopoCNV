#!/usr/bin/env python3
"""Write development-phase candidate configs (configs/v4_dev/) from the decisions made so far.

  python tools/v4_make_configs.py d1 --m 25
  python tools/v4_make_configs.py d2 --m 25
  python tools/v4_make_configs.py d3 --m 25 --priors b,c
  python tools/v4_make_configs.py d4 --m 25 --priors b,c --class5 ignore
Every candidate shares the R0 training block (configs/v4_dev/r0.yaml).
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topocnv.config.loader import load_config, save_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "configs" / "v4_dev"
VIS_WEIGHT = 0.5  # fixed a priori (V4_PLAN §4.3), not tuned


def make(name, arm, m, priors=(), class5=None, skel_w=None):
    cfg = copy.deepcopy(load_config(DEV / "r0.yaml"))
    cfg.update(name=name, arm=arm, roi={"enabled": True, "margin_mm": m}, priors=list(priors),
               prior_shuffle=False,
               vis_head={"enabled": class5 is not None, **({"class5": class5, "weight": VIS_WEIGHT} if class5 else {})},
               skeleton={"enabled": skel_w is not None, **({"weight": skel_w} if skel_w is not None else {})})
    path = DEV / f"{name}.yaml"
    save_config(cfg, path)
    print(path.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["d1", "d2", "d3", "d4"])
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--priors", default="")
    ap.add_argument("--class5", default=None)
    a = ap.parse_args()
    pri = [p for p in a.priors.split(",") if p]
    tag = "".join(pri) or "none"
    if a.step == "d1":
        make(f"r1_m{a.m}", "r1", a.m)
    elif a.step == "d2":
        for combo in ("abc", "bc", "ac", "ab"):
            make(f"r1p_{combo}", "r1p", a.m, list(combo))
    elif a.step == "d3":
        for c5, t in (("background", "bg"), ("ignore", "ign")):
            make(f"r1pv_{tag}_c5{t}", "r1pv", a.m, pri, c5)
    else:
        for w, t in ((0.5, "05"), (1.0, "10")):
            c5t = {"background": "bg", "ignore": "ign"}[a.class5]
            make(f"full_{tag}_c5{c5t}_w{t}", "full", a.m, pri, a.class5, w)


if __name__ == "__main__":
    main()
