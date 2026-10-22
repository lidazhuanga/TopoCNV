#!/usr/bin/env python3
"""Reproduce saved OOF Stage-2 labels with the trainer's own sliding window.

    python tools/analysis/check_oof.py --arm Full --fold 0 --cases 00801656 --gpu 4
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="Full")
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--cases", nargs="+", required=True)
ap.add_argument("--gpu", type=int, required=True)
A = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(A.gpu)

import nrrd  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import mech_common as mc  # noqa: E402

T = mc.trainer()
dev = torch.device("cuda:0")
model, ep = mc.build_model(A.arm, A.fold, dev)
pred_dir = mc.RESULTS / f"{mc.S2_TEMPLATE}{mc.ARMS[A.arm]['suffix']}_label_predictions"
for cid in A.cases:
    img, lbl, prior, _ = mc.load_case(cid, with_hu=False)
    x = T._stage2_input(cid, img, mc.S1_DIR)
    with torch.no_grad():
        y = T.sliding_window_inference(x, model, T.PATCH_SIZE, 1, dev, coord_aware=True)
    mine = y[0] > 0.5
    saved = nrrd.read(str(pred_dir / f"{cid}_stage2_label.nrrd"))[0] > 0
    Y = (lbl >= 1) & (lbl <= 4)
    dice = lambda a, b: 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)  # noqa: E731
    print(f"{A.arm} f{A.fold} {cid} ep={ep} n_mine={mine.sum()} n_saved={saved.sum()} "
          f"agree={dice(mine, saved):.4f} dice_mine={dice(mine, Y):.4f} dice_saved={dice(saved, Y):.4f}",
          flush=True)
