#!/usr/bin/env python3
"""v4 Stage 1: bony-landmark segmentation (background / class 1 / class 4), nested in outer fold k.

Two tiers (V4_PLAN §5 as revised 2026-10-24):

  --tier report  inner folds over T_k. Inner model j trains on T_k minus inner fold j, then predicts
                 its inner-OOF cases (results/v4_stage1/fold{k}/pred) and report fold k
                 (results/v4_stage1/fold{k}/inner{j}/report_pred). No selection happens at this tier,
                 so V_k being ordinary training data is fine.
  --tier dev     inner folds over T_k - V_k only. Inner model j trains on (T_k - V_k) minus inner
                 fold j, then predicts its inner-OOF cases (results/v4_stage1/fold{k}_dev/pred) and
                 V_k (results/v4_stage1/fold{k}_dev/inner{j}/val_pred). V_k is held out of every dev
                 Stage-1 model; tools/v4_build_priors.py --variant dev averages the three val_pred
                 sets, so V_k gets the same 3-model-ensemble inference as report-fold cases.

Last checkpoint is used (no selection, so no further validation split is needed).

  CUDA_VISIBLE_DEVICES=4 python -m trainer.v4.train_stage1 --fold 0 --inner 0 --tier dev
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import STAGE1_DIR
from .data import V4PatchDataset
from .engine import make_optim, save_ckpt, seed_everything, train_epoch
from .infer import predict_stage1
from .losses import MaskedWeightedBCEDice
from .model import build_dynunet
from .splits import inner_folds, internal_split, report_cases, stage1_train_cases

PATCH = (128, 128, 128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--inner", type=int, required=True)
    ap.add_argument("--tier", choices=["report", "dev"], default="report")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--random_crop_prob", type=float, default=0.33)
    ap.add_argument("--step_size", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--ckpt_every", type=int, default=25)
    ap.add_argument("--smoke", type=int, default=0, help="N>0: N training cases, 2 prediction cases, scratch dir")
    ap.add_argument("--smoke_cases", default="", help="smoke: predict these holdout cases (comma list)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    k, j = args.fold, args.inner
    subdir = f"fold{k}" if args.tier == "report" else f"fold{k}_dev"
    out = STAGE1_DIR / (f"smoke/{subdir}" if args.smoke else subdir)
    run = out / f"inner{j}"
    holdout_dir = "report_pred" if args.tier == "report" else "val_pred"
    (run / holdout_dir).mkdir(parents=True, exist_ok=True)
    (out / "pred").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(1000 * k + j + (5000 if args.tier == "dev" else 0))

    train_ids = stage1_train_cases(k, j, tier=args.tier)
    pred_inner = inner_folds(k, tier=args.tier)[j]
    pred_holdout = report_cases(k) if args.tier == "report" else internal_split(k)[1]
    if args.smoke:
        want = set(args.smoke_cases.split(",")) if args.smoke_cases else set(pred_inner[:1])
        train_ids, pred_inner = train_ids[:args.smoke], [c for c in pred_inner if c in want]
        pred_holdout = pred_holdout[:1]
    cfg = vars(args) | {"patch": list(PATCH), "train_cases": train_ids, "inner_pred_cases": pred_inner,
                        "holdout_pred_cases": pred_holdout, "holdout_dir": holdout_dir,
                        "classes": ["bg", "class1", "class4"]}
    (run / "config.json").write_text(json.dumps(cfg, indent=1))

    ds = V4PatchDataset(train_ids, PATCH, stage=1, random_crop_prob=args.random_crop_prob)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0, drop_last=False)
    model = build_dynunet(1, 3).to(device)
    loss_fn = MaskedWeightedBCEDice([0.1, 10.0, 10.0]).to(device)
    opt, sched = make_optim(model, args.lr, args.epochs)
    start = 1
    latest = run / "latest_checkpoint.pth"
    if args.resume and latest.exists():
        st = torch.load(latest, map_location=device)
        model.load_state_dict(st["model_state"])
        opt.load_state_dict(st["optimizer_state"])
        sched.load_state_dict(st["scheduler_state"])
        start = st["epoch"] + 1
        print(f"resumed at epoch {st['epoch']}", flush=True)

    def step_loss(m, batch):
        x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = loss_fn(m(x), y)
        return loss, {"loss": loss.item()}

    log = open(run / "train_log.csv", "a" if start > 1 else "w")
    if start == 1:
        log.write("epoch,loss,lr,sec\n")
    for ep in range(start, args.epochs + 1):
        t0 = time.time()
        parts = train_epoch(model, loader, opt, step_loss, device)
        sched.step()
        log.write(f"{ep},{parts.get('loss', float('nan')):.5f},{opt.param_groups[0]['lr']:.6g},{time.time() - t0:.1f}\n")
        log.flush()
        if ep % 10 == 0 or ep == 1:
            print(f"epoch {ep}/{args.epochs} loss={parts.get('loss', float('nan')):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if ep % args.ckpt_every == 0 or ep == args.epochs:
            save_ckpt(latest, epoch=ep, model_state=model.state_dict(), optimizer_state=opt.state_dict(),
                      scheduler_state=sched.state_dict())
    save_ckpt(run / "last_model.pth", epoch=args.epochs, model_state=model.state_dict())

    model.eval()
    t0 = time.time()
    for case, dest in [(c, out / "pred") for c in pred_inner] + [(c, run / holdout_dir) for c in pred_holdout]:
        p1, p4 = predict_stage1(model, case, PATCH, args.step_size, device)
        np.savez_compressed(dest / f"{case}_s1prob.npz", p1=p1, p4=p4, source=f"fold{k}_tier{args.tier}_inner{j}")
    print(f"predicted {len(pred_inner)} inner-OOF + {len(pred_holdout)} holdout "
          f"({holdout_dir}) cases in {(time.time() - t0) / 60:.1f} min", flush=True)
    (run / "COMPLETED").write_text(f"{datetime.now().isoformat()}\n")


if __name__ == "__main__":
    main()
