#!/usr/bin/env python3
"""v4 Stage-2 统一训练与报告入口，保持内部验证选择及冻结配置访问门。"""

import argparse
import csv
import json
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from topocnv.config.loader import load_config, save_config
from topocnv.data.utils import REPO, RESULTS, RUNS_DIR, load_label
from topocnv.utils.box_ops import mask_Y
from topocnv.data.v4_dataset import V4PatchDataset
from topocnv.engine import make_optim, save_ckpt, train_epoch
from topocnv.utils.seed import seed_everything
from topocnv.engine.v4_inference import predict_stage2, save_label_nrrd
from topocnv.losses import MaskedWeightedBCEDice, VisibilityLoss, skeleton_recall_loss
from topocnv.models.builder import build_dynunet, stage2_channels
from topocnv.data.v4_splits import internal_split, report_cases
from topocnv.metrics.protocol import DEFAULT_SPACING_ZYX, _dice, _hd95


def score_cases(model, cases, fold, cfg, device, with_hd95, label_dir=None, vis_dir=None):
    model.eval()
    rows = []
    for c in cases:
        pred, vis, roi_used = predict_stage2(model, c, fold, cfg, device)
        if label_dir is not None:
            save_label_nrrd(pred.astype(np.int16), label_dir / f"{c}_stage2_label.nrrd")
            if vis is not None and vis_dir is not None:
                save_label_nrrd(vis.astype(np.int16), vis_dir / f"{c}_vis_label.nrrd")
            rows.append({"patient_id": c, "roi_used": roi_used})
            continue
        gt = mask_Y(load_label(c))
        rows.append({"patient_id": c, "Dice_Y": _dice(pred, gt),
                     "HD95_Y": _hd95(pred, gt, DEFAULT_SPACING_ZYX) if with_hd95 else float("nan"),
                     "roi_used": roi_used})
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--mode", choices=["train", "report"], default="train")
    ap.add_argument("--r0_check", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--smoke", type=int, default=0, help="N>0: N train + 2 ival cases, 2 epochs, scratch dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr = cfg["train"]
    k = args.fold
    name = cfg["name"] + ("_smoke" if args.smoke else "")
    run = RUNS_DIR / name / f"fold{k}"
    run.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_in, n_out = stage2_channels(cfg)
    model = build_dynunet(n_in, n_out).to(device)
    best_path, latest = run / "best_model.pth", run / "latest_checkpoint.pth"

    if args.mode == "report":
        if not (cfg.get("frozen") or (args.r0_check and cfg["arm"] == "r0")):
            raise SystemExit("report mode needs a frozen config (or --r0_check on R0)")
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
        label_dir = RESULTS / f"v4_{cfg['arm']}_label_predictions"
        vis_dir = RESULTS / f"v4_{cfg['arm']}_vis_predictions" if n_out > 2 else None
        for d in (label_dir, vis_dir):
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)
        cases = report_cases(k)
        log_dir = REPO / "log" / "v4"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "report_access.log", "a") as f:
            f.write(f"{datetime.now().isoformat()} report-fold prediction: config={args.config} "
                    f"name={cfg['name']} arm={cfg['arm']} fold={k} n={len(cases)} "
                    f"frozen={bool(cfg.get('frozen'))} r0_check={args.r0_check}\n")
        score_cases(model, cases, k, cfg, device, False, label_dir=label_dir, vis_dir=vis_dir)
        print(f"report-fold OOF labels written to {label_dir}", flush=True)
        return

    epochs = 2 if args.smoke else tr["epochs"]
    full_val_every = 1 if args.smoke else tr["full_val_every"]
    train_ids, ival = internal_split(k)
    if args.smoke:
        train_ids, ival = train_ids[:args.smoke], ival[:2]
    seed_everything(7919 * k + 17)
    save_config(cfg | {"fold": k, "train_cases": train_ids, "ival_cases": ival,
                       "started": datetime.now().isoformat()}, run / "config.json")

    ds = V4PatchDataset(train_ids, tr["patch"], stage=2, fold=k, cfg=cfg,
                        random_crop_prob=tr["random_crop_prob"])
    loader = DataLoader(ds, batch_size=tr["batch_size"], shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    main_loss = MaskedWeightedBCEDice([0.1, 10.0]).to(device)
    use_roi = bool(cfg.get("roi", {}).get("enabled"))
    vh = cfg.get("vis_head", {})
    vis_loss = VisibilityLoss(vh.get("class5", "background")) if vh.get("enabled") else None
    sk = cfg.get("skeleton", {})
    sk_w = float(sk.get("weight", 0.0)) if sk.get("enabled") else 0.0

    def step_loss(m, batch):
        x = batch["x"].to(device, non_blocking=True)
        lab = batch["label"].to(device, non_blocking=True)
        roi = batch["roi"].to(device, non_blocking=True) if use_roi else None
        logits = m(x)
        y = ((lab >= 1) & (lab <= 4)).long()
        l_main = main_loss(logits[:, :2], y, roi)
        loss, parts = l_main, {"loss_main": l_main.item()}
        if vis_loss is not None:
            l_vis = vis_loss(logits[:, 2:7], lab, roi)
            loss = loss + float(vh.get("weight", 0.5)) * l_vis
            parts["loss_vis"] = l_vis.item()
        if sk_w > 0:
            p = torch.softmax(logits[:, :2].float(), dim=1)[:, 1]
            l_sk = skeleton_recall_loss(p, batch["skel"].to(device, non_blocking=True))
            loss = loss + sk_w * l_sk
            parts["loss_skel"] = l_sk.item()
        parts["loss"] = loss.item()
        return loss, parts

    opt, sched = make_optim(model, tr["lr"], epochs)
    best, best_ep, no_improve, start = -1.0, 0, 0, 1
    if args.resume and latest.exists():
        st = torch.load(latest, map_location=device)
        model.load_state_dict(st["model_state"])
        opt.load_state_dict(st["optimizer_state"])
        sched.load_state_dict(st["scheduler_state"])
        best, best_ep, no_improve, start = st["best"], st["best_epoch"], st["no_improve"], st["epoch"] + 1
        print(f"resumed at epoch {st['epoch']} (best {best:.4f} @ {best_ep})", flush=True)

    log = open(run / "train_log.csv", "a" if start > 1 else "w", newline="")
    lw = csv.writer(log)
    if start == 1:
        lw.writerow(["epoch", "loss", "loss_main", "loss_vis", "loss_skel", "lr", "sec", "ival_dice_Y"])
    stopped = False
    for ep in range(start, epochs + 1):
        t0 = time.time()
        parts = train_epoch(model, loader, opt, step_loss, device)
        sched.step()
        ival_dice = ""
        if ep % full_val_every == 0:
            rows = score_cases(model, ival, k, cfg, device, with_hd95=False)
            ival_dice = float(np.nanmean([r["Dice_Y"] for r in rows]))
            if ival_dice > best:
                best, best_ep, no_improve = ival_dice, ep, 0
                save_ckpt(best_path, epoch=ep, model_state=model.state_dict(), ival_dice=ival_dice)
            else:
                no_improve += 1
            print(f"  [ival @ {ep}] Dice_Y={ival_dice:.4f} (best {best:.4f} @ {best_ep}, "
                  f"no_improve={no_improve})", flush=True)
            save_ckpt(latest, epoch=ep, model_state=model.state_dict(), optimizer_state=opt.state_dict(),
                      scheduler_state=sched.state_dict(), best=best, best_epoch=best_ep, no_improve=no_improve)
            stopped = no_improve >= tr["early_stop_checks"]
        lw.writerow([ep] + [f"{parts.get(key, float('nan')):.5f}" for key in
                            ("loss", "loss_main", "loss_vis", "loss_skel")]
                    + [f"{opt.param_groups[0]['lr']:.6g}", f"{time.time() - t0:.1f}", ival_dice])
        log.flush()
        if ep % 10 == 0 or ep == 1:
            print(f"epoch {ep}/{epochs} loss={parts.get('loss', float('nan')):.4f} ({time.time() - t0:.0f}s)",
                  flush=True)
        if stopped:
            print(f"early stop at epoch {ep}", flush=True)
            break
    save_ckpt(run / "last_model.pth", epoch=ep, model_state=model.state_dict())

    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    t0 = time.time()
    rows = score_cases(model, ival, k, cfg, device, with_hd95=True)
    with open(run / "ival_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "Dice_Y", "HD95_Y", "roi_used"])
        w.writeheader()
        w.writerows(rows)
    summary = {"name": cfg["name"], "arm": cfg["arm"], "fold": k, "n_ival": len(rows), "best_epoch": best_ep,
               "last_epoch": ep, "ival_Dice_Y": float(np.nanmean([r["Dice_Y"] for r in rows])),
               "ival_HD95_Y": float(np.nanmean([r["HD95_Y"] for r in rows])),
               "ival_HD95_Y_median": float(np.nanmedian([r["HD95_Y"] for r in rows])),
               "finished": datetime.now().isoformat()}
    (run / "ival_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[ival final] {summary} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    (run / "COMPLETED").write_text(f"{summary['finished']}\n")


__all__ = ["score_cases", "main"]

if __name__ == "__main__":
    main()
