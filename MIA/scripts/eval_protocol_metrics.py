#!/usr/bin/env python3
"""
Thin eval wrapper for CNV protocol metrics (Phase 1).

Scores pred vs gt using metrics_protocol.py (CNV protocol v0.3 / §13.1):
  Primary: Dice_Y, HD95_Y
  Secondary: Dice_123, Dice_c4 (+ Secondary_source)
  L1: Dice_L (FP-aware, v2), Precision_L, Recall_L, L_over_Y;
      audit columns Dice_L_legacy (v1 FN-only score) and Dice_H

Secondary policy:
  - --pred_binary_Y without --s1_pred_dir → Secondary N/A (binary-only arms)
  - --pred_binary_Y with --s1_pred_dir → Secondary from Stage-1 multiclass (cascade)
  - multiclass pred → Secondary from pred

Callable paths:
  1) From numpy arrays / .npy folders (standalone; no trained binary head needed)
  2) Merge multiclass preds {1..5} → Y = 1∪2∪3∪4 (class5_bypass excluded)

HU: always from raw NIfTI under data_root/image/ — never from preprocess cache.

NO TRAINING. CPU-safe.

Examples:
  python MIA/scripts/eval_protocol_metrics.py --self-test
  python MIA/scripts/eval_protocol_metrics.py \\
      --pred_dir /path/to/preds --gt_cache results/preprocess_cache/sp_2.500_1.133_1.133 \\
      --data_root npc_data/train --csv /tmp/protocol_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from metrics_protocol import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_SPACING_ZYX,
    score_case_from_paths,
    score_protocol,
    _self_test,
)


def _load_pred(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(str(path))
    if path.suffix in {".nii", ".gz"} or str(path).endswith(".nii.gz"):
        import nibabel as nib
        arr = nib.load(str(path)).get_fdata()
        if arr.ndim == 3:
            return np.transpose(arr, (2, 0, 1)).astype(np.int16)
        return arr.astype(np.int16)
    if path.suffix == ".nrrd":
        import nrrd
        data, _ = nrrd.read(str(path))
        return data.astype(np.int16)
    raise ValueError(f"Unsupported pred format: {path}")


def _load_s1_multiclass(path: Path) -> np.ndarray:
    """Load Stage-1 multiclass labels or softmax → argmax labels for Secondary."""
    arr = _load_pred(path)
    if arr.ndim == 4:
        # [C,D,H,W] or [D,H,W,C] softmax
        if arr.shape[0] <= 16 and arr.shape[0] < min(arr.shape[1:]):
            return np.argmax(arr, axis=0).astype(np.int16)
        if arr.shape[-1] <= 16:
            return np.argmax(arr, axis=-1).astype(np.int16)
        raise ValueError(f"Ambiguous 4D S1 array shape: {arr.shape}")
    return arr.astype(np.int16)


def _resolve_s1(s1_dir: Path, case_id: str):
    candidates = [
        s1_dir / f"{case_id}_stage1_label.npy",
        s1_dir / f"{case_id}_stage1_label.nrrd",
        s1_dir / f"{case_id}_stage1_softmax.npy",
        s1_dir / f"{case_id}_stage1_softmax.nrrd",
        s1_dir / f"{case_id}_pred.npy",
        s1_dir / f"{case_id}.npy",
    ]
    return next((p for p in candidates if p.exists()), None)


def main():
    ap = argparse.ArgumentParser(description="CNV protocol metrics eval (no train)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pred_dir", type=str, default=None,
                    help="Directory with {case}_pred.npy or {case}_stage2_label.nrrd")
    ap.add_argument("--gt_cache", type=str, default=None,
                    help="Preprocess cache dir with {case}_label.npy")
    ap.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--spacing", type=float, nargs=3, default=list(DEFAULT_SPACING_ZYX))
    ap.add_argument("--pred_binary_Y", action="store_true",
                    help="Predictions are already binary Y masks")
    ap.add_argument("--s1_pred_dir", type=str, default=None,
                    help="Stage-1 MODEL multiclass preds for Secondary (cascade); "
                         "NOT ground-truth. Needed for Dice_123/c4 when --pred_binary_Y")
    ap.add_argument("--no_hu", action="store_true", help="Skip L1 (no raw HU load)")
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0, help="Max cases (0=all)")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    if not args.pred_dir or not args.gt_cache:
        ap.error("Need --pred_dir and --gt_cache (or --self-test)")

    pred_dir = Path(args.pred_dir)
    gt_cache = Path(args.gt_cache)
    label_files = sorted(gt_cache.glob("*_label.npy"))
    if args.limit > 0:
        label_files = label_files[: args.limit]

    rows = []
    for lp in label_files:
        case_id = lp.name.replace("_label.npy", "")
        # resolve pred
        candidates = [
            pred_dir / f"{case_id}_pred.npy",
            pred_dir / f"{case_id}_stage2_label.nrrd",
            pred_dir / f"{case_id}_stage2_label.npy",
            pred_dir / f"{case_id}.npy",
        ]
        pred_path = next((p for p in candidates if p.exists()), None)
        if pred_path is None:
            print(f"[skip] no pred for {case_id}")
            continue
        pred = _load_pred(pred_path)
        gt = np.load(str(lp))
        if pred.shape != gt.shape:
            print(f"[skip] shape mismatch {case_id}: pred={pred.shape} gt={gt.shape}")
            continue
        secondary = None
        if args.s1_pred_dir:
            s1_path = _resolve_s1(Path(args.s1_pred_dir), case_id)
            if s1_path is None:
                print(f"[warn] no S1 secondary for {case_id}; Secondary→N/A")
            else:
                secondary = _load_s1_multiclass(s1_path)
                if secondary.shape != gt.shape:
                    print(f"[warn] S1 shape mismatch {case_id}: {secondary.shape} vs {gt.shape}; Secondary→N/A")
                    secondary = None
        elif args.pred_binary_Y:
            # explicit N/A path; score_protocol sets Secondary_source
            pass

        row = score_case_from_paths(
            case_id, pred, gt,
            data_root=args.data_root,
            spacing_zyx=args.spacing,
            pred_binary_Y=args.pred_binary_Y,
            secondary_pred=secondary,
            load_hu=not args.no_hu,
        )
        rows.append(row)
        src = row.get("Secondary_source", "?")
        d123 = row["Dice_123"]
        dc4 = row["Dice_c4"]
        d123_s = "N/A" if src == "N/A" else f"{d123:.4f}"
        dc4_s = "N/A" if src == "N/A" else f"{dc4:.4f}"
        print(
            f"{case_id}: Dice_Y={row['Dice_Y']:.4f} HD95_Y={row.get('HD95_Y', float('nan')):.2f} "
            f"Dice_123={d123_s} Dice_c4={dc4_s} Secondary_source={src} "
            f"Dice_L={row['Dice_L']:.4f} Precision_L={row['Precision_L']:.4f} "
            f"Recall_L={row['Recall_L']:.4f} Dice_L_legacy={row['Dice_L_legacy']:.4f} "
            f"L/Y={row['L_over_Y']:.4f}"
        )

    if not rows:
        print("No cases scored.")
        return

    keys = ["Dice_Y", "HD95_Y", "ASSD_Y", "Dice_123", "Dice_c4", "Dice_L", "Precision_L",
            "Recall_L", "L_over_Y", "Dice_L_legacy", "Dice_H"]
    means = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    sources = sorted({str(r.get("Secondary_source", "")) for r in rows})
    print("\n=== mean over", len(rows), "cases ===")
    for k, v in means.items():
        if k in ("Dice_123", "Dice_c4") and sources == ["N/A"]:
            print(f"  {k}: N/A")
        else:
            print(f"  {k}: {v:.4f}")
    print(f"  Secondary_source: {','.join(sources) if sources else '?'}")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["case_id"] + keys + ["Secondary_source"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
            mean_row = {"case_id": "MEAN", **means, "Secondary_source": ",".join(sources)}
            w.writerow(mean_row)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
