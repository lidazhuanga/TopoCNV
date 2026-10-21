#!/usr/bin/env python3
"""
P0 ablation registry — CNV protocol v0.3-confirmed / methods §14.

Arms (locked): Full | A_mem | A_gate | B0 | C1 | A2
Each arm differs from Full (or from C1 for A2) by one factor:
  Full   : cascade prior + memory + evidence gate g(x) + lambda_L + memory alignment on L
  A_mem  : Full without memory (no memory bank, no Y_mem / alignment); lambda_L kept
  A_gate : Full with g fixed to alpha (no evidence switching); alignment made global
  C1     : cascade prior only (memory, gate, lambda_L off)
  B0     : single-stage CT-only binary Y (= external nnU-Net endpoint)
  A2     : C1 with a spatially scrambled class4 prior
Primary: Dice_Y + HD95_Y on Y=1∪2∪3∪4; L1 + Secondary required.
FORBIDDEN Primary: 5-class / soft-tree mean Dice.
class5_bypass = S1 only (not in Y / L1 / S2 target).
NO auto-launch from this file.
"""
import argparse
import os
from pathlib import Path

PROJECT_ROOT = os.environ.get("CNV_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
RESULTS_ROOT = os.environ.get("CNV_RESULTS_ROOT", f"{PROJECT_ROOT}/results")
LOG_DIR = f"{PROJECT_ROOT}/log"
TRAINER = f"{PROJECT_ROOT}/trainer/npc_vesselFM3D_cascade_MIA_v3_vit_memory.py"
PYTHON = os.environ.get("CNV_PYTHON", "python")

STAGE2_EXP_TEMPLATE = "npc_vesselFM3D_cascade_MIA_v3_vit_memory_stage2"
STAGE1_SUFFIX = "_s1_6class_protocol_v03_20261016"
STAGE1_EXP = f"npc_vesselFM3D_cascade_MIA_v3_vit_memory_stage1{STAGE1_SUFFIX}"
STAGE1_CV_DIR = f"{RESULTS_ROOT}/{STAGE1_EXP}_cv_predictions"

MAIN_SUFFIX = "_p0_binary_y"
EPOCHS = 500
NUM_FOLDS = 5

GATE_TAU = "0.5"
GATE_ALPHA_FIXED = "0.5"

BASE_ARGS = {
    "--backbone": "nnunet_mla",
    "--random_crop_prob": "0.33",
    "--memory_type": "anatomical_v4",
    "--batch_size": "2",
    "--s2_head": "binary_y",
    "--gate_tau": GATE_TAU,
    "--refuse_legacy_heads": "1",
}

CONFIGS = {
    "Full": {
        "suffix": f"{MAIN_SUFFIX}_full",
        "args": "--use_memory_bank 1 --use_confidence_gate 1 --mem_align low",
        "label": "Full (cascade + evidence-gated memory)",
    },
    "A_mem": {
        "suffix": f"{MAIN_SUFFIX}_a_mem",
        "args": "--use_memory_bank 0 --use_confidence_gate 0",
        "label": "A_mem (no memory; evidence-weighted loss kept)",
    },
    "A_gate": {
        "suffix": f"{MAIN_SUFFIX}_a_gate",
        "args": (f"--use_memory_bank 1 --use_confidence_gate 0 "
                 f"--gate_alpha_fixed {GATE_ALPHA_FIXED} --mem_align global"),
        "label": "A_gate (memory on; g fixed to alpha, no evidence switching)",
    },
    "B0": {
        "suffix": f"{MAIN_SUFFIX}_b0",
        "args": ("--use_memory_bank 0 --use_confidence_gate 0 --lambda_L 0 "
                 "--b0_single_stage 1"),
        "label": "B0 (single-stage binary Y; = external nnU-Net endpoint)",
        "b0_single_stage": True,
    },
    "C1": {
        "suffix": f"{MAIN_SUFFIX}_c1",
        "args": "--use_memory_bank 0 --use_confidence_gate 0 --lambda_L 0",
        "label": "C1 (cascade prior only; memory/gate/lambda_L off)",
    },
    "A2": {
        "suffix": f"{MAIN_SUFFIX}_a2",
        "args": ("--use_memory_bank 0 --use_confidence_gate 0 --lambda_L 0 "
                 "--prior_scramble spatial_roll"),
        "label": "A2 (C1 with scrambled class4 prior; Claim-A semantic)",
    },
}

for _cfg in CONFIGS.values():
    _cfg.setdefault("b0_single_stage", False)
    _cfg["use_memory_bank"] = "--use_memory_bank 1" in _cfg["args"]
    _cfg["use_confidence_gate"] = "--use_confidence_gate 1" in _cfg["args"]
    _cfg["zero_prior"] = _cfg["b0_single_stage"]
    _cfg["prior_scramble"] = "spatial_roll" if "--prior_scramble" in _cfg["args"] else "0"

ROW_ORDER = ["Full", "A_mem", "A_gate", "B0", "C1", "A2"]
ABLATION_NAMES = [n for n in ROW_ORDER if n != "Full"]
P0_ARM_NAMES = list(ROW_ORDER)


def stage2_exp(name: str) -> str:
    return f"{STAGE2_EXP_TEMPLATE}{CONFIGS[name]['suffix']}"


def train_args(name: str) -> str:
    cfg = CONFIGS[name]
    flags = dict(BASE_ARGS)
    flags["--exp_suffix"] = cfg["suffix"]
    # B0 needs the dir only because Stage2Dataset loads it; with --b0_single_stage 1 the
    # network input is CT alone (in_channels=1) and the prior channels are dropped.
    flags["--stage1_pred_dir"] = STAGE1_CV_DIR
    tokens = cfg["args"].split()
    i = 0
    while i < len(tokens):
        flag = tokens[i]
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            flags[flag] = tokens[i + 1]
            i += 2
        else:
            flags[flag] = "1"
            i += 1
    return " ".join(f"{flag} {value}" for flag, value in flags.items())


def train_command(name: str, fold: int) -> str:
    return f"{PYTHON} {TRAINER} --stage 2 --fold {fold} --epochs {EPOCHS} {train_args(name)}"


def _assert_unique_arms():
    seen = {}
    for name in ROW_ORDER:
        key = train_args(name).replace(CONFIGS[name]["suffix"], "")
        if key in seen:
            raise AssertionError(f"Arms {seen[key]} and {name} have identical training args")
        seen[key] = name


def main():
    ap = argparse.ArgumentParser(description="P0 ablation configs (binary Y Primary)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--main-exp", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--p0", action="store_true", help="list all six P0 arms")
    ap.add_argument("--commands", type=int, default=None, metavar="FOLD",
                    help="print the stage-2 training command of every arm for FOLD")
    args = ap.parse_args()
    _assert_unique_arms()

    if args.main_exp:
        print(stage2_exp("Full"))
        return
    if args.commands is not None:
        for name in ROW_ORDER:
            print(f"# {name}\n{train_command(name, args.commands)}")
        return
    if args.list or args.p0:
        names = P0_ARM_NAMES if args.p0 else ABLATION_NAMES
        for name in names:
            print(f"{name}\t{stage2_exp(name)}\t{train_args(name)}")
        return
    print(f"Stage-1 predictions (protocol 6-class): {STAGE1_CV_DIR}")
    print("Primary: Dice_Y + HD95_Y on Y=1U2U3U4; L1 + Secondary required.\n")
    for name in ROW_ORDER:
        print(f"[{name}]")
        print(f"  results dir : {stage2_exp(name)}")
        print(f"  train args  : {train_args(name)}")
        print(f"  row label   : {CONFIGS[name]['label']}")
        print()


if __name__ == "__main__":
    main()
