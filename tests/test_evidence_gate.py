"""CPU tests for the evidence-gated anatomical memory (Claim-M★) wiring.

Run from the repository root:
    python tests/test_evidence_gate.py
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "trainer", ROOT / "MIA" / "scripts"):
    sys.path.insert(0, str(p))

import ablation_configs as AC  # noqa: E402
import npc_vesselFM3D_cascade_MIA_v3_vit_memory as T  # noqa: E402
from cnv_protocol_s2 import (  # noqa: E402
    BinaryYStage2Model, EvidenceGatedYLoss, evidence_map_g1,
)

PATCH = (32, 64, 64)


def _args(**over):
    ns = argparse.Namespace(
        num_prototypes_per_class=T.NUM_PROTOTYPES_PER_CLASS, memory_momentum=T.MEMORY_MOMENTUM,
        memory_temperature=T.MEMORY_TEMPERATURE, memory_top_k=T.MEMORY_TOP_K,
        retrieval_mode="cross_attention", memory_type="anatomical_v4", backbone="nnunet_mla",
        use_mla=1, dual_path_stages=[], residual_conv_stages=[],
        bone_norm_thr=0.97, evidence_temp=0.01,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _build(use_mb=True, use_gate=True, alpha=None):
    T.GATE_ALPHA_FIXED = alpha
    m = T.build_binary_y_stage2_model(_args(), use_mb, use_gate=use_gate, in_channels=3)
    if use_mb:
        m.net.memory_bank.enabled.fill_(True)
    return m


def _input(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(1, 3, *PATCH)
    x[:, 0] = torch.rand(*PATCH, generator=g) * 0.8   # soft-tissue CT
    x[:, 0, 10:14, 20:30, 20:30] = 1.0          # bone-like CT
    x[:, 1:, 16:20, 30:40, 30:40] = 1.0         # class4 prior
    return x


def test_evidence_map():
    x = _input()
    g = evidence_map_g1(x, smooth=1)
    assert g.shape == (1, 1, *PATCH)
    assert 0.0 <= g.min() and g.max() <= 1.0
    assert g[0, 0, 12, 25, 25] > 0.9, "bone must be high evidence"
    assert g[0, 0, 18, 35, 35] > 0.9, "class4 canal prior must be high evidence"
    assert g[0, 0, 2, 5, 5] < 0.1, "soft tissue must be low evidence"
    g_ct = evidence_map_g1(x[:, :1], smooth=1)
    assert g_ct[0, 0, 18, 35, 35] < 0.1, "CT-only input has no canal prior"


def test_loss_terms():
    torch.manual_seed(0)
    y = torch.zeros(1, *PATCH)
    y[0, 8:24, 16:48, 16:48] = 1
    logits = torch.randn(1, 1, *PATCH, requires_grad=True)
    mem = torch.randn(1, 1, *PATCH)
    g = torch.zeros(1, 1, *PATCH)
    g[..., :16, :, :] = 1.0
    ignore = torch.zeros(1, *PATCH, dtype=torch.bool)

    full = EvidenceGatedYLoss(mem_align="low")(logits, y, ignore, g=g, y_mem_logits=mem)
    assert {"seg", "low", "mem_sup", "align", "total"} <= set(full)
    no_low = EvidenceGatedYLoss(lambda_L=0.0, mem_align="low")(logits, y, ignore, g=g, y_mem_logits=mem)
    assert "low" not in no_low and no_low["total"] < full["total"]
    glob = EvidenceGatedYLoss(mem_align="global")(logits, y, ignore, g=torch.ones_like(g), y_mem_logits=mem)
    assert "align" in glob, "global alignment must not depend on L"
    lowonly = EvidenceGatedYLoss(mem_align="low")(logits, y, ignore, g=torch.ones_like(g), y_mem_logits=mem)
    assert "align" not in lowonly and "low" not in lowonly, "no L voxels -> no L terms"
    full["total"].backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_gate_switches_inference_source():
    torch.manual_seed(0)
    m = _build(use_mb=True, use_gate=True).eval()
    x = _input()
    with torch.no_grad():
        m.gate_mode = "fixed"
        m.gate_alpha_fixed = 1.0                          # f = f_loc everywhere
        y_loc = m(x)[1]
        m.net.memory_bank.enabled.fill_(False)            # memory bypassed entirely
        m.gate_mode = "none"
        y_nomem = m(x)[1]
        m.net.memory_bank.enabled.fill_(True)
        m.gate_alpha_fixed = 0.0                          # f = f_mem everywhere
        m.gate_mode = "fixed"
        y_mem = m(x)[1]
        m.gate_mode = "evidence"
        y_ev = m(x)[1]
    assert torch.allclose(y_loc, y_nomem, atol=1e-5), "g=1 must reproduce the memory-free path"
    assert not torch.allclose(y_loc, y_mem, atol=1e-4), "g=0 must use memory features"
    assert not torch.allclose(y_ev, y_loc, atol=1e-4) and not torch.allclose(y_ev, y_mem, atol=1e-4)


def test_training_step_gradients():
    torch.manual_seed(0)
    m = _build(use_mb=True, use_gate=True).train()
    x = _input()
    y = torch.zeros(1, *PATCH)
    y[0, 8:24, 16:48, 16:48] = 1
    y_prob, y_logits, g, y_mem = m(x)
    assert y_mem is not None and y_mem.shape == y_logits.shape == (1, 1, *PATCH)
    assert torch.allclose(y_prob, torch.sigmoid(y_logits)), "prediction must be the trained head"
    loss = EvidenceGatedYLoss()(y_logits, y, None, g=g, y_mem_logits=y_mem)["total"]
    loss.backward()
    assert m.y_head.weight.grad.abs().sum() > 0
    assert m.mem_head.weight.grad.abs().sum() > 0, "memory decoder must be trained"


def test_arm_models():
    a_mem = _build(use_mb=False, use_gate=False)
    assert a_mem.mem_head is None and a_mem.gate_mode == "none"
    a_gate = _build(use_mb=True, use_gate=False, alpha=0.5)
    assert a_gate.gate_mode == "fixed" and a_gate.gate_alpha_fixed == 0.5
    T.GATE_ALPHA_FIXED = None


def test_ablation_arms_unique():
    AC._assert_unique_arms()
    args = {n: AC.train_args(n) for n in AC.ROW_ORDER}
    assert "--mem_align low" in args["Full"] and "--use_confidence_gate 1" in args["Full"]
    assert "--gate_alpha_fixed 0.5" in args["A_gate"] and "--mem_align global" in args["A_gate"]
    assert "--use_memory_bank 0" in args["A_mem"] and "--lambda_L" not in args["A_mem"]
    for n in ("C1", "B0", "A2"):
        assert "--lambda_L 0" in args[n]
    assert "--stage1_pred_dir" not in args["B0"]
    assert "spatial_roll" in args["A2"]


def test_stage2_input_real_case():
    s1_dir = Path(AC.STAGE1_CV_DIR)
    cache = Path(T.PREPROCESS_CACHE_ROOT) / "sp_2.500_1.133_1.133"
    files = sorted(s1_dir.glob("*_stage1_softmax.nrrd")) if s1_dir.exists() else []
    if not files or not cache.exists():
        print("  [skip] no Stage-1 predictions / cache on this machine")
        return
    case_id = files[0].name.replace("_stage1_softmax.nrrd", "")
    image = torch.from_numpy(T.np.load(cache / f"{case_id}_image.npy")).numpy()
    T.B0_SINGLE_STAGE = False
    x = T._stage2_input(case_id, image, s1_dir)
    assert x.shape == (3, *image.shape), x.shape
    assert 0.0 <= x[1:].min() and x[1:].max() <= 1.0
    T.B0_SINGLE_STAGE = True
    assert T._stage2_input(case_id, image, s1_dir).shape == (1, *image.shape)
    T.B0_SINGLE_STAGE = False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
