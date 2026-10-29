"""Claim-M★ evidence map 与门控 Stage-2 模型。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_GATE_TAU = 0.5
DEFAULT_GATE_ALPHA_FIXED = 0.5
DEFAULT_BONE_NORM_THR = 0.97
DEFAULT_EVIDENCE_TEMP = 0.01
DEFAULT_EVIDENCE_SMOOTH = 3


def evidence_map_g1(x, bone_thr=DEFAULT_BONE_NORM_THR,
                    temp=DEFAULT_EVIDENCE_TEMP,
                    smooth=DEFAULT_EVIDENCE_SMOOTH):
    """G1 evidence map；高值表示局部 CT 证据充分。"""
    ct = x[:, :1].float()
    g = torch.sigmoid((ct - bone_thr) / temp)
    if x.shape[1] >= 2:
        c4 = x[:, 1:min(3, x.shape[1])].float().clamp(0, 1).amax(
            dim=1, keepdim=True)
        g = torch.maximum(g, c4)
    if smooth > 1:
        g = F.avg_pool3d(g, kernel_size=smooth, stride=1,
                         padding=smooth // 2, count_include_pad=False)
    return g.clamp(0, 1)


class BinaryYStage2Model(nn.Module):
    """CT + cascade prior 到 binary Y 的 evidence-gated 模型。"""

    def __init__(self, net, hier_decoder, feat_dim: int,
                 use_confidence_gate: bool = True, in_channels: int = 3,
                 gate_alpha_fixed=None,
                 bone_thr: float = DEFAULT_BONE_NORM_THR,
                 evidence_temp: float = DEFAULT_EVIDENCE_TEMP,
                 mem_dim=None):
        super().__init__()
        self.net = net
        self.hier_decoder = hier_decoder
        self.in_channels = in_channels
        self.use_memory_bank = getattr(net, "memory_bank", None) is not None
        if gate_alpha_fixed is not None:
            self.gate_mode = "fixed"
        elif use_confidence_gate:
            self.gate_mode = "evidence"
        else:
            self.gate_mode = "none"
        self.gate_alpha_fixed = gate_alpha_fixed
        self.bone_thr = float(bone_thr)
        self.evidence_temp = float(evidence_temp)
        self.y_head = nn.Conv3d(feat_dim, 1, kernel_size=1)
        self.mem_head = (nn.Linear(mem_dim, 1)
                         if self.use_memory_bank and mem_dim and
                         hasattr(net, "evidence_gate") else None)
        print(f"✓ BinaryY Stage2 (Y=1∪2∪3∪4; class5_bypass not in target) "
              f"memory={self.use_memory_bank} gate={self.gate_mode} "
              f"mem_head={self.mem_head is not None}")

    def evidence(self, x):
        return evidence_map_g1(x, bone_thr=self.bone_thr,
                               temp=self.evidence_temp)

    def _memory_logits(self, target_shape):
        tok = getattr(self.net, "_mem_tok0", None)
        if self.mem_head is None or tok is None:
            return None
        tokens, grid = tok
        B = tokens.shape[0]
        logit = self.mem_head(tokens).transpose(1, 2).reshape(B, 1, *grid)
        padded = getattr(self.net, "_patch_padded", tuple(target_shape))
        logit = F.interpolate(logit.float(), size=padded, mode="trilinear",
                              align_corners=False)
        D, H, W = target_shape
        return logit[:, :, :D, :H, :W]

    def forward(self, x, labels=None, coord_info=None):
        g = self.evidence(x)
        if hasattr(self.net, "evidence_gate"):
            if not self.use_memory_bank or self.gate_mode == "none":
                self.net.evidence_gate = None
            elif self.gate_mode == "fixed":
                self.net.evidence_gate = float(self.gate_alpha_fixed)
            else:
                self.net.evidence_gate = g
        if hasattr(self.net, "evidence_gate"):
            feats, stem_skips, target_shape = self.net.forward_features(
                x, labels, coord_info)
        else:
            feats, stem_skips, target_shape = self.net.forward_features(x, labels)
        if getattr(self.net, "provides_fullres_features", False):
            feat = self.hier_decoder(feats)
        else:
            feat = self.hier_decoder(feats, stem_skips, target_shape)
        y_logits = self.y_head(feat)
        y_prob = torch.sigmoid(y_logits)
        y_mem_logits = self._memory_logits(tuple(x.shape[2:]))
        return y_prob, y_logits, g, y_mem_logits


__all__ = [
    "DEFAULT_GATE_TAU", "DEFAULT_GATE_ALPHA_FIXED", "DEFAULT_BONE_NORM_THR",
    "DEFAULT_EVIDENCE_TEMP", "DEFAULT_EVIDENCE_SMOOTH", "evidence_map_g1",
    "BinaryYStage2Model",
]
