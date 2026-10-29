"""配置驱动的 TopoCNV 模型与记忆库工厂。"""
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Optional

import torch.nn as nn

from topocnv.config import ModelConfig
from topocnv.config import defaults
from .dynunet_mla import DynUNetMLA3D, MEMORY_FEATURE_DIMS
from .evidence_gate import BinaryYStage2Model
from .memory_bank import AnatomySlotMemoryBank, AnatomySlotMemoryBankV4
from .sota import SUPPORTED_MODELS, build_sota_model
from .vit_unetr_mla import CNNDecoder3D, ContextAnchorMemoryBank, MLAMemoryBank, ViTUNet3D


def _values(config: Optional[Any] = None, **overrides):
    if config is None:
        values = {}
    elif isinstance(config, Mapping):
        values = dict(config)
    elif is_dataclass(config):
        values = asdict(config)
    else:
        values = dict(vars(config))
    values.update({key: value for key, value in overrides.items() if value is not None})
    return values


def _stage_list(value):
    if value is None or isinstance(value, (list, tuple)):
        return value
    if isinstance(value, str):
        return [int(item) for item in value.split(",") if item.strip()]
    return value


def build_memory_bank(
    num_classes: int,
    num_prototypes_per_class: int = defaults.NUM_PROTOTYPES_PER_CLASS,
    momentum: float = defaults.MEMORY_MOMENTUM,
    temperature: float = defaults.MEMORY_TEMPERATURE,
    top_k: int = defaults.MEMORY_TOP_K,
    retrieval_mode: str = "cross_attention",
    memory_type: str = "mla",
    feature_dims=None,
    kv_latent_dim: int = defaults.VIT_KV_LATENT_DIM,
    num_heads: int = defaults.VIT_NUM_HEADS,
    head_dim: int = defaults.VIT_HEAD_DIM,
    patch_grid=defaults.MEMORY_PATCH_GRID,
):
    """按历史训练入口的参数约定构建 memory bank。"""
    fd = list(feature_dims or [defaults.VIT_EMBED_DIM,
                               defaults.VIT_EMBED_DIM * 2,
                               defaults.VIT_EMBED_DIM * 4])
    common = dict(feature_dims=fd, num_classes=num_classes,
                  kv_latent_dim=kv_latent_dim,
                  num_slots_per_class=num_prototypes_per_class,
                  num_heads=num_heads, head_dim=head_dim,
                  temperature=temperature, top_k=top_k, momentum=momentum,
                  patch_grid=patch_grid)
    if memory_type == "anatomical_v6":
        from topocnv.models.memory_bank_v6 import AnatomySlotMemoryBankV6
        return AnatomySlotMemoryBankV6(**common)
    if memory_type == "anatomical_v4_mgp":
        from topocnv.models.memory_bank_mgp import AnatomySlotMemoryBankV4MGP
        return AnatomySlotMemoryBankV4MGP(**common)
    if memory_type == "anatomical_v5":
        from topocnv.models.memory_bank_v5 import AnatomySlotMemoryBankV5
        return AnatomySlotMemoryBankV5(**common)
    if memory_type == "anatomical_v4":
        return AnatomySlotMemoryBankV4(**common)
    if memory_type == "anatomical":
        return AnatomySlotMemoryBank(**common)
    if memory_type == "context_anchor":
        num_anchors = 8
        protos_per_anchor = max(1, num_prototypes_per_class // num_anchors)
        return ContextAnchorMemoryBank(
            feature_dims=fd, num_classes=num_classes,
            kv_latent_dim=kv_latent_dim, num_anchors=num_anchors,
            protos_per_anchor=protos_per_anchor, num_heads=num_heads,
            head_dim=head_dim, temperature=temperature,
            patch_grid=patch_grid, top_k=top_k, momentum=momentum)
    if memory_type != "mla":
        raise ValueError(f"Unknown memory_type: {memory_type}")
    return MLAMemoryBank(
        feature_dims=fd, num_prototypes_per_class=num_prototypes_per_class,
        num_classes=num_classes, kv_latent_dim=kv_latent_dim,
        num_heads=num_heads, head_dim=head_dim, momentum=momentum,
        temperature=temperature, patch_grid=patch_grid, top_k=top_k,
        retrieval_mode=retrieval_mode)


def build_backbone(
    in_channels: int,
    num_classes: int,
    memory_bank=None,
    use_mla: bool = True,
    backbone: str = "vit",
    dual_path_stages=None,
    residual_conv_stages=None,
    embed_dim: int = defaults.VIT_EMBED_DIM,
    stem_channels=defaults.VIT_STEM_CHANNELS,
    tokenize_kernel=defaults.VIT_TOKENIZE_KERNEL,
    num_heads: int = defaults.VIT_NUM_HEADS,
    head_dim: int = defaults.VIT_HEAD_DIM,
    kv_latent_dim: int = defaults.VIT_KV_LATENT_DIM,
    encoder_depths=defaults.VIT_ENCODER_DEPTHS,
    decoder_depths=defaults.VIT_DECODER_DEPTHS,
    mlp_ratio: float = defaults.VIT_MLP_RATIO,
    dropout: float = defaults.VIT_DROPOUT,
    drop_path_rate: float = defaults.VIT_DROP_PATH_RATE,
):
    """构建 ViT、nnFormer-v5 或 DynUNet-MLA backbone。"""
    if backbone == "nnunet_mla":
        return DynUNetMLA3D(
            in_channels=in_channels, num_classes=num_classes,
            num_heads=num_heads, head_dim=head_dim,
            kv_latent_dim=kv_latent_dim, use_mla=use_mla,
            memory_bank=memory_bank)
    common = dict(
        in_channels=in_channels, num_classes=num_classes, embed_dim=embed_dim,
        stem_channels=tuple(stem_channels), tokenize_kernel=tuple(tokenize_kernel),
        num_heads=num_heads, head_dim=head_dim, kv_latent_dim=kv_latent_dim,
        encoder_depths=list(encoder_depths), decoder_depths=list(decoder_depths),
        mlp_ratio=mlp_ratio, dropout=dropout, drop_path_rate=drop_path_rate,
        memory_bank=memory_bank, use_mla=use_mla)
    if backbone == "nnformer_v5":
        from topocnv.models.nnformer_mla import NNFormerUNet3D
        return NNFormerUNet3D(
            dual_path_stages=_stage_list(dual_path_stages),
            residual_conv_stages=_stage_list(residual_conv_stages), **common)
    if backbone != "vit":
        raise ValueError(f"Unknown backbone: {backbone}")
    return ViTUNet3D(**common)


def build_model(config: Optional[ModelConfig] = None, **overrides) -> nn.Module:
    """从 ModelConfig、mapping 或关键字构建基础模型。"""
    cfg = ModelConfig.from_dict(_values(config, **overrides))
    if cfg.model in SUPPORTED_MODELS:
        return build_sota_model(cfg.model, cfg.in_channels, cfg.num_classes)
    if cfg.model in ("unet3d", "unet"):
        from .unet3d import UNet3D
        return UNet3D(cfg.in_channels, cfg.num_classes)
    backbone = cfg.backbone
    if cfg.model == "dynunet_mla":
        backbone = "nnunet_mla"
    elif cfg.model == "nnformer_v5":
        backbone = "nnformer_v5"
    elif cfg.model not in ("vit_unetr_mla", "vit", "backbone"):
        raise ValueError(f"Unknown model: {cfg.model}")
    memory_bank = None
    if cfg.use_memory_bank:
        memory_bank = build_memory_bank(
            num_classes=cfg.num_classes,
            num_prototypes_per_class=cfg.num_prototypes_per_class,
            momentum=cfg.memory_momentum, temperature=cfg.memory_temperature,
            top_k=cfg.memory_top_k, retrieval_mode=cfg.retrieval_mode,
            memory_type=cfg.memory_type,
            feature_dims=MEMORY_FEATURE_DIMS if backbone == "nnunet_mla" else None,
            kv_latent_dim=cfg.kv_latent_dim, num_heads=cfg.heads,
            head_dim=cfg.head_dim, patch_grid=cfg.memory_patch_grid)
    return build_backbone(
        cfg.in_channels, cfg.num_classes, memory_bank=memory_bank,
        use_mla=cfg.use_mla, backbone=backbone,
        dual_path_stages=cfg.dual_path_stages,
        residual_conv_stages=cfg.residual_conv_stages,
        embed_dim=cfg.embed_dim, stem_channels=cfg.stem_channels,
        tokenize_kernel=cfg.tokenize_kernel, num_heads=cfg.heads,
        head_dim=cfg.head_dim, kv_latent_dim=cfg.kv_latent_dim,
        encoder_depths=cfg.encoder_depths, decoder_depths=cfg.decoder_depths,
        mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
        drop_path_rate=cfg.drop_path_rate)


def build_binary_y_stage2_model(config: Optional[ModelConfig] = None,
                                use_memory_bank: Optional[bool] = None,
                                use_gate: Optional[bool] = None,
                                in_channels: int = 3,
                                **overrides) -> BinaryYStage2Model:
    """构建与主训练脚本相同的 BinaryY evidence-gated Stage-2 模型。"""
    cfg = ModelConfig.from_dict(_values(config, **overrides))
    use_mb = cfg.use_memory_bank if use_memory_bank is None else use_memory_bank
    gate = cfg.use_confidence_gate if use_gate is None else use_gate
    memory_bank = None
    if use_mb:
        memory_bank = build_memory_bank(
            2, cfg.num_prototypes_per_class, cfg.memory_momentum,
            cfg.memory_temperature, cfg.memory_top_k, cfg.retrieval_mode,
            cfg.memory_type,
            MEMORY_FEATURE_DIMS if cfg.backbone == "nnunet_mla" else None,
            cfg.kv_latent_dim, cfg.heads, cfg.head_dim, cfg.memory_patch_grid)
    net = build_backbone(
        in_channels, 1, memory_bank=memory_bank, use_mla=cfg.use_mla,
        backbone=cfg.backbone, dual_path_stages=cfg.dual_path_stages,
        residual_conv_stages=cfg.residual_conv_stages,
        embed_dim=cfg.embed_dim, stem_channels=cfg.stem_channels,
        tokenize_kernel=cfg.tokenize_kernel, num_heads=cfg.heads,
        head_dim=cfg.head_dim, kv_latent_dim=cfg.kv_latent_dim,
        encoder_depths=cfg.encoder_depths, decoder_depths=cfg.decoder_depths,
        mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
        drop_path_rate=cfg.drop_path_rate)
    stem_channels = tuple(cfg.stem_channels)
    if getattr(net, "provides_fullres_features", False):
        hier_decoder = nn.Sequential(
            nn.Conv3d(32, stem_channels[0], kernel_size=3, padding=1),
            nn.InstanceNorm3d(stem_channels[0]), nn.GELU())
    else:
        hier_decoder = CNNDecoder3D(
            cfg.embed_dim, stem_channels, tuple(cfg.tokenize_kernel),
            stem_channels[0])
        hier_decoder.out_conv = nn.Sequential(
            nn.Conv3d(stem_channels[0], stem_channels[0], 3, padding=1),
            nn.InstanceNorm3d(stem_channels[0]), nn.GELU())
    mem_dim = MEMORY_FEATURE_DIMS[0] if cfg.backbone == "nnunet_mla" else None
    return BinaryYStage2Model(
        net, hier_decoder, feat_dim=stem_channels[0],
        use_confidence_gate=gate, in_channels=in_channels,
        gate_alpha_fixed=cfg.gate_alpha_fixed, bone_thr=cfg.bone_norm_thr,
        evidence_temp=cfg.evidence_temp, mem_dim=mem_dim)


def build_dynunet(in_channels=1, out_channels=2):
    """复用统一 SOTA DynUNet 工厂，保留 v4 模型类型及 checkpoint 键。"""
    return build_sota_model("nnunet", in_channels, out_channels)


def stage2_channels(cfg):
    """Stage-2 输入为 CT 加先验；输出为二类 Y 加可选五类可见度头。"""
    n_in = 1 + len(cfg.get("priors", []))
    n_out = 2 + (5 if cfg.get("vis_head", {}).get("enabled") else 0)
    return n_in, n_out


_build_memory_bank = build_memory_bank
_build_vit = build_backbone

__all__ = [
    "build_memory_bank", "build_backbone", "build_model",
    "build_binary_y_stage2_model", "_build_memory_bank", "_build_vit",
    "build_dynunet", "stage2_channels",
]
