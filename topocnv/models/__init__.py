"""TopoCNV 模型架构、记忆库、门控与构建工厂。"""
from .vit_unetr_mla import (
    CNNDecoder3D,
    ContextAnchorMemoryBank,
    ConvStem3D,
    DropPath,
    MLAMemoryBank,
    MultiHeadLatentAttention,
    PatchEmbedding3D,
    PatchExpanding3D,
    PatchMerging3D,
    StandardMHAWrapper,
    TransformerEncoderBlock,
    ViTUNet3D,
    _DecoderBlock,
    _MLACrossFusion,
)
from .dynunet_mla import (
    FILTERS,
    GLOBAL_MLA_LEVELS,
    LOCAL_MLA_LEVELS,
    MEMORY_FEATURE_DIMS,
    MEMORY_LEVELS,
    STRIDES,
    DynUNetMLA3D,
    MLAInject,
    windowed_attention,
)
from .unet3d import ConvBlock3D, UNet3D
from .evidence_gate import BinaryYStage2Model, evidence_map_g1
from .sota import NNFormer3D, SUPPORTED_MODELS, build_sota_model


_LAZY_EXPORTS = {
    "AnatomySlotMemoryBank": (".memory_bank_v1", "AnatomySlotMemoryBank"),
    "AnatomySlotMemoryBankV4": (".memory_bank", "AnatomySlotMemoryBankV4"),
    "AnatomySlotMemoryBankV4MGP": (".memory_bank_mgp", "AnatomySlotMemoryBankV4MGP"),
    "AnatomySlotMemoryBankV5": (".memory_bank_v5", "AnatomySlotMemoryBankV5"),
    "AnatomySlotMemoryBankV6": (".memory_bank_v6", "AnatomySlotMemoryBankV6"),
    "DualPathMLABlock": (".nnformer_mla", "DualPathMLABlock"),
    "NNFormerUNet3D": (".nnformer_mla", "NNFormerUNet3D"),
    "build_memory_bank": (".builder", "build_memory_bank"),
    "build_backbone": (".builder", "build_backbone"),
    "build_binary_y_stage2_model": (".builder", "build_binary_y_stage2_model"),
    "build_model": (".builder", "build_model"),
    "build_dynunet": (".builder", "build_dynunet"),
    "stage2_channels": (".builder", "stage2_channels"),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    from importlib import import_module
    value = getattr(import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value

__all__ = [
    "DropPath", "MultiHeadLatentAttention", "StandardMHAWrapper",
    "TransformerEncoderBlock", "ConvStem3D", "PatchEmbedding3D",
    "PatchMerging3D", "PatchExpanding3D", "_DecoderBlock", "CNNDecoder3D",
    "ViTUNet3D", "_MLACrossFusion", "MLAMemoryBank", "ContextAnchorMemoryBank",
    "FILTERS", "STRIDES", "LOCAL_MLA_LEVELS", "GLOBAL_MLA_LEVELS",
    "MEMORY_LEVELS", "MEMORY_FEATURE_DIMS", "windowed_attention", "MLAInject",
    "DynUNetMLA3D", "ConvBlock3D", "UNet3D", "AnatomySlotMemoryBank",
    "AnatomySlotMemoryBankV4", "AnatomySlotMemoryBankV4MGP",
    "AnatomySlotMemoryBankV5", "AnatomySlotMemoryBankV6",
    "DualPathMLABlock", "NNFormerUNet3D", "BinaryYStage2Model", "evidence_map_g1",
    "NNFormer3D", "SUPPORTED_MODELS", "build_sota_model", "build_memory_bank",
    "build_backbone", "build_binary_y_stage2_model", "build_model",
    "build_dynunet", "stage2_channels",
]
