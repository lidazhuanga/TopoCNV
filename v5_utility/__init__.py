"""独立 V5 公共 API；轻量导入，训练器和张量处理按需加载。"""
from typing import Any

from .protocol import (V5Protocol, ProtocolError, assert_dataset_slots_free, build_cohort_split,
                       data_registration, freeze_protocol, load_protocol, validate_cohort_split)
from .randomness import initialize_seed_runtime, runtime_environment

__all__ = ["V5Protocol", "ProtocolError", "freeze_protocol", "load_protocol", "build_cohort_split",
           "validate_cohort_split", "assert_dataset_slots_free", "data_registration",
           "initialize_seed_runtime", "runtime_environment", "V5Trainer", "prepare_modalities"]


def __getattr__(name: str) -> Any:
    if name == "V5Trainer":
        from .trainer import V5Trainer
        return V5Trainer
    if name == "prepare_modalities":
        from .augmentation import prepare_modalities
        return prepare_modalities
    raise AttributeError(name)
