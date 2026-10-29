"""向后兼容 wrapper；实现已迁移至 ``topocnv.models.memory_bank``。"""
from topocnv.models.memory_bank import AnatomySlotMemoryBankV4

__all__ = ["AnatomySlotMemoryBankV4"]


if __name__ == "__main__":
    from topocnv.models.memory_bank import _self_test
    _self_test()
