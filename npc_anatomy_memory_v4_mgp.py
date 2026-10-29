"""向后兼容 wrapper；实现已迁移至 ``topocnv.models.memory_bank_mgp``。"""
from topocnv.models import memory_bank_mgp as _impl
from topocnv.models.memory_bank_mgp import *

__all__ = _impl.__all__


if __name__ == "__main__":
    from topocnv.models.memory_bank_mgp import _self_test
    _self_test()
