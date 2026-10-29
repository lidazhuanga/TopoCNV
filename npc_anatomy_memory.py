"""向后兼容 wrapper；实现已迁移至 ``topocnv.models.memory_bank_v1``。"""
from topocnv.models import memory_bank_v1 as _impl
from topocnv.models.memory_bank_v1 import *

__all__ = _impl.__all__


if __name__ == "__main__":
    from topocnv.models.memory_bank_v1 import _self_test
    _self_test()
