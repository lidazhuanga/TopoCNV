"""向后兼容 wrapper；实现已迁移至 ``topocnv.models.nnformer_mla``。"""
from topocnv.models import nnformer_mla as _impl
from topocnv.models.nnformer_mla import *

__all__ = _impl.__all__


if __name__ == "__main__":
    from topocnv.models.nnformer_mla import _self_test
    _self_test()
