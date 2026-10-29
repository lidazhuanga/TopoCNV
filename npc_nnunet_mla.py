"""向后兼容 wrapper；实现已迁移至 ``topocnv.models.dynunet_mla``。"""
from topocnv.models import dynunet_mla as _impl
from topocnv.models.dynunet_mla import *

__all__ = _impl.__all__


if __name__ == "__main__":
    from topocnv.models.dynunet_mla import _self_test
    _self_test()
