"""向后兼容 wrapper；实现已迁移至 ``topocnv.models.vit_unetr_mla``。"""
from topocnv.models import vit_unetr_mla as _impl
from topocnv.models.vit_unetr_mla import *

__all__ = _impl.__all__
