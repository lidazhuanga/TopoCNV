"""向后兼容 wrapper；实现已迁移至 ``topocnv.models.unet3d``。"""
from topocnv.models import unet3d as _impl
from topocnv.models.unet3d import *

__all__ = _impl.__all__
