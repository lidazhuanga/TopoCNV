"""保持 v4 随机数初始化行为的兼容接口。"""

import numpy as np
import torch


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


__all__ = ["seed_everything"]
