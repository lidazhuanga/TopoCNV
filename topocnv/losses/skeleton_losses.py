"""预计算管状 GT 骨架上的软召回损失。"""


def skeleton_recall_loss(prob_fg, skel, smooth=1e-5):
    """返回一减软骨架召回率；空骨架返回零损失。"""
    s = skel.float()
    return 1.0 - ((prob_fg.float() * s).sum() + smooth) / (s.sum() + smooth)


__all__ = ["skeleton_recall_loss"]
