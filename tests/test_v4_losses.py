"""v4 losses: masked loss == SOTA WeightedBCEDiceLoss without a mask; skeleton / visibility losses."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topocnv.losses import MaskedWeightedBCEDice, VisibilityLoss, skeleton_recall_loss  # noqa: E402
from topocnv.losses import WeightedBCEDiceLoss  # noqa: E402


def sota_loss(cw):
    return WeightedBCEDiceLoss(weight_bce=0.3, weight_dice=0.7, include_background=False,
                               class_weights=torch.tensor(cw), focal_alpha=0.25, focal_gamma=3.0,
                               use_focal=True)


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(2, 2, 16, 16, 16)
        self.y = (torch.rand(2, 16, 16, 16) > 0.9).long()

    def test_equals_sota_loss(self):
        ours = MaskedWeightedBCEDice([0.1, 10.0])
        a, b = sota_loss([0.1, 10.0])(self.logits, self.y), ours(self.logits, self.y)
        self.assertAlmostEqual(a.item(), b.item(), places=5)
        c = ours(self.logits, self.y, torch.ones(2, 16, 16, 16))
        self.assertAlmostEqual(a.item(), c.item(), places=5)
        l3, y3 = torch.randn(2, 3, 8, 8, 8), torch.randint(0, 3, (2, 8, 8, 8))
        self.assertAlmostEqual(sota_loss([0.1, 10.0, 10.0])(l3, y3).item(),
                               MaskedWeightedBCEDice([0.1, 10.0, 10.0])(l3, y3).item(), places=5)

    def test_mask_ignores_outside(self):
        ours = MaskedWeightedBCEDice([0.1, 10.0])
        mask = torch.zeros(2, 16, 16, 16)
        mask[:, :8] = 1
        l2 = self.logits.clone()
        l2[:, :, 8:] = torch.randn_like(l2[:, :, 8:]) * 10
        self.assertAlmostEqual(ours(self.logits, self.y, mask).item(), ours(l2, self.y, mask).item(), places=5)

    def test_skeleton_recall(self):
        skel = torch.zeros(1, 8, 8, 8)
        skel[0, 4, 4, :] = 1
        self.assertAlmostEqual(skeleton_recall_loss(skel.clone(), skel).item(), 0.0, places=5)
        self.assertAlmostEqual(skeleton_recall_loss(torch.zeros_like(skel), skel).item(), 1.0, places=4)
        self.assertAlmostEqual(skeleton_recall_loss(torch.rand_like(skel), torch.zeros_like(skel)).item(), 0.0,
                               places=5)

    def test_visibility_class5(self):
        lab = torch.randint(0, 6, (1, 8, 8, 8))
        logits = torch.randn(1, 5, 8, 8, 8)
        ign, bg = VisibilityLoss("ignore"), VisibilityLoss("background")
        sel = lab[0] == 5
        l2 = logits.clone()
        l2[0][:, sel] = torch.randn(5, int(sel.sum())) * 10
        self.assertAlmostEqual(ign(logits, lab).item(), ign(l2, lab).item(), places=5)
        self.assertNotAlmostEqual(bg(logits, lab).item(), bg(l2, lab).item(), places=3)
        tgt = torch.where(lab == 5, torch.zeros_like(lab), lab)
        perfect = torch.nn.functional.one_hot(tgt, 5).permute(0, 4, 1, 2, 3).float() * 50
        self.assertLess(bg(perfect, lab).item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
