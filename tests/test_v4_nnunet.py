"""v4 nnU-Net trainer pieces (run under the nnU-Net venv):
  ~/cnv/envs/nnunetv2_cnv/bin/python -m unittest tests.test_v4_nnunet"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.v4.nnunet import nnUNetTrainerV4 as T  # noqa: E402
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss  # noqa: E402
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper  # noqa: E402
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss  # noqa: E402

W = np.array([1, 0.5, 0.25, 0.0])
W = W / W.sum()


def targets(shape=(2, 1, 16, 32, 32), seed=0):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 6, shape, generator=g).to(torch.int16)
    out = [raw]
    for f in (2, 4, 8):
        out.append(raw[:, :, ::f, ::f, ::f].contiguous())
    return out


def outputs(n_out, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn((2, n_out, 16 // f, 32 // f, 32 // f), generator=g) for f in (1, 2, 4, 8)]


class TestV4nnUNet(unittest.TestCase):
    def test_merge_locations(self):
        cl = {1: np.ones((3, 4), int), 2: np.zeros((0, 4), int), 4: 2 * np.ones((2, 4), int), 5: np.ones((9, 4), int)}
        m = T.merge_y_locations(cl)
        self.assertEqual(list(m), [1])
        self.assertEqual(len(m[1]), 5)
        self.assertEqual(len(T.merge_y_locations({5: np.ones((4, 4), int)})[1]), 0)

    def test_main_loss_equals_nnunet(self):
        tg, out = targets(), outputs(2)
        ref = DeepSupervisionWrapper(DC_and_CE_loss({"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False},
                                                    {}, weight_ce=1, weight_dice=1, ignore_label=None,
                                                    dice_class=MemoryEfficientSoftDiceLoss), W)
        ty = [((t >= 1) & (t <= 4)).to(t.dtype) for t in tg]
        ours = T.V4Loss(W, True)
        self.assertAlmostEqual(ref(out, ty).item(), ours(out, tg).item(), places=5)

    def test_vis_ignore_class5(self):
        tg, out = targets(), outputs(7)
        loss = T.V4Loss(W, True, vis="ignore")
        out2 = [o.clone() for o in out]
        for o, t in zip(out2, tg):
            m = (t[:, 0] == 5)
            o[:, 2:7].permute(1, 0, 2, 3, 4)[:, m] = 30.0 * torch.randn(5, int(m.sum()))
        # class-5 voxels still count for the main Y loss (as background), so compare the vis part only
        main = T.V4Loss(W, True)
        d1 = loss(out, tg) - main(out, tg)
        d2 = loss(out2, tg) - main(out2, tg)
        self.assertAlmostEqual(d1.item(), d2.item(), places=4)
        bg = T.V4Loss(W, True, vis="background")
        self.assertNotAlmostEqual((bg(out, tg) - main(out, tg)).item(), (bg(out2, tg) - main(out2, tg)).item(), places=2)

    def test_skeleton(self):
        y = np.zeros((20, 40, 40), bool)
        y[10, 5:35, 18:21] = True
        y[9:12, 5:35, 19] = True
        sk = T.tubed_skeleton(y)
        self.assertTrue(sk.any())
        self.assertFalse((sk.astype(bool) & ~y).any())
        self.assertEqual(int(T.tubed_skeleton(np.zeros_like(y)).sum()), 0)
        tr = T.make_skeleton_transform()
        seg = torch.zeros((1, 20, 40, 40), dtype=torch.int16)
        seg[0][torch.from_numpy(y)] = 2
        seg[0, 0, 0, 0] = 5
        d = tr(**{"image": torch.zeros((4, 20, 40, 40)), "segmentation": seg})
        self.assertEqual(tuple(d["segmentation"].shape), (2, 20, 40, 40))
        np.testing.assert_array_equal(d["segmentation"][1].numpy(), sk)

    def test_skeleton_recall_term(self):
        tg = targets()
        tg = [torch.cat([t, ((t >= 1) & (t <= 4)).to(t.dtype)], 1) for t in tg]
        out = outputs(2)
        with_s = T.V4Loss(W, True, skel_w=1.0)(out, tg)
        without = T.V4Loss(W, True)(out, tg)
        self.assertGreater(with_s.item(), without.item())

    def test_arms(self):
        a = T.nnUNetTrainerV4_Pbc_Vign_S05
        self.assertEqual((a.PRIORS, a.VIS, a.SKEL_W, a.N_OUT), (("b", "c"), "ignore", 0.5, 7))
        self.assertEqual(T.nnUNetTrainerV4_R1.N_OUT, 2)
        self.assertEqual(T.nnUNetTrainerV4_R1.PRIORS, ())
        self.assertEqual(T.nnUNetTrainerV4_noP_Vbg_S10.PRIORS, ())


if __name__ == "__main__":
    unittest.main()
