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

    def test_v42_arms(self):
        """V4.2 candidates (V4_PLAN §11.6): no visibility head, 2 outputs, priors and S weight from the name."""
        for wt, w in (("05", 0.5), ("10", 1.0)):
            b = getattr(T, f"nnUNetTrainerV4_noP_S{wt}")
            self.assertEqual((b.PRIORS, b.VIS, b.SKEL_W, b.N_OUT), ((), None, w, 2))
            for combo in ("abc", "bc", "ac", "ab"):
                m = getattr(T, f"nnUNetTrainerV4_P{combo}_S{wt}")
                self.assertEqual((m.PRIORS, m.VIS, m.SKEL_W, m.N_OUT), (tuple(combo), None, w, 2))
        self.assertEqual(T.nnUNetTrainerV4_Pbc.SKEL_W, 0.0)       # the w* = 0 member of D2'
        s = T.nnUNetTrainerV4_SMOKE_noP_S10
        self.assertEqual((s.PRIORS, s.VIS, s.SKEL_W), ((), None, 1.0))

    def test_prior_arm_needs_prior_channels(self):
        """A prior arm on the CT-only Dataset540 fails at network build instead of training without priors."""
        with self.assertRaises(AssertionError):
            T.nnUNetTrainerV4_Pabc_S05.build_network_architecture("x", {}, [], 1, 2, True)
        self.assertTrue(issubclass(T.nnUNetTrainer_SMOKE_2x5, T.nnUNetTrainer))
        self.assertFalse(issubclass(T.nnUNetTrainer_SMOKE_2x5, T.nnUNetTrainerV4))   # stock output paths

    def test_trainer_class_resolution(self):
        """train42 takes the stock vs V4 output path from the class nnU-Net resolves, not from the name."""
        import importlib.util
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("v4_nnunet_tool", root / "tools" / "v4_nnunet.py")
        N = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(N)
        self.assertTrue(N.is_v4(N.trainer_class("nnUNetTrainerV4_noP_S05")))
        for stock in (N.R0_TRAINER, N.SMOKE_STOCK):
            self.assertFalse(N.is_v4(N.trainer_class(stock)), stock)
        with self.assertRaises(SystemExit):
            N.trainer_class("nnUNetTrainerV4_noSuchArm")

    def test_binary_labels_equal_stock(self):
        """On the binary Dataset540 (labels 0/1) the S0 loss is exactly the stock DC_and_CE on Y = label 1."""
        tg = [(t % 2).to(t.dtype) for t in targets()]
        out = outputs(2)
        ref = DeepSupervisionWrapper(DC_and_CE_loss({"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False},
                                                    {}, weight_ce=1, weight_dice=1, ignore_label=None,
                                                    dice_class=MemoryEfficientSoftDiceLoss), W)
        self.assertAlmostEqual(ref(out, tg).item(), T.V4Loss(W, True)(out, tg).item(), places=5)


class TestChannel0Only(unittest.TestCase):
    """Production convention (nnUNetDataLoader3D.generate_train_batch): transforms are called
    per sample as f(**{'image': t[C, X, Y, Z], 'segmentation': s}) — channel axis 0, no batch axis."""

    class Triple:
        def __call__(self, **d):
            d["image"] = d["image"] * 3.0
            return d

    def test_torch_only_channel0_changed(self):
        img = torch.ones(4, 2, 2, 2)
        seg = torch.zeros(1, 2, 2, 2, dtype=torch.int16)
        d = T.Channel0Only(self.Triple())(image=img, segmentation=seg)
        self.assertTrue(torch.equal(d["image"][0], torch.full((2, 2, 2), 3.0)))
        self.assertTrue(torch.equal(d["image"][1:], torch.ones(3, 2, 2, 2)))
        self.assertTrue(torch.equal(d["segmentation"], seg))

    def test_numpy_only_channel0_changed(self):
        img = np.ones((4, 2, 2, 2), np.float32)
        d = T.Channel0Only(self.Triple())(image=img)
        self.assertAlmostEqual(float(d["image"][0].mean()), 3.0)
        self.assertAlmostEqual(float(d["image"][1:].mean()), 1.0)
        self.assertEqual(d["image"].shape, img.shape)

    def test_single_channel_passthrough(self):
        img = np.ones((1, 2, 2, 2), np.float32)
        d = T.Channel0Only(self.Triple())(image=img)
        self.assertEqual(d["image"].shape, img.shape)
        self.assertAlmostEqual(float(d["image"].mean()), 3.0)

    def test_intensity_list_covers_installed_pipeline(self):
        """INTENSITY_DA must name every stock intensity transform of the INSTALLED nnU-Net v2
        (a typo here silently leaves a transform unwrapped -> priors corrupted)."""
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        tr_t = nnUNetTrainer.get_training_transforms(
            (32, 32, 32), (0.0, 1.0), [(1, 1, 1)], (0, 1, 2), False,
            [False] * 4, False, [1], None, None)
        names = set()
        for x in tr_t.transforms:
            names.add(type(x).__name__)
            inner = getattr(x, "transform", None)
            if inner is not None:
                names.add(type(inner).__name__)
        stock = {"GaussianNoiseTransform", "GaussianBlurTransform", "MultiplicativeBrightnessTransform",
                 "ContrastTransform", "SimulateLowResolutionTransform", "GammaTransform"}
        self.assertTrue(stock <= names, f"installed pipeline lost transforms: {stock - names}")
        self.assertTrue(stock <= set(T.INTENSITY_DA), f"INTENSITY_DA misses: {stock - set(T.INTENSITY_DA)}")
        wrapped = T.nnUNetTrainerV4.wrap_intensity_ct_only(tr_t)
        inner = set()
        for x in wrapped.transforms:
            if isinstance(x, T.Channel0Only):
                inner.add(type(x.inner).__name__)
            elif type(x).__name__ == "RandomTransform" and isinstance(getattr(x, "transform", None), T.Channel0Only):
                inner.add(type(x.transform.inner).__name__)
        self.assertTrue(stock <= inner, f"wrap missed: {stock - inner}")
        others = {type(x).__name__ for x in wrapped.transforms} - {"RandomTransform"}
        self.assertIn("SpatialTransform", others)          # spatial DA untouched
        self.assertIn("MirrorTransform", others)

    def test_real_intensity_chain_leaves_priors_identical(self):
        """End-to-end over the real stock pipeline (spatial/mirror removed, probabilities forced
        to 1): channels 1..3 must come back byte-identical, channel 0 must be augmented."""
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        tr_t = nnUNetTrainer.get_training_transforms(
            (32, 32, 32), (0.0, 1.0), None, (0, 1, 2), False, [False] * 4, False, [1], None, None)
        T.nnUNetTrainerV4.wrap_intensity_ct_only(tr_t)
        keep = []
        for x in tr_t.transforms:
            name = type(x).__name__
            if name in ("SpatialTransform", "MirrorTransform", "RemoveLabelTransform",
                        "DownsampleSegForDSTransform", "NumpyToTensor"):
                continue
            if name == "RandomTransform":
                x.apply_probability = 1.0
                keep.append(x)
        self.assertGreaterEqual(len(keep), 4, "expected several wrapped intensity transforms")
        torch.manual_seed(0)
        img = torch.rand(4, 32, 32, 32)
        seg = torch.zeros((1, 32, 32, 32), dtype=torch.int16)
        d = {"image": img.clone(), "segmentation": seg.clone()}
        for x in keep:
            d = x(**d)
        self.assertTrue(torch.equal(d["image"][1:], img[1:]), "prior channels must pass through")
        self.assertFalse(torch.equal(d["image"][0], img[0]), "CT channel must be augmented")
        self.assertTrue(torch.equal(d["segmentation"], seg))


class TestNOutLabelManager(unittest.TestCase):
    def test_reports_n_out_and_delegates(self):
        lm = T._NOutLabelManager(type("LM", (), {"num_segmentation_heads": 6,
                                                                 "foreground_labels": [1, 2, 3, 4]})(), 7)
        self.assertEqual(lm.num_segmentation_heads, 7)
        self.assertEqual(lm.foreground_labels, [1, 2, 3, 4])

    def test_arm_head_counts(self):
        self.assertEqual(T.nnUNetTrainerV4_R1._n_out(), 2)
        self.assertEqual(T.nnUNetTrainerV4_SMOKE_Pabc_Vign_S10._n_out(), 7)


if __name__ == "__main__":
    unittest.main()
