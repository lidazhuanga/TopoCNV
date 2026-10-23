"""v4 priors: ICP recovers a known similarity; atlas leave-one-out; distance prior; box reads."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.v4.common import SPACING, case_shift, region_from_box  # noqa: E402
from trainer.v4.priors import Atlas, apply_sim, dist_prior, icp_similarity  # noqa: E402


def rot(ax, deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    R = np.eye(3)
    i, j = [a for a in range(3) if a != ax]
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def toy_case(shape, off):
    lab = np.zeros(shape, np.uint8)
    for y0 in (20, 52):
        lab[18:22, y0 + off:y0 + off + 6, 25:31] = 4
        lab[22:26, y0 + off + 1:y0 + off + 4, 28:40] = 2
    return lab


class TestPriors(unittest.TestCase):
    def test_icp_similarity(self):
        rng = np.random.RandomState(1)
        src = np.concatenate([rng.randn(300, 3) * [3, 8, 4], rng.randn(200, 3) * [2, 2, 6] + [10, 15, 0]])
        T = (1.08, rot(0, 6) @ rot(2, -4), np.array([3.0, -5.0, 2.0]))
        dst = apply_sim(T, src)
        Tf, rmse = icp_similarity(src, dst)
        self.assertLess(np.linalg.norm(apply_sim(Tf, src) - dst, axis=1).max(), 1.0)
        self.assertAlmostEqual(Tf[0], 1.08, places=2)

    def test_dist_prior(self):
        m = np.zeros((10, 20, 20), bool)
        m[5, 10, 10] = True
        d = dist_prior(m, SPACING, 30.0)
        self.assertAlmostEqual(float(d[5, 10, 10]), 1.0)
        self.assertAlmostEqual(float(d[5, 10, 15]), 1 - 5 * SPACING[2] / 30, places=4)
        self.assertEqual(float(dist_prior(np.zeros_like(m)).max()), 0.0)

    def test_region_from_box(self):
        rng = np.random.RandomState(0)
        full = np.zeros((20, 30, 25), np.float32)
        box = [[4, 12], [5, 20], [3, 10]]
        full[4:12, 5:20, 3:10] = rng.rand(8, 15, 7)
        arr = full[4:12, 5:20, 3:10][None]
        reg = [[0, 16], [2, 30], [1, 9]]
        np.testing.assert_array_equal(region_from_box(arr, box, reg)[0], full[0:16, 2:30, 1:9])
        sh = case_shift("01234", full.shape)
        rolled = np.roll(full, sh, axis=(0, 1, 2))
        np.testing.assert_array_equal(region_from_box(arr, box, reg, full.shape, sh)[0], rolled[0:16, 2:30, 1:9])

    def test_atlas_loo_and_identity(self):
        shape = (40, 80, 60)
        img = np.zeros(shape, np.float32)
        img[:, 10:70, 5:55] = 0.6
        tpl = toy_case(shape, 0)
        atlas = Atlas(tpl, img)
        for cid, off in (("a", 0), ("b", 1), ("c", -1)):
            atlas.add_case(cid, toy_case(shape, off), img)
        self.assertEqual(atlas.n, 3)
        loo = atlas.side_atlas(0, exclude="a")
        expect = (atlas.warped["b"][0].astype(np.float32) + atlas.warped["c"][0].astype(np.float32)) / 2
        np.testing.assert_allclose(loo, expect, atol=1e-5)
        self.assertGreater(atlas.side_atlas(0).max(), 0.9)
        region = [[0, shape[0]], [0, shape[1]], [0, shape[2]]]
        prior, info = atlas.warp_to_target(tpl == 4, img, region)
        y = (tpl >= 1) & (tpl <= 4)
        self.assertGreater(prior[y].mean(), 0.7)
        self.assertLess(prior[~y].mean(), 0.02)
        self.assertEqual([i["how"] for i in info], ["side", "side"])


if __name__ == "__main__":
    unittest.main()
