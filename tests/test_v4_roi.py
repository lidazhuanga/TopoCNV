"""v4 ROI: far false positives removed, mirroring, margins in mm, coverage."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topocnv.data.utils import SPACING  # noqa: E402
from topocnv.priors.roi import clean_cluster, compute_case_roi, coverage, margin_vox, roi_boxes  # noqa: E402


def synthetic():
    shape = (60, 120, 100)  # far FP at x 95-99 is > 45 mm from the side-0 cluster
    image = np.zeros(shape, np.float32)
    image[:, 20:100, 10:90] = 0.6  # body, LR centre 59.5
    lm = np.zeros(shape, bool)
    lm[28:32, 35:40, 45:50] = True   # side 0 landmark
    lm[28:32, 80:85, 45:50] = True   # side 1 landmark
    lm[29:32, 37:40, 58:61] = True   # side 0 second component (27 vox), ~9 mm away -> kept
    lm[2:5, 25:28, 95:100] = True    # far false positive (> 45 mm) -> dropped
    return image, lm


class TestROI(unittest.TestCase):
    def test_clean_and_boxes(self):
        image, lm = synthetic()
        rec, clean, c4 = compute_case_roi(lm, lm, image)
        self.assertFalse(clean[2:5, 25:28, 95:100].any())
        self.assertTrue(clean[29:32, 37:40, 58:61].all())
        self.assertEqual([s["source"] for s in rec["sides"]], ["own", "own"])
        self.assertEqual(rec["sides"][0]["bbox"], [[28, 32], [35, 40], [45, 61]])
        mz, my, mx = margin_vox(15)
        self.assertEqual(mz, 6)
        self.assertEqual(rec["sides"][0]["boxes"]["15"], [[22, 38], [35 - my, 40 + my], [45 - mx, 61 + mx]])

    def test_mirror(self):
        image, lm = synthetic()
        lm[:, 60:] = False
        rec, _, _ = compute_case_roi(lm, lm, image)
        self.assertEqual(rec["sides"][1]["source"], "mirror")
        lo, hi = rec["sides"][1]["bbox"][1]
        self.assertLessEqual(abs((lo + hi - 1) / 2 - (2 * rec["midline"] - (35 + 39) / 2)), 1.0)

    def test_coverage(self):
        image, lm = synthetic()
        rec, _, _ = compute_case_roi(lm, lm, image)
        gt = np.zeros(lm.shape, bool)
        gt[30, 36:38, 46:48] = True
        gt[30, 2, 2] = True  # one voxel far outside
        cov, vol = coverage(rec, 15, gt)
        self.assertAlmostEqual(cov, 4 / 5)
        self.assertLess(vol, 1.0)
        self.assertEqual(len(roi_boxes(rec, 25)), 2)

    def test_single_component(self):
        m = np.zeros((10, 10, 10), bool)
        m[2:5, 2:5, 2:5] = True  # 27 voxels >= MIN_VOX
        self.assertTrue((clean_cluster(m, SPACING) == m).all())


if __name__ == "__main__":
    unittest.main()
