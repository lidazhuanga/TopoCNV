import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topocnv.data.harmonize import harmonise, run_length, thickness  # noqa: E402

HAS_SKIMAGE = importlib.util.find_spec("skimage") is not None


def thin_line():
    """1-voxel-wide nerve along axis 2 on slices 1-4: class 2 for q < 20, class 3 for q >= 20."""
    lab = np.zeros((6, 40, 40), np.uint8)
    lab[1:5, 20, 10:20] = 2
    lab[1:5, 20, 20:31] = 3
    lab[2, 21, 15] = 5   # muscle bypass next to the nerve: must never become Y
    lab[3, 21, 12] = 4   # z-neighbour of candidate (2, 21, 12): ties in voxel units, not in mm
    return lab


class TestHarmonise(unittest.TestCase):
    def test_growth_rule(self):
        lab = thin_line()
        out = harmonise(lab, 0.20)
        y0 = (lab >= 1) & (lab <= 4)
        np.testing.assert_array_equal(out[y0], lab[y0])          # original voxels unchanged
        self.assertEqual(out[2, 21, 15], 5)                      # class 5 kept
        added = ((out >= 1) & (out <= 4)) & ~y0
        z, r, q = np.nonzero(added)
        self.assertTrue(set(z) <= {1, 2, 3, 4})                  # no growth along z
        self.assertTrue(set(r) <= {19, 21})                      # one in-plane ring only
        self.assertEqual(out[2, 19, 15], 2)
        self.assertEqual(out[2, 19, 25], 3)
        self.assertEqual(out[2, 21, 16], 2)
        self.assertEqual(out[2, 21, 12], 2)                      # nearest in mm: in-plane class 2, not z class 4
        self.assertEqual(out[2, 21, 10], 0)                      # line end: smoothed Y below t
        self.assertEqual(out[2, 20, 9], 0)                       # no growth past the tip

    def test_threshold_and_empty(self):
        lab = thin_line()
        lab[3, 21, 12] = 0
        np.testing.assert_array_equal(harmonise(lab, 0.30), lab)  # next to a straight line: ~0.228 < 0.30
        empty = np.zeros((4, 8, 8), np.uint8)
        np.testing.assert_array_equal(harmonise(empty, 0.2), empty)

    def test_run_length(self):
        y = np.zeros((3, 10, 10), bool)
        y[1, 2:4, 3:6] = True
        self.assertAlmostEqual(run_length(y, 2), 3.0)
        self.assertAlmostEqual(run_length(y, 1), 2.0)
        self.assertAlmostEqual(run_length(y, 0), 1.0)

    @unittest.skipUnless(HAS_SKIMAGE, "scikit-image only in the nnU-Net venv")
    def test_thickness_grows(self):
        lab = thin_line()
        self.assertLess(thickness(lab), thickness(harmonise(lab, 0.20)))


if __name__ == "__main__":
    unittest.main()
