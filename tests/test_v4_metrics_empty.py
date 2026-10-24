"""Empty-prediction HD95 penalty (user decision 2026-10-24): pred empty + GT non-empty ->
the case's physical diagonal (never NaN, so nanmean cannot drop it); Empty_pred_Y flag exposed;
empty GT keeps NaN semantics."""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "MIA" / "scripts"))
sys.path.insert(0, str(REPO))
from metrics_protocol import _hd95, score_protocol  # noqa: E402


class TestEmptyPredPenalty(unittest.TestCase):
    def test_empty_pred_returns_diagonal(self):
        gt = np.zeros((4, 5, 6), bool)
        gt[2, 2, 2] = True
        sp = (2.5, 1.0, 1.0)
        want = float(np.sqrt((4 * 2.5) ** 2 + (5 * 1.0) ** 2 + (6 * 1.0) ** 2))
        self.assertAlmostEqual(_hd95(np.zeros_like(gt), gt, sp), want)

    def test_empty_pred_same_for_any_all_background_pred(self):
        gt = np.zeros((3, 3, 3), bool)
        gt[0, 0, 0] = True
        self.assertAlmostEqual(_hd95(np.zeros_like(gt), gt, (1.0, 1.0, 1.0)),
                               float(np.sqrt(27.0)))

    def test_empty_gt_still_nan(self):
        self.assertTrue(np.isnan(_hd95(np.ones((4, 5, 6), bool), np.zeros((4, 5, 6), bool), (1, 1, 1))))

    def test_score_protocol_flags_empty(self):
        gt = np.zeros((5, 5, 5), np.uint8)
        gt[2, 2, 2] = 2                     # a Y-class voxel
        m = score_protocol(np.zeros_like(gt), gt, spacing_zyx=(1.0, 1.0, 1.0), pred_binary_Y=True)
        self.assertEqual(m["Empty_pred_Y"], 1.0)
        self.assertAlmostEqual(m["HD95_Y"], float(np.sqrt(3 * 25.0)))
        self.assertEqual(m["Dice_Y"], 0.0)

    def test_empty_gt_not_counted_as_empty_pred(self):
        gt = np.zeros((5, 5, 5), np.uint8)          # GT itself empty: not an empty-pred failure
        m = score_protocol(np.zeros_like(gt), gt, spacing_zyx=(1.0, 1.0, 1.0), pred_binary_Y=True)
        self.assertEqual(m["Empty_pred_Y"], 0.0)
        self.assertTrue(np.isnan(m["HD95_Y"]))

    def test_score_protocol_flags_nonempty_zero(self):
        gt = np.zeros((5, 5, 5), np.uint8)
        gt[2, 2, 2] = 2
        pred = np.zeros_like(gt)
        pred[1:4, 1:4, 1:4] = 1             # non-empty (wrong everywhere, but non-empty)
        m = score_protocol(pred, gt, spacing_zyx=(1.0, 1.0, 1.0), pred_binary_Y=True)
        self.assertEqual(m["Empty_pred_Y"], 0.0)


if __name__ == "__main__":
    unittest.main()
