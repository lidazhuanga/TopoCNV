"""V4-AM-C3W-01 (eval-only class-3 weighting): wDSC_Y(lam) degenerates to Dice_Y at lam=1,
discounts only GT==3 misses, never discounts false positives; Recall_GT3 coverage; empty-case
conventions match _dice."""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "MIA" / "scripts"))
sys.path.insert(0, str(REPO))
from topocnv.metrics.protocol import _dice, _recall_gt3, _wdice, mask_Y, score_protocol  # noqa: E402
import math  # noqa: E402


def case_labels():
    gt = np.zeros((8, 8, 8), np.uint8)
    gt[:2, :2, :2] = 1          # class 1 block
    gt[2:4, :2, :2] = 3         # class 3 block (probable course)
    return gt


class TestWDice(unittest.TestCase):
    def setUp(self):
        self.gt = case_labels()

    def test_lam1_equals_standard_dice(self):
        pred = np.zeros_like(self.gt)
        pred[:3, :2, :2] = 1     # overlaps class1 fully, class3 partially
        for p in (pred, np.zeros_like(pred), np.ones_like(pred)):
            self.assertAlmostEqual(_wdice(p, self.gt, 1.0), _dice(p > 0, mask_Y(self.gt)))

    def test_class3_miss_is_discounted(self):
        perfect = np.isin(self.gt, [1, 2, 3, 4])
        only_c1 = perfect.copy()
        only_c1[self.gt == 3] = False            # miss all of class 3
        d1 = _dice(only_c1, mask_Y(self.gt))
        w05 = _wdice(only_c1, self.gt, 0.5)
        self.assertGreater(w05, d1)              # discounted miss scores higher than standard
        self.assertLess(w05, 1.0)

    def test_false_positive_never_discounted(self):
        # GT with ONLY class-3 voxels; prediction covers them all + equal-sized background FP.
        # If FP were wrongly weighted, wDSC would be lambda-independent (2/3); the correct
        # closed form is 2*4*lam / ((4*lam + 4) + 4*lam).
        gt = np.zeros((4, 4, 4), np.uint8)
        gt[0, 0, :4] = 3
        pred = np.zeros_like(gt)
        pred[0, 0, :4] = 1
        pred[2, 2, :4] = 1                      # 4 background FPs, weight must stay 1
        for lam in (0.25, 0.5, 0.75):
            self.assertAlmostEqual(_wdice(pred, gt, lam), 2 * 4 * lam / ((4 * lam + 4) + 4 * lam))

    def test_multiclass_path_and_class5(self):
        # score_protocol with the raw 0-5 label array as prediction (pred_binary_Y=False):
        # perfect multiclass prediction -> Dice_Y == wDSC == Recall_GT3 == 1; a class-5-only
        # prediction is empty w.r.t. Y on both the standard and the weighted view
        gt = case_labels()
        gt[4:6, :2, :2] = 5
        m = score_protocol(gt.copy(), gt, spacing_zyx=(1.0, 1.0, 1.0), pred_binary_Y=False)
        for k in ("Dice_Y", "wDSC_Y_0.25", "wDSC_Y_0.50", "wDSC_Y_0.75", "Recall_GT3"):
            self.assertAlmostEqual(m[k], 1.0, msg=k)
        only5 = np.zeros_like(gt)
        only5[4:6, :2, :2] = 5
        m2 = score_protocol(only5, gt, spacing_zyx=(1.0, 1.0, 1.0), pred_binary_Y=False)
        self.assertEqual(m2["Dice_Y"], 0.0)
        self.assertEqual(m2["wDSC_Y_0.50"], 0.0)
        self.assertTrue(math.isnan(m2["Recall_GT3"]) is False)   # GT has class 3 -> defined
        self.assertEqual(m2["Recall_GT3"], 0.0)

    def test_perfect_prediction_scores_one(self):
        perfect = np.isin(self.gt, [1, 2, 3, 4])
        for lam in (0.25, 0.5, 0.75):
            self.assertAlmostEqual(_wdice(perfect, self.gt, lam), 1.0)

    def test_empty_conventions_match_dice(self):
        empty = np.zeros_like(self.gt)
        self.assertEqual(_wdice(empty, self.gt, 0.5), 0.0)
        gt_empty = np.zeros_like(self.gt)
        self.assertTrue(np.isnan(_wdice(empty, gt_empty, 0.5)))

    def test_score_protocol_exposes_columns(self):
        pred = np.isin(self.gt, [1, 2, 3, 4])
        m = score_protocol(pred, self.gt, spacing_zyx=(1.0, 1.0, 1.0), pred_binary_Y=True)
        for col in ("wDSC_Y_0.25", "wDSC_Y_0.50", "wDSC_Y_0.75", "Recall_GT3"):
            self.assertIn(col, m)
        self.assertAlmostEqual(m["wDSC_Y_0.50"], 1.0)
        self.assertAlmostEqual(m["Recall_GT3"], 1.0)

    def test_recall_gt3(self):
        self.assertAlmostEqual(_recall_gt3(np.isin(self.gt, [1, 2, 3, 4]), self.gt), 1.0)
        only_c1 = np.isin(self.gt, [1, 2, 3, 4]) & (self.gt != 3)
        self.assertAlmostEqual(_recall_gt3(only_c1, self.gt), 0.0)
        self.assertTrue(np.isnan(_recall_gt3(np.ones_like(self.gt), np.zeros_like(self.gt))))


if __name__ == "__main__":
    unittest.main()
