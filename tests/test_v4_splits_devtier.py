"""v4 splits, dev tier: Stage-1 inner folds live inside T_k - V_k and never touch V_k (nested
selection, V4_PLAN §5 as revised 2026-10-24). Report-tier behaviour must stay unchanged."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topocnv.data import v4_splits as S  # noqa: E402


class TestDevTier(unittest.TestCase):
    def test_dev_inner_folds_partition_Tk_minus_Vk(self):
        for k in range(5):
            dev_tr, v = S.internal_split(k)
            vset, trset = set(v), set(dev_tr)
            inner = S.inner_folds(k, tier="dev")
            flat = sorted(c for f in inner for c in f)
            self.assertEqual(flat, sorted(dev_tr), "dev inner folds must partition T_k - V_k")
            self.assertFalse(vset & set(flat))
            for j in range(S.N_INNER):
                tr_j = set(S.stage1_train_cases(k, j, tier="dev"))
                self.assertFalse(tr_j & set(inner[j]))
                self.assertFalse(tr_j & vset, "no dev Stage-1 model may train on V_k")
                self.assertFalse(tr_j & set(S.report_cases(k)))
                self.assertEqual(tr_j | set(inner[j]), trset)
                for c in inner[j]:
                    self.assertEqual(S.stage1_source(k, c, tier="dev"), ("inner", j))

    def test_dev_Vk_gets_ensemble(self):
        for k in range(5):
            for c in S.internal_split(k)[1]:
                self.assertEqual(S.stage1_source(k, c, tier="dev"), ("ensemble", None))
            for c in S.report_cases(k):
                self.assertEqual(S.stage1_source(k, c, tier="dev"), ("ensemble", None))

    def test_report_tier_unchanged(self):
        for k in range(5):
            T = sorted(S.train_cases(k))
            self.assertEqual(sorted(c for f in S.inner_folds(k) for c in f), T)
            self.assertEqual(sorted(c for f in S.inner_folds(k, tier="report") for c in f), T)
            for j in range(S.N_INNER):
                self.assertEqual(S.stage1_train_cases(k, j, tier="report"), S.stage1_train_cases(k, j))
            for c in T:
                self.assertEqual(S.stage1_source(k, c), S.stage1_source(k, c, tier="report"))
            rep = set(S.report_cases(k))
            dev_tr = set(S.internal_split(k)[0])
            for c in rep:
                self.assertEqual(S.stage1_source(k, c), ("ensemble", None))
            self.assertFalse(set(S.inner_folds(k, tier="dev")[0]) & rep)

    def test_invalid_tier_raises(self):
        with self.assertRaises(ValueError):
            S.inner_folds(0, tier="bogus")


if __name__ == "__main__":
    unittest.main()
