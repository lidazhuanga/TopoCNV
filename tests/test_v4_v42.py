"""V4.2 dataset helpers in tools/v4_nnunet.py (label maps, split tables); runs in either env."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainer.v4.splits import (N_INNER, all_cases, inner_folds, internal_split, report_cases,  # noqa: E402
                               stage1_pool, train_cases)

_spec = importlib.util.spec_from_file_location("v4_nnunet", ROOT / "tools" / "v4_nnunet.py")
N = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(N)


class TestV42(unittest.TestCase):
    def test_labels(self):
        lab = np.arange(6, dtype=np.uint8).reshape(1, 2, 3)
        np.testing.assert_array_equal(N.v42_label(lab, "binY").ravel(), [0, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(N.v42_label(lab, "s1").ravel(), [0, 1, 2, 3, 4, 0])
        with self.assertRaises(AssertionError):
            N.v42_label(np.full((1, 1, 1), 6, np.uint8), "binY")

    def test_binY_splits(self):
        s = N.v42_splits("binY")
        self.assertEqual(len(s), 10)
        for k in range(5):
            rep, dev = s[N.v42_split_index("binY", k, "report")], s[N.v42_split_index("binY", k, "dev")]
            self.assertEqual(rep["val"], report_cases(k))
            self.assertEqual(sorted(rep["train"]), sorted(train_cases(k)))
            tr, iv = internal_split(k)
            self.assertEqual(dev["val"], iv)
            self.assertEqual(dev["train"], tr)
            self.assertFalse(set(dev["train"]) & set(iv))
            self.assertFalse(set(dev["train"]) & set(report_cases(k)))   # report fold never in dev training

    def test_s1_splits(self):
        s = N.v42_splits("s1")
        self.assertEqual(len(s), 2 * 5 * N_INNER)
        seen = set()
        for tier in ("dev", "report"):
            for k in range(5):
                held = set(report_cases(k)) | (set(internal_split(k)[1]) if tier == "dev" else set())
                for j in range(N_INNER):
                    i = N.v42_split_index("s1", k, tier, j)
                    seen.add(i)
                    e = s[i]
                    self.assertEqual(e["val"], inner_folds(k, tier=tier)[j])
                    self.assertEqual(sorted(e["train"] + e["val"]), sorted(stage1_pool(k, tier)))
                    self.assertFalse(set(e["train"]) & set(e["val"]))
                    self.assertFalse((set(e["train"]) | set(e["val"])) & held)  # V_k / report fold held out
        self.assertEqual(seen, set(range(2 * 5 * N_INNER)))
        with self.assertRaises(AssertionError):
            N.v42_split_index("s1", 0, "dev", None)

    def test_ids(self):
        self.assertEqual({d for d, _ in N.V42.values()}, {540, 541})
        self.assertTrue(all(n.startswith(f"Dataset{d}_") for d, n in N.V42.values()))
        self.assertEqual(len(all_cases()), 154)


if __name__ == "__main__":
    unittest.main()
