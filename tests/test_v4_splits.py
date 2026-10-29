"""v4 splits: report folds = extras/splits.json; V_k, inner folds and report fold never overlap."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topocnv.data import v4_splits as S  # noqa: E402


class TestSplits(unittest.TestCase):
    def test_report_folds_match_json(self):
        js = json.loads(S.SPLITS_PATH.read_text())["folds"]
        self.assertEqual([sorted(f) for f in js], S.report_folds())
        self.assertEqual(len(S.all_cases()), 154)

    def test_internal_and_inner(self):
        for k in range(5):
            rep = set(S.report_cases(k))
            T = set(S.train_cases(k))
            tr, iv = S.internal_split(k)
            self.assertFalse(rep & T)
            self.assertEqual(set(tr) | set(iv), T)
            self.assertFalse(set(tr) & set(iv))
            self.assertEqual(len(iv), round(0.15 * len(T)))
            inner = S.inner_folds(k)
            self.assertEqual(sorted(c for f in inner for c in f), sorted(T))
            for j in range(S.N_INNER):
                s1 = set(S.stage1_train_cases(k, j))
                self.assertFalse(s1 & rep, "Stage 1 must never train on the report fold")
                self.assertFalse(s1 & set(inner[j]))
                for c in inner[j]:
                    self.assertEqual(S.stage1_source(k, c), ("inner", j))
            for c in rep:
                self.assertEqual(S.stage1_source(k, c), ("ensemble", None))
            self.assertEqual(S.internal_split(k), (tr, iv), "deterministic")


if __name__ == "__main__":
    unittest.main()
