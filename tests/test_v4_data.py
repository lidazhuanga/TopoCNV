"""v4 data: an R0 sample equals the SOTA CachedSOTADataset sample for the same RNG state (real cache)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topocnv.data.utils import CACHE_DIR  # noqa: E402
from topocnv.data.v4_dataset import V4PatchDataset, stage1_target  # noqa: E402
from topocnv.data.v4_splits import all_cases  # noqa: E402


class TestData(unittest.TestCase):
    def test_r0_matches_sota_dataset(self):
        from topocnv.data.dataset import CachedSOTADataset
        cases = all_cases()[:3]
        sota = CachedSOTADataset([(c, "", "") for c in cases], patch_size=(128, 128, 128),
                                 cache_dir=str(CACHE_DIR), training=True, random_crop_prob=0.33)
        ours = V4PatchDataset(cases, (128, 128, 128), stage=2, fold=0, cfg={}, random_crop_prob=0.33)
        for seed in range(6):
            np.random.seed(seed)
            xs, ys = sota[seed % 3]
            np.random.seed(seed)
            d = ours[seed % 3]
            np.testing.assert_array_equal(xs.numpy(), d["x"].numpy())
            np.testing.assert_array_equal(ys.numpy(), d["label"].numpy())
            self.assertTrue(bool(d["roi"].all()))

    def test_stage1_target(self):
        raw = np.arange(6).reshape(1, 1, 6)
        np.testing.assert_array_equal(stage1_target(raw)[0, 0], [0, 1, 0, 0, 2, 0])


if __name__ == "__main__":
    unittest.main()
