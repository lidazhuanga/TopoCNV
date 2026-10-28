"""V4.3 (V4_PLAN §12): dropout geometry, input masking, MM-shuf placement, selection rule, guards, jobs.
Trainer tests need the nnU-Net venv with tools/v4_mm43.py install43 done; the rest run in either env."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from trainer.v4.splits import internal_split  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


N = _load("v4_nnunet")
M = _load("v4_mm43")
S43 = _load("v4_select43")
J = _load("v4_jobs")
try:
    import torch
    from nnunetv2.training.nnUNetTrainer.variants.zz_cnv_v4 import nnUNetTrainerV43 as T
except Exception:          # topocnv env, or the trainer is not installed
    T = None


@unittest.skipIf(T is None, "needs the nnU-Net venv with the V4.3 trainer installed")
class TestDropout(unittest.TestCase):
    SH = (56, 192, 192)
    SP = (2.5, 1.1328, 1.1328)

    def test_case_and_untilted_truncation(self):
        self.assertFalse(T.keep_mask({"case": True}, self.SH, self.SP).any())
        for side in ("cranial", "caudal"):
            d = {"case": False, "trunc": {"z0": 20.5, "tilt": 0.0, "az": 0.0, "side": side}}
            k = T.keep_mask(d, self.SH, self.SP)
            kept = k.reshape(self.SH[0], -1).all(1).numpy()
            dropped = (~k.bool()).reshape(self.SH[0], -1).all(1).numpy()
            if side == "cranial":       # axis 0 increases cranially: slices above z0 dropped
                self.assertTrue(kept[:21].all() and dropped[21:].all())
            else:
                self.assertTrue(dropped[:21].all() and kept[21:].all())

    def test_tilted_plane(self):
        d = {"case": False, "trunc": {"z0": 28.0, "tilt": float(np.deg2rad(20)), "az": 0.0, "side": "cranial"}}
        k = T.keep_mask(d, self.SH, self.SP).numpy()
        # tilt about axis 2 (azimuth 0): the cut height moves along axis 1, not along axis 2
        top = [int(np.argmin(k[:, a, 96])) for a in (0, 191)]
        self.assertNotEqual(top[0], top[1])
        self.assertTrue((k[:, 50, :] == k[:, 50, :1]).all())
        frac = 1 - k.mean()
        self.assertTrue(0.4 < frac < 0.6, frac)

    def test_slabs(self):
        d = {"case": False, "slabs": [(3, 9), (40, 50)]}
        k = T.keep_mask(d, self.SH, self.SP).numpy()
        z = k.reshape(self.SH[0], -1).max(1)
        self.assertEqual(sorted(np.nonzero(z == 0)[0].tolist()), list(range(3, 9)) + list(range(40, 50)))

    def test_draw_frequencies(self):
        rng = np.random.default_rng(0)
        n, pc, pt, ps = 20000, 0.4, 0.5, 0.3
        ds = [T.draw_drop(rng, self.SH, pc, pt, ps) for _ in range(n)]
        case = np.mean([d["case"] for d in ds])
        rest = [d for d in ds if not d["case"]]
        tr = np.mean(["trunc" in d for d in rest])
        sl = np.mean(["slabs" in d for d in rest])
        self.assertAlmostEqual(case, pc, delta=0.015)
        self.assertAlmostEqual(tr, pt, delta=0.02)
        self.assertAlmostEqual(sl, ps, delta=0.02)
        self.assertTrue(all("trunc" not in d and "slabs" not in d for d in ds if d["case"]))
        sides = np.mean([d["trunc"]["side"] == "cranial" for d in rest if "trunc" in d])
        self.assertAlmostEqual(sides, 0.5, delta=0.03)
        tilts = [d["trunc"]["tilt"] for d in rest if "trunc" in d]
        self.assertTrue(0 <= min(tilts) and max(tilts) <= np.deg2rad(T.TILT_MAX_DEG))
        for d in rest:
            for s, e in d.get("slabs", ()):
                self.assertTrue(4 <= e - s <= 10 and 0 <= s and e <= self.SH[0])
            self.assertIn(len(d.get("slabs", [0])), (1, 2))

    def test_candidates_fixed(self):
        self.assertEqual(T.CANDIDATES, {"MMd1": (0.2, 0.3, 0.2), "MMd2": (0.4, 0.5, 0.3)})
        for name, (pc, pt, ps) in (("nnUNetTrainerV43_MMd1", (0.2, 0.3, 0.2)), ("nnUNetTrainerV43_MMd2", (0.4, 0.5, 0.3)),
                                   ("nnUNetTrainerV43_MMdcase1", (0.2, 0, 0)), ("nnUNetTrainerV43_MMdcase2", (0.4, 0, 0)),
                                   ("nnUNetTrainerV43_MM", (0, 0, 0))):
            c = getattr(T, name)
            self.assertEqual((c.P_CASE, c.P_TRUNC, c.P_SLAB), (pc, pt, ps), name)
            self.assertEqual((c.PRIORS, c.VIS, c.SKEL_W), ((), None, 0.0), name)


@unittest.skipIf(T is None, "needs the nnU-Net venv with the V4.3 trainer installed")
class TestMasking(unittest.TestCase):
    def _trainer(self, cls):
        t = object.__new__(cls)
        t.zero_channels, t.drop_spacing = [], [2.5, 1.1328, 1.1328]
        t.drop_rng = np.random.default_rng(1)
        t._dropping = t._no_mri = False
        t.drop_counts = {"samples": 0, "case": 0, "trunc": 0, "slab": 0}
        return t

    def _batch(self):
        g = torch.Generator().manual_seed(0)
        x = torch.rand((2, 3, 16, 24, 24), generator=g)
        x[:, 2] = x[:, 2] * 1.2 - 0.1        # mask after spatial / intensity-free DA: not exactly 0/1
        return x

    def test_mask_binarised_and_applied(self):
        t = self._trainer(T.nnUNetTrainerV43_MM)
        x = self._batch()
        y = t._mask(x)
        m = (x[:, 2] > 0.5).float()
        self.assertTrue(torch.equal(y[:, 2], m))
        self.assertTrue(torch.equal(y[:, 1], x[:, 1] * m))
        self.assertTrue(torch.equal(y[:, 0], x[:, 0]))
        t._no_mri = True
        self.assertFalse(t._mask(x)[:, 1:].any())
        with self.assertRaises(AssertionError):
            t._mask(x[:, :2])

    def test_dropout_only_while_training(self):
        t = self._trainer(T.nnUNetTrainerV43_SMOKE_MMd)
        x = self._batch()
        base = t._mask(x)                       # validation / inference: no dropout
        self.assertEqual(t.drop_counts["samples"], 0)
        t._dropping = True
        for _ in range(30):
            y = t._mask(x)
            self.assertTrue(torch.equal(y[:, 0], x[:, 0]))                       # CT never touched
            self.assertTrue(((y[:, 1:] == base[:, 1:]) | (y[:, 1:] == 0)).all())  # only zeroed
            self.assertTrue(torch.equal(y[:, 1] != 0, (y[:, 1] != 0) & (y[:, 2] > 0)))   # MRI only where mask
        self.assertEqual(t.drop_counts["samples"], 60)
        self.assertTrue(all(t.drop_counts[k] > 0 for k in ("case", "trunc", "slab")))
        t2 = self._trainer(T.nnUNetTrainerV43_MM)
        t2._dropping = True
        self.assertTrue(torch.equal(t2._mask(x), base))                         # MM: no dropout at all

    def test_intensity_wrapper_skips_mask(self):
        seen = {}

        class Inner:
            def __call__(self, **d):
                seen["c"] = d["image"].shape[0]
                d["image"] = d["image"] * 0 + 7
                return d

        w = T.FirstChannels(Inner(), 2)
        img = torch.rand((3, 4, 4, 4))
        out = w(image=img.clone(), segmentation=None)["image"]
        self.assertEqual(seen["c"], 2)
        self.assertTrue((out[:2] == 7).all() and torch.equal(out[2], img[2]))


class TestShuffle(unittest.TestCase):
    def test_derangement(self):
        cases = [f"c{i:03d}" for i in range(153)]
        d = M.derangement(cases)
        self.assertEqual(sorted(d), sorted(cases))
        self.assertEqual(sorted(d.values()), sorted(cases))
        self.assertTrue(all(k != v for k, v in d.items()))
        self.assertEqual(d, M.derangement(list(reversed(cases))))       # deterministic, order-free

    def test_place(self):
        a = np.arange(2 * 3 * 4).reshape(2, 3, 4) + 1
        out = M.place(a, [1, -1, 2], (3, 3, 4))
        self.assertEqual(out[1, 0, 2], a[0, 1, 0])
        self.assertEqual(out[2, 1, 3], a[1, 2, 1])
        self.assertEqual(int((out > 0).sum()), 2 * 2 * 2)
        self.assertFalse(M.place(a, [5, 0, 0], (3, 3, 4)).any())
        m = np.zeros((10, 10, 10), np.uint8)
        m[2:4, 2:4, 2:4] = 1
        c = M.centroid(m)
        self.assertTrue(np.allclose(c, [2.5, 2.5, 2.5]))
        self.assertIsNone(M.centroid(m * 0))

    def test_widen_plans(self):
        p = N.NNUNET_HOME / "nnUNet_preprocessed" / N.V42["binY"][1] / f"{N.PLANS}.json"
        if not p.exists():
            self.skipTest("Dataset540 plans not on this host")
        w = M.widen_plans3(p, "DatasetX")
        c = w["configurations"]["3d_fullres"]
        self.assertEqual(c["normalization_schemes"], ["NoNormalization"] * 3)
        self.assertEqual(c["patch_size"], json.loads(p.read_text())["configurations"]["3d_fullres"]["patch_size"])
        self.assertEqual(w["dataset_name"], "DatasetX")


class TestSelect(unittest.TestCase):
    O = ["v43_mmd1", "v43_mmd2"]

    def pick(self, w1, w2, n1, n2, r=0.70):
        return S43.choose43({"v43_mmd1": w1, "v43_mmd2": w2}, {"v43_mmd1": n1, "v43_mmd2": n2}, r, self.O)

    def test_rule(self):
        self.assertEqual(self.pick(0.72, 0.7229, 0.70, 0.70)[0], "v43_mmd1")      # both eligible, < 0.003 -> MMd-1
        self.assertEqual(self.pick(0.72, 0.7230, 0.70, 0.70)[0], "v43_mmd2")      # exactly 0.003 is not < 0.003
        self.assertEqual(self.pick(0.75, 0.72, 0.68, 0.70)[0], "v43_mmd2")        # MMd-1 not eligible (0.68 < 0.69)
        c, tr = self.pick(0.75, 0.72, 0.69, 0.60)
        self.assertEqual((c, tr["eligible"]), ("v43_mmd1", ["v43_mmd1"]))         # exactly R - 0.01 is eligible
        c, tr = self.pick(0.75, 0.74, 0.60, 0.65)
        self.assertEqual((c, tr["robustness_met"]), ("v43_mmd2", False))          # none eligible: higher S_none
        c, tr = self.pick(0.75, 0.74, 0.65, 0.65)
        self.assertEqual((c, tr["robustness_met"]), ("v43_mmd1", False))          # tie -> MMd-1
        self.assertTrue(self.pick(0.72, 0.73, 0.70, 0.70)[1]["robustness_met"])
        with self.assertRaises(AssertionError):
            self.pick(float("nan"), 0.7, 0.7, 0.7)

    def test_candidates(self):
        self.assertEqual([c[2] for c in S43.CANDS], ["nnUNetTrainerV43_MMd1", "nnUNetTrainerV43_MMd2"])
        self.assertEqual(S43.R0U, ("v42_r0u", N.R0_TRAINER))


class TestGuards(unittest.TestCase):
    def test_check_trainer(self):
        with self.assertRaises(SystemExit):
            M.check_trainer("mm", 0, "dev", "nnUNetTrainerV43_MM")          # MM is not a dev candidate
        with self.assertRaises(SystemExit):
            M.check_trainer("shuf", 0, "dev", "nnUNetTrainerV43_MMd1")
        with self.assertRaises(SystemExit):
            M.check_trainer("mm", 0, "dev", "nnUNetTrainer_300epochs")
        self.assertIsNone(M.check_trainer("mm", 0, "dev", "nnUNetTrainerV43_MMd2"))
        with tempfile.TemporaryDirectory() as d:
            old = M.FROZEN43
            M.FROZEN43 = Path(d)
            try:
                with self.assertRaises(SystemExit):                          # report before the freeze
                    M.check_trainer("mm", 0, "report", "nnUNetTrainerV43_MMd1")
                (Path(d) / "fold0.json").write_text(json.dumps({"chosen": {"trainer": "nnUNetTrainerV43_MMd2"}}))
                for what, tr in (("mm", "nnUNetTrainerV43_MMd2"), ("mm", "nnUNetTrainerV43_MMdcase2"),
                                 ("mm", "nnUNetTrainerV43_MM"), ("shuf", "nnUNetTrainerV43_MM")):
                    self.assertTrue(M.check_trainer(what, 0, "report", tr))
                for what, tr in (("mm", "nnUNetTrainerV43_MMd1"), ("mm", "nnUNetTrainerV43_MMdcase1"),
                                 ("shuf", "nnUNetTrainerV43_MMd2")):
                    with self.assertRaises(SystemExit):
                        M.check_trainer(what, 0, "report", tr)
            finally:
                M.FROZEN43 = old

    def test_mri_sums(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "cache").mkdir()
            old = (M.MRI_ROOT, M.MRI_CACHE)
            M.MRI_ROOT, M.MRI_CACHE = root, root / "cache"
            try:
                (root / "cache" / "a_mri.npy").write_bytes(b"x")
                (root / "qa_summary.csv").write_bytes(b"y")
                with self.assertRaises(SystemExit):
                    M.mri_sums()                                             # not frozen
                (root / "cache" / "FROZEN").write_text("frozen")
                (root / "cache" / "SHA256SUMS").write_text(
                    f"{M._sha_file(root / 'cache' / 'a_mri.npy')}  a_mri.npy\n{M._sha_file(root / 'qa_summary.csv')}  qa_summary.csv\n")
                sha, sums = M.mri_sums()
                self.assertEqual(sorted(sums), ["a_mri.npy", "qa_summary.csv"])
                self.assertTrue(all(M._check_one(x)[1] for x in sums.items()))
                (root / "qa_summary.csv").write_bytes(b"z")
                self.assertFalse(M._check_one(("qa_summary.csv", sums["qa_summary.csv"]))[1])
            finally:
                M.MRI_ROOT, M.MRI_CACHE = old


class TestJobs(unittest.TestCase):
    def test_mm43_chain(self):
        jobs = [{"key": "nnprep42_binY"}, {"key": "smoke42p"}, {"key": "nn42_v42_d2ab_dev_f0"}]
        J.mm43(jobs, [0, 1], "smoke42p")
        keys = [j["key"] for j in jobs]
        self.assertEqual(keys[:2], ["nnprep42_binY", "smoke42p"])
        self.assertEqual(keys[-1], "nn42_v42_d2ab_dev_f0")                    # D2' prior arm stays behind
        self.assertLess(keys.index("nn43_v43_mmd2_dev_f1"), keys.index("nn42_v42_d2ab_dev_f0"))
        by = {j["key"]: j for j in jobs}
        self.assertEqual(sorted(by["nn43_v43_mmd1_dev_f0"]["deps"]), ["nnprep43_mm", "smoke43"])
        self.assertIn("sc42_v42_r0u_f1", by["sel43_f1"]["deps"])
        self.assertEqual(len([k for k in keys if k.startswith("sc43_") and k.endswith("_f0")]), 4)
        self.assertEqual(by["sc43_v43_mmd2_nomri_f0"]["done"], "results/v4_dev_eval_harm_v1/v43_mmd2_nomri/fold0.csv")
        self.assertEqual(by["sc43_v43_mmd2_nomri_f0"]["deps"], ["nn43_v43_mmd2_dev_f0"])
        self.assertTrue(all(by[k]["kind"] == "gpu" for k in keys if k.startswith("nn43_") or k == "smoke43"))
        cmd = by["nn43_v43_mmd1_dev_f1"]["cmd"]
        self.assertEqual(cmd[cmd.index("--trainer") + 1], "nnUNetTrainerV43_MMd1")
        with self.assertRaises(SystemExit):
            J.mm43([], [0], None)

    def test_dev_cases(self):
        tr, v = internal_split(0)
        self.assertFalse(set(tr) & set(v))


if __name__ == "__main__":
    unittest.main()
