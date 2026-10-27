"""V4.2 D2' chain (V4_PLAN §11.6.1-11.6.2): selector rule, D1-AM2 rule, prior pieces, datasets, jobs. Either env."""
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from trainer.v4.splits import N_INNER, internal_split  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


N = _load("v4_nnunet")
B = _load("v4_build_priors42")
S = _load("v4_select42")
J = _load("v4_jobs")


class TestRule(unittest.TestCase):
    def test_choose(self):
        lv = {"B": 0, "ab": 1, "ac": 1, "bc": 1, "abc": 2}
        order = ["B", "ab", "ac", "bc", "abc"]
        pick = lambda sc: S.choose(dict(zip(order, sc)), lv, order)[0]  # noqa: E731
        self.assertEqual(pick([0.60, 0.61, 0.62, 0.63, 0.64]), "abc")           # clear winner
        self.assertEqual(pick([0.6375, 0.62, 0.63, 0.6380, 0.6400]), "B")       # B within 0.003 of the best
        self.assertEqual(pick([0.6369, 0.62, 0.6372, 0.6380, 0.6400]), "bc")    # B out; best two-channel
        self.assertEqual(pick([0.60, 0.6390, 0.6390, 0.62, 0.6400]), "ab")      # same level, same score: order
        self.assertEqual(pick([0.60, 0.61, 0.62, 0.63, 0.6400]), "abc")
        # strict '<': a gap of 0.003 plus a hair is out, minus a hair is in
        self.assertEqual(pick([0.7 - 0.003 - 1e-9, 0.1, 0.1, 0.1, 0.7]), "abc")
        self.assertEqual(pick([0.7 - 0.003 + 1e-9, 0.1, 0.1, 0.1, 0.7]), "B")
        d4 = [n for _, n, _, _ in S.D4]
        self.assertEqual(d4, ["v42_r0u", "v42_s05", "v42_s10"])
        self.assertEqual(S.choose({"v42_r0u": 0.65, "v42_s05": 0.652, "v42_s10": 0.66},
                                  {n: lvl for _, n, lvl, _ in S.D4}, d4)[0], "v42_s10")
        self.assertEqual(S.choose({"v42_r0u": 0.658, "v42_s05": 0.659, "v42_s10": 0.66},
                                  {n: lvl for _, n, lvl, _ in S.D4}, d4)[0], "v42_r0u")
        with self.assertRaises(AssertionError):
            S.choose({"B": float("nan"), "ab": 0.5}, {"B": 0, "ab": 1}, ["B", "ab"])
        with self.assertRaises(AssertionError):
            S.choose({"B": 0.5}, {"B": 0, "ab": 1}, ["B", "ab"])

    def test_per_case_dice_and_manifest(self):
        want = internal_split(0)[1]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.csv"
            rows = ["case_id,Dice_Y,HD95_Y"] + [f"{c},0.5,3.0" for c in want] + ["MEAN,0.5,3.0"]
            p.write_text("\n".join(rows) + "\n")
            self.assertEqual(len(S.per_case_dice(p, want)), len(want))
            p.write_text("\n".join(rows[:-2]) + "\n")
            with self.assertRaises(SystemExit):
                S.per_case_dice(p, want)                       # one V_k case missing
            p.write_text("\n".join(rows[:-1] + [f"{want[0]},0.4,1"]) + "\n")
            with self.assertRaises(SystemExit):
                S.per_case_dice(p, want)                       # duplicate
            m = Path(d) / "m.json"
            S.write_manifest(m, {"a": 1, "x": [0.1]})
            S.write_manifest(m, {"a": 1, "x": [0.1]})          # reproduced: fine
            with self.assertRaises(SystemExit):
                S.write_manifest(m, {"a": 2, "x": [0.1]})      # write-once


class TestD1AM2(unittest.TestCase):
    def test_freeze_rule(self):
        cases = [f"c{i:03d}" for i in range(100)]
        cov = {c: {15: 0.995, 25: 1.0, 35: 1.0} for c in cases}
        d = {c: 50.0 + (i % 10) * 0.1 for i, c in enumerate(cases)}
        r = B.freeze_rule(cov, d)
        self.assertEqual((r["margin"], r["rule_feasible"]), (15, True))
        cov[cases[3]] = {15: 0.98, 25: 0.991, 35: 1.0}
        self.assertEqual(B.freeze_rule(cov, d)["margin"], 25)
        cov[cases[4]] = {15: 0.0, 25: 0.0, 35: 0.0}            # mapping failure: infeasible -> 35
        r = B.freeze_rule(cov, d)
        self.assertEqual((r["margin"], r["rule_feasible"], r["min_cov_at_margin"]), (35, False, 0.0))
        q1, q3 = np.quantile([d[c] for c in cases], [0.25, 0.75])
        self.assertAlmostEqual(r["L"], q1 - 3 * (q3 - q1))
        self.assertAlmostEqual(r["U"], q3 + 3 * (q3 - q1))
        d[cases[0]] = None
        self.assertEqual(B.freeze_rule(cov, d)["d_invalid"], [cases[0]])
        for c in cases[:11]:
            d[c] = float("nan")
        with self.assertRaises(AssertionError):                 # fewer than 90 valid D values
            B.freeze_rule(cov, d)

    def test_fallback(self):
        self.assertEqual(B.fallback_reasons(50.0, True, ["side", "side"], 40, 60), [])
        self.assertEqual(B.fallback_reasons(70.0, True, ["side", "side"], 40, 60), ["D_out_of_bounds"])
        self.assertEqual(B.fallback_reasons(None, True, ["side", "side"], 40, 60), ["D_uncomputable"])
        self.assertEqual(B.fallback_reasons(50.0, False, ["side", "bilateral"], 40, 60), ["structural:side/bilateral"])

    def test_write_once(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fold0.json"
            self.assertTrue(B.write_once(p, {"margin": 25, "L": 1.5, "x": {15: 0.9}}, ["margin", "L", "x"]))
            self.assertFalse(B.write_once(p, {"margin": 25, "L": 1.5, "x": {15: 0.9}, "t": 2}, ["margin", "L", "x"]))
            with self.assertRaises(SystemExit):
                B.write_once(p, {"margin": 35, "L": 1.5, "x": {15: 0.9}}, ["margin", "L", "x"])


class TestPriors(unittest.TestCase):
    def test_dist_on_box(self):
        rng = np.random.default_rng(0)
        c4 = np.zeros((40, 90, 90), bool)
        c4[tuple(rng.integers(0, s, 30) for s in c4.shape)] = True
        full = B.dist_prior(c4)
        for box in ([[10, 20], [30, 60], [5, 40]], [[0, 40], [0, 90], [0, 90]], [[30, 40], [80, 90], [0, 5]]):
            sl = tuple(slice(a, b) for a, b in box)
            np.testing.assert_allclose(B.dist_on_box(c4, box), full[sl], atol=1e-6)

    def test_s1_metrics(self):
        gt = np.zeros((10, 4, 4), np.uint8)
        gt[2:6, 1, 1] = 1
        gt[6:8, 1, 1] = 4
        gt[0, 0, 0] = 5
        am = gt.copy()
        am[0, 0, 0] = 0
        am[8, 1, 1] = 4                                          # one extra c4 voxel one slice lower
        m = B.s1_metrics(am, gt)
        self.assertEqual(m["dice_c1"], 1.0)
        self.assertAlmostEqual(m["dice_c4"], 2 * 2 / (3 + 2))
        self.assertTrue(math.isnan(m["dice_c2"]))
        self.assertEqual((m["z_lo_err_mm"], m["z_hi_err_mm"]), (0.0, 2.5))
        self.assertEqual(m["volratio_c4"], 1.5)

    def test_p_y_and_landmarks(self):
        p = np.zeros((5, 1, 1, 4), np.float32)
        p[:, 0, 0, 0] = [0.6, 0.1, 0.1, 0.1, 0.1]
        p[:, 0, 0, 1] = [0.2, 0.5, 0, 0, 0.3]
        p[:, 0, 0, 2] = [0.1, 0, 0.8, 0, 0.1]
        p[:, 0, 0, 3] = [0.1, 0, 0, 0.2, 0.7]
        am, lm, is4 = B.landmarks(p)
        np.testing.assert_array_equal(am.ravel(), [0, 1, 2, 4])
        np.testing.assert_array_equal(lm.ravel(), [False, True, False, True])   # {1, 4} only, as V4.0
        np.testing.assert_array_equal(is4.ravel(), [False, False, False, True])
        np.testing.assert_allclose(B.p_y(p).ravel(), [0.4, 0.8, 0.9, 0.9], atol=1e-6)

    def test_prior_sha(self):
        a = np.arange(12, dtype=np.uint8).reshape(3, 2, 2, 1)
        box = np.array([[0, 2], [0, 2], [5, 6]])
        self.assertEqual(N.prior_sha(a, box), N.prior_sha(a.copy(), box.tolist()))
        self.assertNotEqual(N.prior_sha(a, box), N.prior_sha(a, box + 1))
        with tempfile.TemporaryDirectory() as d:
            N._save_npz(Path(d) / "x_prior.npz", prior=a, bbox=box)
            with np.load(Path(d) / "x_prior.npz") as z:
                self.assertEqual(N.prior_sha(z["prior"], z["bbox"]), N.prior_sha(a, box))
            self.assertEqual([p.name for p in Path(d).iterdir()], ["x_prior.npz"])   # no temp file left


class TestDatasets(unittest.TestCase):
    def test_ids(self):
        ids = [N.v42p(k, v, s) for k in range(5) for v, s in (("dev", False), ("report", False), ("report", True))]
        self.assertEqual(sorted(d for d, _ in ids), list(range(542, 557)))
        self.assertTrue(all(n.startswith(f"Dataset{d}_") for d, n in ids))
        self.assertEqual(len({n for _, n in ids}), 15)
        with self.assertRaises(AssertionError):
            N.v42p(0, "dev", True)
        self.assertEqual(N.split_copy(N.copy_id("2200498367", 2)), ("2200498367", 2))
        self.assertEqual(N.prior_trainer("ab", "S0"), "nnUNetTrainerV4_Pab")
        self.assertEqual(N.prior_trainer("abc", "S10"), "nnUNetTrainerV4_Pabc_S10")
        self.assertEqual(N.holdout_cases(1, "dev"), internal_split(1)[1])

    def test_prior_trainers_exist(self):
        try:
            import nnunetv2  # noqa: F401
        except ImportError:
            self.skipTest("nnunetv2 not installed in this env")
        for combo in N.COMBOS:
            for w in N.W_SUFFIX:
                cls = N.trainer_class(N.prior_trainer(combo, w))
                self.assertTrue(N.is_v4(cls))
                self.assertEqual("".join(sorted(cls.PRIORS)), combo)
                self.assertEqual(cls.SKEL_W, {"S0": 0.0, "S05": 0.5, "S10": 1.0}[w])
                self.assertIsNone(cls.VIS)

    def test_train42p_guards(self):
        base = dict(dataset="prior", fold=0, variant="dev", trainer=None, name="v42_d2ab", priors="ab", inner=None,
                    cases="", device="cuda")
        old = N.check_dataset42p, N.subprocess.run, N.SELECT42
        with tempfile.TemporaryDirectory() as d:
            N.check_dataset42p, N.subprocess.run, N.SELECT42 = _sentinel, _sentinel, Path(d)
            try:
                for kw in (dict(variant="report"), dict(priors="a"), dict(priors="abcd"), dict(name=None),
                           dict(trainer="nnUNetTrainerV4_Pab"), dict()):            # last: no d4.json yet
                    with self.assertRaises(SystemExit, msg=kw):
                        N.train42(SimpleNamespace(**{**base, **kw}))
                (Path(d) / "fold0").mkdir()
                (Path(d) / "fold0" / "d4.json").write_text(json.dumps(
                    {"step": "D4'", "fold": 0, "chosen": {"name": "v42_s05", "w": "S05"}}))
                self.assertEqual(N.d4_choice(0)[0], "S05")
                with self.assertRaises(GuardPassed):
                    N.train42(SimpleNamespace(**base))
            finally:
                N.check_dataset42p, N.subprocess.run, N.SELECT42 = old


class GuardPassed(Exception):
    pass


def _sentinel(*a, **k):
    raise GuardPassed("a guard let the call through")


class TestJobs(unittest.TestCase):
    def test_d2chain(self):
        argv = sys.argv
        with tempfile.TemporaryDirectory() as d:
            J.JOBS = Path(d) / "jobs.json"
            try:
                for a in (["s1train42", "--folds", "0", "--variant", "dev"],
                          ["nntrain42", "--folds", "0", "--variant", "dev", "--name", "v42_r0u"],
                          ["nntrain42", "--folds", "0", "--variant", "dev", "--name", "v42_s05",
                           "--trainer", "nnUNetTrainerV4_noP_S05"],
                          ["nntrain42", "--folds", "0", "--variant", "dev", "--name", "v42_s10",
                           "--trainer", "nnUNetTrainerV4_noP_S10"],
                          ["nntrain42", "--folds", "1", "--variant", "dev", "--name", "v42_r0u"],
                          ["d2chain42", "--folds", "0"], ["smoke42p", "--after", "s142_dev_f0_i2"],
                          ["d2chain42", "--folds", "0"]):                           # idempotent
                    sys.argv = ["v4_jobs.py"] + a
                    J.main()
            finally:
                sys.argv = argv
            jobs = json.loads(J.JOBS.read_text())
        keys = [j["key"] for j in jobs]
        by = {j["key"]: j for j in jobs}
        self.assertEqual(len(keys), len(set(keys)))
        i = keys.index
        self.assertEqual(keys[i("s142_dev_f0_i2") + 1: i("s142_dev_f0_i2") + 5],
                         ["s1p42_dev_f0_i0", "s1p42_dev_f0_i1", "s1p42_dev_f0_i2", "smoke42p"])
        self.assertLess(i("smoke42p"), i("nn42_v42_r0u_dev_f0"))
        self.assertGreater(i("nn42_v42_d2ab_dev_f0"), i("nn42_v42_s10_dev_f0"))
        self.assertLess(i("d2sel42_f0"), i("nn42_v42_r0u_dev_f1"))
        self.assertEqual(by["s1p42_dev_f0_i1"]["deps"], ["s142_dev_f0_i1"])
        self.assertEqual(set(by["pri42_dev_f0"]["deps"]),
                         {f"s142_dev_f0_i{j}" for j in range(3)} | {f"s1p42_dev_f0_i{j}" for j in range(3)})
        self.assertEqual(set(by["nn42_v42_d2abc_dev_f0"]["deps"]), {"nnprep42p_dev_f0", "d4sel42_f0", "smoke42p"})
        self.assertEqual(set(by["d2sel42_f0"]["deps"]),
                         {"d4sel42_f0"} | {f"sc42_v42_d2{c}_f0" for c in N.COMBOS})
        self.assertEqual(set(by["d4sel42_f0"]["deps"]), {f"sc42_{n}_f0" for n in ("v42_r0u", "v42_s05", "v42_s10")})
        rel = lambda p: p.relative_to(ROOT).as_posix()  # noqa: E731
        self.assertEqual(by["pri42_dev_f0"]["done"], rel(N.PRIORS42 / "fold0_dev" / "DONE"))
        self.assertEqual(by["nnprep42p_dev_f0"]["done"], f"log/v4/nnprep_{N.v42p(0, 'dev')[0]}.done")
        self.assertEqual(by["d4sel42_f0"]["done"], rel(N.SELECT42 / "fold0" / "d4.json"))
        self.assertEqual(by["d2sel42_f0"]["done"], rel(N.SELECT42 / "fold0" / "d2.json"))
        self.assertEqual(by["nn42_v42_d2bc_dev_f0"]["done"], rel(N.RESULTS / "v4_dev_preds" / "v42_d2bc" / "fold0" / "EXPORTED"))
        _, ev = S.DS.GT["harm_v1"]
        self.assertEqual(by["sc42_v42_s05_f0"]["done"], rel(ev / "v42_s05" / "fold0.csv"))
        self.assertEqual(by["sc42_v42_d2ab_f0"]["deps"], ["nn42_v42_d2ab_dev_f0"])
        gpu = [k for k in keys if by[k]["kind"] == "gpu"]
        self.assertEqual({k for k in gpu if "d2" in k or "s1p42" in k or k == "smoke42p"},
                         {f"nn42_v42_d2{c}_dev_f0" for c in N.COMBOS} | {f"s1p42_dev_f0_i{j}" for j in range(3)}
                         | {"smoke42p"})


if __name__ == "__main__":
    unittest.main()
