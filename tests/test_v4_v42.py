"""V4.2 dataset helpers in tools/v4_nnunet.py (label maps, split tables); runs in either env."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainer.v4.splits import (N_INNER, all_cases, inner_folds, internal_split, report_cases,  # noqa: E402
                               stage1_pool, train_cases)

_spec = importlib.util.spec_from_file_location("v4_nnunet", ROOT / "tools" / "v4_nnunet.py")
N = importlib.util.module_from_spec(_spec)
sys.modules["v4_nnunet"] = N            # Pool workers (harm_cache) pickle its functions by module name
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

    def test_split_bounds(self):
        """Out-of-table indices would make nnU-Net fall back to a random 80:20 split of all cases."""
        for bad in ((5, "dev", None), (-1, "report", None), (0, "test", None), (0, "dev", 0)):
            with self.assertRaises(AssertionError):
                N.v42_split_index("binY", *bad)
        for bad in ((5, "dev", 0), (0, "report", 3), (0, "dev", None)):
            with self.assertRaises(AssertionError):
                N.v42_split_index("s1", *bad)

    def test_train42_guards(self):
        """Refused before anything runs: prior arms on CT-only Dataset540, --inner on binY, V4 report runs,
        non-stock Stage 1 trainers, folds / inner folds outside the split tables."""
        base = dict(dataset="binY", fold=0, variant="dev", trainer=None, name="x", inner=None, cases="")
        for kw in (dict(trainer="nnUNetTrainerV4_Pabc_S05"), dict(trainer="nnUNetTrainerV4_SMOKE_Pabc_Vign_S10"),
                   dict(inner=0), dict(variant="report", trainer="nnUNetTrainerV4_noP_S05"),
                   dict(dataset="s1", inner=0, trainer="nnUNetTrainerV4_noP_S05")):
            with self.assertRaises(SystemExit, msg=kw):
                N.train42(SimpleNamespace(**{**base, **kw}))
        for kw in (dict(fold=5), dict(fold=-1), dict(dataset="s1", inner=3), dict(dataset="s1")):
            with self.assertRaises(AssertionError, msg=kw):
                N.train42(SimpleNamespace(**{**base, **kw}))

    def test_harm_cache_checks(self):
        """harm_cache verifies config sha, rule-code sha and every label array against the manifest."""
        import hashlib
        import json
        old = N.HARM_DIR, N.HARM_CFG
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cache = d / N.CACHE_DIR.name
            cache.mkdir()
            rule = hashlib.sha256((ROOT / "trainer" / "v4" / "harmonize.py").read_bytes()).hexdigest()
            cfg = d / "harm.json"
            cfg.write_text(json.dumps({"t": 0.19, "code_sha256": {"trainer/v4/harmonize.py": rule}}))
            man = {"config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest(), "cases": {}}
            for i, c in enumerate(all_cases()):
                lab = np.full((1, 2, 2), i % 6, np.uint8)
                np.save(cache / f"{c}_label.npy", lab)
                man["cases"][c] = {"sha256_label": hashlib.sha256(lab.tobytes()).hexdigest()}
            (cache / "manifest.json").write_text(json.dumps(man))
            (cache / "DONE").write_text("t\n")
            N.HARM_DIR, N.HARM_CFG = d, cfg
            try:
                got, sha, digest = N.harm_cache()
                self.assertEqual((got, sha), (cache, man["config_sha256"]))
                lines = "".join(f"{c}:{man['cases'][c]['sha256_label']}\n" for c in sorted(man["cases"]))
                self.assertEqual(digest, hashlib.sha256(lines.encode()).hexdigest())
                c0 = all_cases()[0]
                np.save(cache / f"{c0}_label.npy", np.full((1, 2, 2), 1, np.uint8))   # case 0 was all zeros
                with self.assertRaises(AssertionError):
                    N.harm_cache()
                N.harm_cache(verify=False)                                         # content check is opt-out only
                cfg.write_text(cfg.read_text() + " ")
                with self.assertRaises(AssertionError):
                    N.harm_cache(verify=False)
            finally:
                N.HARM_DIR, N.HARM_CFG = old

    def test_prep42_cases_guard(self):
        """--cases must refuse a raw dir that already holds other cases (it would rewrite their splits)."""
        old = N.NNUNET_HOME
        with tempfile.TemporaryDirectory() as d:
            N.NNUNET_HOME = Path(d)
            try:
                img = Path(d) / "nnUNet_raw" / N.V42["binY"][1] / "imagesTr"
                img.mkdir(parents=True)
                c0, c1 = all_cases()[:2]
                (img / f"{c1}_0000.nii.gz").touch()
                with self.assertRaises(AssertionError):
                    N.prep42(SimpleNamespace(what="binY", cases=c0, overwrite=False, np=1))
                self.assertFalse((img.parent / "dataset.json").exists())
            finally:
                N.NNUNET_HOME = old

    def test_export42_roundtrip(self):
        """export42 writes the cache (z, LR, AP) grid from both validation formats: nnU-Net's NIfTI (stored
        transposed by NibabelIO.write_seg) and the V4 trainer's full-volume npz."""
        try:
            import nibabel as nib
            import nrrd
        except ImportError:
            self.skipTest("needs nibabel + pynrrd (nnU-Net venv)")
        c = internal_split(0)[1][0]
        if not (N.CACHE_DIR / f"{c}_label.npy").exists():
            self.skipTest("preprocess cache not here")
        shape = np.load(N.CACHE_DIR / f"{c}_label.npy", mmap_mode="r").shape
        y = (np.random.default_rng(0).random(shape) > 0.999).astype(np.uint8)
        old = N.RESULTS
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            N.RESULTS = d / "results"
            try:
                fd = d / "fold_5"
                (fd / "validation").mkdir(parents=True)
                (fd / "validation_v4").mkdir()
                nib.save(nib.Nifti1Image(y.transpose(2, 1, 0), np.eye(4)), str(fd / "validation" / f"{c}.nii.gz"))
                np.savez_compressed(fd / "validation_v4" / f"{c}.npz", Y=y)
                for stock, name in ((True, "stock"), (False, "v4")):
                    N.export42(SimpleNamespace(fold=0, variant="dev", cases=c, name=name), fd, stock)
                    got, _ = nrrd.read(str(N.RESULTS / "v4_dev_preds" / name / "fold0" / f"{c}_stage2_label.nrrd"))
                    np.testing.assert_array_equal(got, y)
            finally:
                N.RESULTS = old

    def test_jobs_done_paths(self):
        """Queue done paths (tools/v4_jobs.py) are the markers prep42 / train42 / export42 write."""
        import json
        spec = importlib.util.spec_from_file_location("v4_jobs", ROOT / "tools" / "v4_jobs.py")
        J = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(J)
        argv = sys.argv
        with tempfile.TemporaryDirectory() as d:
            J.JOBS = Path(d) / "jobs.json"
            try:
                for a in (["nnprep42", "--what42", "binY"], ["nnprep42", "--what42", "s1"],
                          ["nntrain42", "--folds", "0", "--variant", "dev", "--name", "v42_r0u"],
                          ["nntrain42", "--folds", "0", "--variant", "report", "--name", "v42_r0u"],
                          ["s1train42", "--folds", "0", "--variant", "dev"]):
                    sys.argv = ["v4_jobs.py"] + a
                    J.main()
            finally:
                sys.argv = argv
            jobs = {j["key"]: j for j in json.loads(J.JOBS.read_text())}
        rel = lambda p: p.relative_to(ROOT).as_posix()  # noqa: E731
        self.assertEqual(jobs["nnprep42_binY"]["done"], "log/v4/nnprep_540.done")
        self.assertEqual(jobs["nnprep42_s1"]["done"], "log/v4/nnprep_541.done")
        self.assertEqual(jobs["nnprep42_s1"]["deps"], ["nnprep42_binY"])
        self.assertEqual(jobs["nn42_v42_r0u_dev_f0"]["done"],
                         rel(N.RESULTS / "v4_dev_preds" / "v42_r0u" / "fold0" / "EXPORTED"))
        self.assertEqual(jobs["nn42_v42_r0u_report_f0"]["done"],
                         rel(N.RESULTS / "v4_v42_r0u_label_predictions" / "EXPORTED_fold0"))
        for j in range(N_INNER):
            self.assertEqual(jobs[f"s142_dev_f0_i{j}"]["done"], f"log/v4/s142_dev_f0_i{j}.done")
            self.assertEqual(jobs[f"s142_dev_f0_i{j}"]["deps"], ["nnprep42_s1"])

    def test_dev_summary_gt(self):
        """One eval directory and one default reference per reference standard; tagged directories."""
        import json
        spec = importlib.util.spec_from_file_location("v4_dev_summary", ROOT / "tools" / "v4_dev_summary.py")
        S = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S)
        self.assertEqual(len({e for _, e in S.GT.values()}), len(S.GT))
        self.assertEqual(len({c for c, _ in S.GT.values()}), len(S.GT))
        self.assertEqual(S.GT["harm_v1"][0], N.HARM_DIR / N.CACHE_DIR.name)    # the cache prep42 trains on
        self.assertEqual(S.REF, {"orig": "r0_dev", "harm_v1": "v42_r0u"})
        self.assertTrue(np.isnan(S.boot_ci(np.array([]))[0]))
        with tempfile.TemporaryDirectory() as d:
            S.GT = {"orig": (Path(d) / "cache", Path(d) / "eval")}
            S.check_tag("orig", write=True)
            S.check_tag("orig")
            (Path(d) / "eval" / "GT.json").write_text(json.dumps({"gt": "harm_v1"}))
            with self.assertRaises(SystemExit):
                S.check_tag("orig")

    def test_ids(self):
        self.assertEqual({d for d, _ in N.V42.values()}, {540, 541})
        self.assertTrue(all(n.startswith(f"Dataset{d}_") for d, n in N.V42.values()))
        self.assertEqual(len(all_cases()), 154)


if __name__ == "__main__":
    unittest.main()
