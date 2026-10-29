"""V4.3 (V4_PLAN §12): dropout geometry, input masking, MM-shuf placement, selection rule, guards, jobs.
Trainer tests need the nnU-Net venv with tools/v4_mm43.py install43 done; the rest run in either env."""
import importlib.util
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from topocnv.data.v4_splits import internal_split  # noqa: E402


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

    def test_derangement_rejects_invalid_cases_before_rng(self):
        with mock.patch.object(M.np.random, "default_rng", side_effect=AssertionError("invalid cases reached RNG")):
            for cases in ([], ["a"], ["a", "a"], ["a", "b", "a"]):
                with self.subTest(cases=cases), self.assertRaises(ValueError):
                    M.derangement(cases)
        self.assertEqual(M.derangement(["b", "a"]), {"a": "b", "b": "a"})

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


class TestShuffleFoldIsolation(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.folds = [[f"f{k}_c{i}" for i in range(4)] for k in range(5)]
        self.cases = sum(self.folds, [])
        self.ok = {c: int(not c.endswith("c3")) for c in self.cases}
        self.splits = []
        for variant in ("report", "dev"):
            for k in range(5):
                val = self.folds[k if variant == "report" else (k + 1) % 5]
                excluded = set(self.folds[k] + val)
                self.splits.append({"train": [c for c in reversed(self.cases) if c not in excluded],
                                    "val": list(reversed(val))})
        self.base_meta = {"harm_cache": str(self.root / "harm"), "harm_config_sha256": "config-sha",
                          "harm_label_digest": "label-sha", "image_digest": "ct-sha"}
        self.run, self.export, self.log = mock.Mock(), mock.Mock(), mock.Mock()
        self.cls = type("nnUNetTrainerV43", (), {})
        self.trainer_class = mock.Mock(return_value=self.cls)
        self.verify = mock.Mock(return_value=("mri-sums", {}))
        self.split_source = mock.Mock(return_value=self.splits)
        for obj, attr, value in (
            (M, "ROOT", self.root), (N, "NNUNET_HOME", self.root / "nnunet"),
            (M, "CACHE_DIR", self.root / "ct"), (M, "MRI_ROOT", self.root / "mri"),
            (M, "MRI_CACHE", self.root / "mri" / "cache"),
            (M, "all_cases", mock.Mock(return_value=self.cases)),
            (M, "reg_ok", mock.Mock(return_value=self.ok)), (N, "v42_splits", self.split_source),
            (M, "mri_sums", mock.Mock(return_value=("mri-sums", {}))),
            (M, "verify_mri_cache", self.verify),
            (N, "check_direction", mock.Mock(return_value=("config-sha", "label-sha"))),
            (N, "_versions", mock.Mock(return_value={})), (M, "subprocess", SimpleNamespace(run=self.run)),
            (N, "trainer_class", self.trainer_class), (N, "_log_run", self.log), (M, "export43", self.export),
            (M, "frozen_choice", mock.Mock(return_value=({"trainer": "nnUNetTrainerV43_MMd1"}, "freeze-sha"))),
        ):
            patcher = mock.patch.object(obj, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def _meta_path(self, name):
        return N.NNUNET_HOME / "nnUNet_raw" / name / "v42_meta.json"

    def _split_path(self, name):
        return N.NNUNET_HOME / "nnUNet_preprocessed" / name / "splits_final.json"

    def _subset(self, fold, variant):
        split = self.splits[fold + (5 if variant == "dev" else 0)]
        train = sorted(split["train"])
        return ([c for c in train if self.ok[c]][:2] + [c for c in train if not self.ok[c]][:1]
                + split["val"])

    def _dataset(self, fold=2, variant="dev", cases=None, what="shuf"):
        _, name = M.dataset43(what, fold, variant)
        shuffle = M.shuffle_assignment(fold, variant, cases) if what == "shuf" else None
        selected = shuffle["split"]["train"] + shuffle["split"]["val"] if shuffle else self.cases
        donors = shuffle["donors"] if shuffle else {}
        meta = dict(self.base_meta, protocol=M.PROTOCOL, what=f"mm43_{what}", mri_sums_sha256="mri-sums",
                    mri_digest="mri-digest", shuffle=shuffle, shuffle_seed=shuffle["seed"] if shuffle else None,
                    partial=cases is not None, cases={c: {"reg_ok": self.ok[c], "donor": donors.get(c)}
                                                     for c in selected})
        self._json(self._meta_path(name), meta)
        self._json(self._split_path(name), [shuffle["split"]] if shuffle else self.splits)
        return name, meta

    def test_dataset_ids_are_fold_and_variant_specific(self):
        seen = set()
        self.assertEqual(M.dataset43("mm"), (557, "Dataset557_CNVv43mm"))
        for variant, base in (("report", 559), ("dev", 564)):
            for k in range(5):
                with self.subTest(fold=k, variant=variant):
                    expected = (base + k, f"Dataset{base + k}_CNVv43mmshuf_{variant}_f{k}")
                    self.assertEqual(M.dataset43("shuf", k, variant), expected)
                    seen.add(expected[0])
                    self.assertEqual(M.dataset43("mm", k, variant), M.dataset43("mm"))
        self.assertEqual(seen, set(range(559, 569)))
        for args in (("shuf",), ("shuf", 0), ("shuf", None, "dev"), ("unknown", 0, "dev")):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                M.dataset43(*args)
        for k, variant in ((-1, "dev"), (5, "report"), (0, "other")):
            with self.subTest(fold=k, variant=variant), self.assertRaises(AssertionError):
                M.dataset43("shuf", k, variant)

    def test_assignments_use_only_this_training_pool(self):
        original = deepcopy(self.splits)
        for variant in ("report", "dev"):
            for k in range(5):
                with self.subTest(fold=k, variant=variant):
                    index = k + (5 if variant == "dev" else 0)
                    split = {key: sorted(value) for key, value in self.splits[index].items()}
                    pool = [c for c in split["train"] if self.ok[c]]
                    val = [c for c in split["val"] if self.ok[c]]
                    with mock.patch.object(M, "derangement", wraps=M.derangement) as draw:
                        assignment = M.shuffle_assignment(k, variant)
                    draw.assert_called_once_with(pool, 20261028 + index)
                    self.assertEqual({key: assignment[key] for key in ("policy", "fold", "variant", "seed")},
                                     {"policy": "train_pool_v1", "fold": k, "variant": variant,
                                      "seed": 20261028 + index})
                    self.assertEqual(assignment["split"], split)
                    donors = assignment["donors"]
                    self.assertEqual(set(donors), set(pool + val))
                    self.assertEqual(sorted(donors[c] for c in pool), pool)
                    self.assertTrue(all(c != donors[c] for c in donors))
                    self.assertLessEqual(set(donors.values()), set(pool))
                    self.assertFalse(set(donors.values()) & set(split["val"] + self.folds[k]))
                    self.assertFalse(set(donors) & {c for c in self.cases if not self.ok[c]})
                    rng = np.random.default_rng([20261028 + index, 1])
                    self.assertEqual([donors[c] for c in val], rng.choice(pool, len(val), replace=False).tolist())
        self.split_source.assert_called_with("binY")
        self.assertEqual(self.splits, original)

    def test_assignments_are_seeded_and_order_independent(self):
        reversed_splits = [{key: list(reversed(value)) for key, value in split.items()} for split in self.splits]
        for variant in ("report", "dev"):
            for k in range(5):
                for cases in (None, self._subset(k, variant)):
                    with self.subTest(fold=k, variant=variant, partial=cases is not None):
                        expected = M.shuffle_assignment(k, variant, cases)
                        self.assertEqual(M.shuffle_assignment(k, variant, cases), expected)
                        with mock.patch.object(N, "v42_splits", return_value=reversed_splits):
                            reordered = list(reversed(cases)) if cases is not None else None
                            self.assertEqual(M.shuffle_assignment(k, variant, reordered), expected)

    def test_subset_never_borrows_outside_it(self):
        cases = self._subset(2, "dev")
        assignment = M.shuffle_assignment(2, "dev", cases)
        split, donors = assignment["split"], assignment["donors"]
        pool = sorted(c for c in cases[:3] if self.ok[c])
        self.assertEqual(split, {"train": sorted(cases[:3]), "val": sorted(cases[3:])})
        self.assertEqual({c: donors[c] for c in pool}, {pool[0]: pool[1], pool[1]: pool[0]})
        self.assertEqual(set(donors), {c for c in cases if self.ok[c]})
        self.assertLessEqual(set(donors.values()), set(pool))
        self.assertGreater(sum(self.ok[c] for c in split["val"]), len(pool))
        self.assertFalse(set(donors.values()) & (set(self.cases) - set(cases)))

    def test_invalid_subsets_and_insufficient_donors_fail_closed(self):
        split = self.splits[7]
        for cases in ([], split["train"], split["val"]):
            with self.subTest(cases=cases), self.assertRaisesRegex(SystemExit, "nonempty train and validation"):
                M.shuffle_assignment(2, "dev", cases)
        for extra in ("unknown", self.folds[2][0]):
            with self.subTest(extra=extra), self.assertRaisesRegex(SystemExit, "outside dev fold 2"):
                M.shuffle_assignment(2, "dev", self._subset(2, "dev") + [extra])
        train = sorted(c for c in split["train"] if self.ok[c])
        val = sorted(c for c in split["val"] if self.ok[c])
        with self.assertRaisesRegex(SystemExit, "at least two MRI-usable training cases"):
            M.shuffle_assignment(2, "dev", [train[0], val[0]])
        for ntrain, nval in ((0, 1), (1, 0), (1, 1), (0, 0)):
            ok = dict.fromkeys(self.cases, 1)
            ok.update(dict.fromkeys(split["train"] + split["val"], 0))
            ok.update(dict.fromkeys(train[:ntrain] + val[:nval], 1))
            with self.subTest(ntrain=ntrain, nval=nval), mock.patch.object(M, "reg_ok", return_value=ok):
                if ntrain or nval:
                    with self.assertRaisesRegex(SystemExit, "at least two MRI-usable training cases"):
                        M.shuffle_assignment(2, "dev")
                else:
                    self.assertEqual(M.shuffle_assignment(2, "dev")["donors"], {})

    def test_check_dataset_accepts_own_fold_only(self):
        for variant in ("report", "dev"):
            for k in range(5):
                with self.subTest(fold=k, variant=variant):
                    name, meta = self._dataset(k, variant)
                    self.assertEqual(M.check_dataset43(name, k, variant), meta)
                    for other_k, other_variant in (((k + 1) % 5, variant),
                                                   (k, "dev" if variant == "report" else "report")):
                        with self.assertRaisesRegex(SystemExit, "does not match the requested fold/variant"):
                            M.check_dataset43(name, other_k, other_variant)
        name, meta = self._dataset(cases=self._subset(2, "dev"))
        self.assertEqual(M.check_dataset43(name, 2, "dev"), meta)
        for k, variant in ((None, None), (2, None), (None, "dev")):
            with self.assertRaises(SystemExit):
                M.check_dataset43(name, k, variant)
        old_name = "Dataset558_CNVv43mmshuf"
        self._json(self._meta_path(old_name), meta)
        with self.assertRaisesRegex(SystemExit, "does not match the requested fold/variant"):
            M.check_dataset43(old_name, 2, "dev")
        mm_name, mm_meta = self._dataset(what="mm")
        self.assertEqual(M.check_dataset43(mm_name), mm_meta)

    def test_check_dataset_rejects_legacy_and_incomplete_shuffle_metadata(self):
        name, meta = self._dataset()
        for key in ("shuffle", "shuffle_seed"):
            for value in (None, 20261028):
                bad = deepcopy(meta)
                if value is None:
                    bad.pop(key)
                else:
                    bad[key] = value
                self._json(self._meta_path(name), bad)
                with self.subTest(key=key, value=value), self.assertRaisesRegex(SystemExit, "legacy/global or mismatched"):
                    M.check_dataset43(name, 2, "dev")
        shuffle = meta["shuffle"]
        wrong = {"policy": "global", "fold": 3, "variant": "report", "seed": shuffle["seed"] + 1,
                 "split": {"train": shuffle["split"]["val"], "val": shuffle["split"]["train"]},
                 "donors": {c: self.folds[2][0] for c in shuffle["donors"]}}
        for key, value in wrong.items():
            for missing in (False, True):
                bad = deepcopy(meta)
                if missing:
                    bad["shuffle"].pop(key)
                else:
                    bad["shuffle"][key] = value
                self._json(self._meta_path(name), bad)
                with self.subTest(key=key, missing=missing), self.assertRaisesRegex(SystemExit, "legacy/global or mismatched"):
                    M.check_dataset43(name, 2, "dev")
        _, other_meta = self._dataset(3, "dev")
        self._json(self._meta_path(name), other_meta)
        with self.assertRaisesRegex(SystemExit, "legacy/global or mismatched"):
            M.check_dataset43(name, 2, "dev")

    def test_check_dataset_checks_case_donors_availability_and_membership(self):
        name, meta = self._dataset()
        split = meta["shuffle"]["split"]
        for partition in split.values():
            for available in (0, 1):
                case = next(c for c in partition if self.ok[c] == available)
                for key, value, message in (("donor", self.folds[2][0], "donor mismatch"),
                                            ("reg_ok", 1 - available, "MRI availability changed")):
                    bad = deepcopy(meta)
                    bad["cases"][case][key] = value
                    self._json(self._meta_path(name), bad)
                    with self.subTest(case=case, key=key), self.assertRaisesRegex(AssertionError, message):
                        M.check_dataset43(name, 2, "dev")
        for extra in (False, True):
            bad = deepcopy(meta)
            if extra:
                bad["cases"][self.folds[2][0]] = {"reg_ok": 1, "donor": None}
            else:
                bad["cases"].pop(split["train"][0])
            self._json(self._meta_path(name), bad)
            with self.subTest(extra=extra), self.assertRaisesRegex(AssertionError, "case set does not match"):
                M.check_dataset43(name, 2, "dev")

    def test_check_dataset_rejects_modified_or_multiple_splits(self):
        name, meta = self._dataset()
        split = meta["shuffle"]["split"]
        for splits in ([], [split, split], self.splits, [self.splits[8]],
                       [{"train": split["train"][1:], "val": split["val"]}],
                       [{"train": split["val"], "val": split["train"]}]):
            self._json(self._split_path(name), splits)
            with self.subTest(splits=splits), self.assertRaisesRegex(AssertionError, "exactly its own split at index 0"):
                M.check_dataset43(name, 2, "dev")

    def test_prep_wires_assignment_jobs_metadata_and_single_split(self):
        self._json(self._meta_path(N.V42["binY"][1]), self.base_meta)
        src_pre = self._split_path(N.V42["binY"][1]).parent
        self._json(src_pre / "splits_final.json", self.splits)
        self._json(src_pre / f"{N.PLANS}.json", {"configurations": {N.CONFIG: {
            "normalization_schemes": ["NoNormalization"], "use_mask_for_norm": [False], "patch_size": [2, 3, 4]}}})
        marker = M.ROOT / "log" / "v4" / "nnprep_540.done"
        marker.parent.mkdir(parents=True)
        marker.write_text("synthetic")
        M.MRI_CACHE.mkdir(parents=True)
        (M.MRI_CACHE / "FROZEN").write_text("synthetic")

        def write(job):
            case, _, _, donor = job
            return case, self.ok[case], f"mri-{case}", f"valid-{case}", [1, 0, -1] if donor else None

        for what, k, variant, partial in (("shuf", 2, "report", False), ("shuf", 2, "dev", False),
                                          ("shuf", 4, "dev", True), ("mm", 2, "dev", False)):
            with self.subTest(what=what, fold=k, variant=variant, partial=partial):
                cases = self._subset(k, variant) if partial else None
                expected = M.shuffle_assignment(k, variant, cases) if what == "shuf" else None
                selected = sorted(expected["split"]["train"] + expected["split"]["val"]) if expected else self.cases
                donors = expected["donors"] if expected else {}
                did, name = M.dataset43(what, k, variant)
                raw = self._meta_path(name).parent
                args = SimpleNamespace(what=what, fold=k, variant=variant, cases=",".join(cases or []),
                                       np=1, skip_verify=False)
                self.run.reset_mock()
                self.verify.reset_mock()
                with mock.patch("multiprocessing.Pool") as pool, mock.patch.object(M, "_write43", side_effect=write) as writer, \
                        mock.patch.object(M, "shuffle_assignment", wraps=M.shuffle_assignment) as assign:
                    pool.return_value.__enter__.return_value.imap_unordered.side_effect = lambda fn, jobs: map(fn, jobs)
                    M.prep43(args)
                    pool.assert_called_once_with(1)
                    if what == "shuf":
                        assign.assert_called_once_with(k, variant, cases)
                    else:
                        assign.assert_not_called()
                    self.assertEqual(writer.call_args_list, [mock.call((c, what, str(raw), donors.get(c))) for c in selected])
                self.verify.assert_called_once_with(1)
                meta = json.loads(self._meta_path(name).read_text())
                self.assertEqual(meta["shuffle"], expected)
                self.assertEqual(meta["shuffle_seed"], expected["seed"] if expected else None)
                self.assertEqual(meta["partial"], partial)
                self.assertEqual(sorted(meta["cases"]), selected)
                for case, rec in meta["cases"].items():
                    self.assertEqual(rec["reg_ok"], self.ok[case])
                    self.assertEqual((rec["mri_sha"], rec["valid_sha"]), (f"mri-{case}", f"valid-{case}"))
                    if what == "shuf":
                        self.assertEqual(rec["donor"], donors.get(case))
                        self.assertEqual(rec["shift"], [1, 0, -1] if case in donors else None)
                self.assertEqual(json.loads(self._split_path(name).read_text()), [expected["split"]] if expected else self.splits)
                self.assertEqual(json.loads((raw / "dataset.json").read_text())["numTraining"], len(selected))
                plans = json.loads((self._split_path(name).parent / f"{N.PLANS}.json").read_text())
                self.assertEqual(plans["dataset_name"], name)
                self.assertEqual(plans["configurations"][N.CONFIG]["normalization_schemes"], ["NoNormalization"] * 3)
                self.assertEqual(M.check_dataset43(name, k, variant), meta)
                commands = [call.args[0] for call in self.run.call_args_list]
                self.assertEqual(len(commands), 3)
                for command, executable in zip(commands[:2], ("nnUNetv2_preprocess", "nnUNetv2_extract_fingerprint")):
                    self.assertEqual(command[0], N.bin_cmd(executable))
                    self.assertEqual(command[command.index("-d") + 1], str(did))
                self.assertIn(str(self._split_path(name).parent / f"{N.PLANS}_{N.CONFIG}"), commands[2][2])
                self.assertEqual((M.ROOT / "log" / "v4" / f"nnprep_{did}.done").exists(), not partial)

    def test_prep_refuses_any_existing_directory_with_same_dataset_id(self):
        did, name = M.dataset43("shuf", 2, "dev")
        args = SimpleNamespace(what="shuf", fold=2, variant="dev", cases="", np=1, skip_verify=False)
        for folder in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            for existing_name in (name, f"Dataset{did}_other_arm"):
                with self.subTest(folder=folder, name=existing_name), tempfile.TemporaryDirectory(dir=self.root) as d:
                    home = Path(d)
                    existing = home / folder / existing_name
                    existing.mkdir(parents=True)
                    sentinel = existing / "keep"
                    sentinel.write_bytes(b"unchanged")
                    before = sorted(home.rglob("*"))
                    with mock.patch.object(N, "NNUNET_HOME", home), mock.patch.object(N, "check_dataset42") as check:
                        with self.assertRaisesRegex(SystemExit, f"refusing to overwrite MM-shuf dataset ID {did}"):
                            M.prep43(args)
                        check.assert_not_called()
                    self.assertEqual(sorted(home.rglob("*")), before)
                    self.assertEqual(sentinel.read_bytes(), b"unchanged")
        self.verify.assert_not_called()
        self.run.assert_not_called()

    def test_train_uses_new_dataset_split_zero_and_mm_original_index(self):
        for what in ("shuf", "mm"):
            for variant in ("report", "dev"):
                for k in range(5):
                    with self.subTest(what=what, fold=k, variant=variant):
                        name, meta = self._dataset(k, variant, what=what)
                        (self._split_path(name).parent / f"{N.PLANS}_{N.CONFIG}").mkdir(exist_ok=True)
                        trainer = ("nnUNetTrainerV43_MMd1" if what == "mm" else
                                   "nnUNetTrainerV43_MM" if variant == "report" else "nnUNetTrainerV43_SMOKE_MMd")
                        args = SimpleNamespace(what=what, fold=k, variant=variant, trainer=trainer, name="synthetic", cases="")
                        did = 557 if what == "mm" else 559 + k + (5 if variant == "dev" else 0)
                        split = k + (5 if variant == "dev" else 0) if what == "mm" else 0
                        for mocked in (self.run, self.log, self.export, self.trainer_class):
                            mocked.reset_mock()
                        with mock.patch.object(M, "check_dataset43", wraps=M.check_dataset43) as check, \
                                mock.patch.object(N, "results_dir", wraps=N.results_dir) as results:
                            M.train43(args)
                            check.assert_called_once_with(name, k, variant)
                            results.assert_called_once_with(did, name, trainer, split)
                        self.trainer_class.assert_called_once_with(trainer)
                        command = [N.bin_cmd("nnUNetv2_train"), str(did), N.CONFIG, str(split), "-tr", trainer, "-p", N.PLANS]
                        self.assertEqual(self.run.call_count, 2)
                        self.run.assert_called_with(command, env=N.env(), check=True)
                        self.log.assert_called_once()
                        self.assertEqual(self.log.call_args.args[3:5], (meta, command))
                        self.assertEqual(self.log.call_args.kwargs["extra"]["mm43"]["shuffle"], meta["shuffle"])
                        self.export.assert_called_once_with(args, N.results_dir(did, name, trainer, split), trainer, meta)

    def test_partial_shuf_requires_smoke_even_without_export_subset(self):
        name, meta = self._dataset(2, "report", self._subset(2, "report"))
        args = SimpleNamespace(what="shuf", fold=2, variant="report", trainer="nnUNetTrainerV43_MM", name="synthetic", cases="")
        with self.assertRaisesRegex(SystemExit, "partial MM-shuf dataset is for smoke trainers only"):
            M.train43(args)
        for mocked in (self.trainer_class, self.run, self.log, self.export):
            mocked.assert_not_called()
        args.trainer = "nnUNetTrainerV43_SMOKE_MMd"
        (self._split_path(name).parent / f"{N.PLANS}_{N.CONFIG}").mkdir()
        M.train43(args)
        self.assertEqual(self.run.call_args.args[0][1:4], ["561", N.CONFIG, "0"])
        self.export.assert_called_once_with(args, N.results_dir(561, name, args.trainer, 0), args.trainer, meta)

    def test_write43_uses_training_donor_channels_and_preserves_recipient_ct_labels(self):
        import nibabel as nib

        train = sorted(self.splits[7]["train"])
        val = sorted(self.splits[7]["val"])
        a, b, missing_train, recipient, missing_val = train[0], train[1], train[3], val[0], val[3]
        cases = [a, b, missing_train, recipient, missing_val]
        assignment = M.shuffle_assignment(2, "dev", cases)
        centers = {a: (1, 1, 1), b: (3, 3, 3), recipient: (4, 3, 4)}
        offsets = {a: [(0, 0, 0)], b: [(-1, 0, 0), (0, 0, 0), (1, 0, 0)],
                   recipient: [(0, -1, 0), (0, 0, 0), (0, 1, 0)]}
        intensity = {a: 0.125, b: 0.75, recipient: 0.5}
        shape = (7, 7, 7)
        src = N.NNUNET_HOME / "nnUNet_raw" / N.V42["binY"][1]
        raw = N.NNUNET_HOME / "nnUNet_raw" / M.dataset43("shuf", 2, "dev")[1]
        for folder in (M.CACHE_DIR, M.MRI_CACHE, src / "imagesTr", src / "labelsTr", raw / "imagesTr", raw / "labelsTr"):
            folder.mkdir(parents=True, exist_ok=True)
        originals = {}
        for i, case in enumerate(cases):
            ct = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + i * 1000
            label = np.zeros(shape, np.uint8)
            label[i, 0, 6] = 1
            valid = np.zeros(shape, np.uint8)
            for offset in offsets.get(case, []):
                valid[tuple(np.array(centers[case]) + offset)] = 1
            np.save(M.CACHE_DIR / f"{case}_label.npy", label)
            np.save(M.MRI_CACHE / f"{case}_mri.npy", valid.astype(np.float32) * intensity.get(case, 0))
            np.save(M.MRI_CACHE / f"{case}_mri_valid.npy", valid)
            for arr, path in ((ct, src / "imagesTr" / f"{case}_0000.nii.gz"), (label, src / "labelsTr" / f"{case}.nii.gz")):
                N.write_nifti(arr, path, arr.dtype)
                originals[path] = arr, path.read_bytes()
        for case in cases:
            with self.subTest(case=case):
                donor = assignment["donors"].get(case)
                record = M._write43((case, "shuf", str(raw), donor))
                expected_valid = np.zeros(shape, np.uint8)
                if donor:
                    self.assertIn(donor, (a, b))
                    for offset in offsets[donor]:
                        expected_valid[tuple(np.array(centers[case]) + offset)] = 1
                expected_mri = expected_valid.astype(np.float32) * intensity.get(donor, 0)
                for channel, expected in ((1, expected_mri), (2, expected_valid)):
                    path = raw / "imagesTr" / f"{case}_{channel:04d}.nii.gz"
                    actual = np.asarray(nib.load(str(path)).dataobj).transpose(2, 1, 0)
                    np.testing.assert_array_equal(actual, expected)
                shift = (np.array(centers[case]) - centers[donor]).tolist() if donor else None
                self.assertEqual(record, (case, int(expected_valid.sum()), N._arr_sha(expected_mri),
                                          N._arr_sha(expected_valid), shift))
        for path, (arr, contents) in originals.items():
            linked = raw / path.relative_to(src)
            self.assertTrue(linked.is_symlink())
            self.assertEqual(linked.resolve(), path)
            self.assertEqual(path.read_bytes(), contents)
            np.testing.assert_array_equal(np.asarray(nib.load(str(linked)).dataobj).transpose(2, 1, 0), arr)
        for case, what, donor, error in ((a, "shuf", None, SystemExit), (a, "shuf", a, AssertionError),
                                         (missing_train, "shuf", a, AssertionError), (a, "mm", b, AssertionError)):
            with self.subTest(case=case, what=what, donor=donor), self.assertRaises(error):
                M._write43((case, what, str(raw), donor))


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
