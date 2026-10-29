"""抽取组件的 CPU 数值兼容门禁，仅使用临时合成数据。"""
import ast
import copy
import importlib
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import nrrd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from topocnv.data.labels import BinaryYDataset, load_label_volume
from topocnv.data.utils import sample_validation_patch_positions
from topocnv.engine import segmentation

CASCADE = "npc_vesselFM3D_cascade_MIA_v3_vit_memory"


def reference(name, symbols):
    source = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", "85a6f2d:trainer/" + name + ".py"], text=True,
    )
    namespace = vars(importlib.import_module("trainer." + name)).copy()
    selected = [node for node in ast.parse(source).body if getattr(node, "name", None) in symbols]
    exec(compile(ast.Module(body=selected, type_ignores=[]), "reference", "exec"), namespace)
    return namespace


class TrainingComponents(unittest.TestCase):
    def test_epoch_values_updates_empty_and_nonfinite(self):
        old = reference("npc_baseline", {"train_one_epoch", "evaluate"})
        torch.manual_seed(34)
        image = torch.rand(1, 1, 2, 3, 4)
        label = torch.randint(0, 6, (1, 2, 3, 4))
        for dtype in (None, torch.bfloat16):
            for corrupted in (False, True):
                models = [nn.Conv3d(1, 6, 1)]
                models.append(copy.deepcopy(models[0]))
                results = []
                for index, model in enumerate(models):
                    calls = [0]
                    def loss_fn(logits, labels):
                        calls[0] += 1
                        if corrupted and calls[0] == 2:
                            return logits.sum() * float("nan")
                        return nn.functional.cross_entropy(logits, labels)
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    old["AMP_DTYPE"] = dtype
                    with redirect_stdout(io.StringIO()):
                        if index == 0:
                            result = old["train_one_epoch"](model, [(image, label)] * 3, optimizer, loss_fn, "cpu")
                        else:
                            result = segmentation.train_supervised_epoch(
                                model, [(image, label)] * 3, optimizer, loss_fn, "cpu", amp_dtype=dtype)
                    results.append(result)
                np.testing.assert_array_equal(*results)
                for a, b in zip(models[0].parameters(), models[1].parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                expected = old["evaluate"](models[0], [(image, label)], nn.functional.cross_entropy, "cpu")
                actual = segmentation.evaluate_supervised_epoch(
                    models[1], [(image, label)], nn.functional.cross_entropy, "cpu", amp_dtype=dtype)
                np.testing.assert_array_equal(expected, actual)
        model = nn.Conv3d(1, 6, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        self.assertEqual(segmentation.train_supervised_epoch(model, [], optimizer, None, "cpu"), (0., 0.))
        self.assertEqual(segmentation.evaluate_supervised_epoch(model, [], None, "cpu"), (0., 0.))

    def test_label_loading_and_binary_wrapper_types(self):
        old = reference("npc_sota_train", {"_load_label_volume"})
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            volumes = [np.arange(24).reshape(2, 3, 4).astype(np.int16),
                       np.ones((7, 2, 3, 4), dtype=np.int16)]
            for index, volume in enumerate(volumes):
                path = root / f"label{index}.nrrd"
                nrrd.write(str(path), volume)
                np.testing.assert_array_equal(old["_load_label_volume"](path), load_label_volume(path))
            path = root / "label.nii.gz"
            nib.save(nib.Nifti1Image(volumes[0], np.eye(4)), path)
            np.testing.assert_array_equal(old["_load_label_volume"](path), load_label_volume(path))
        for dtype in (torch.int16, torch.int64, torch.float32):
            labels = torch.arange(6, dtype=dtype)
            image = torch.zeros(1)
            result = BinaryYDataset([(image, labels, "meta")])[0]
            self.assertIs(result[0], image)
            self.assertEqual(result[1].dtype, dtype)
            self.assertEqual(result[1].tolist(), [0, 1, 1, 1, 1, 0])
            self.assertEqual(result[2], "meta")
            base = {"label": labels, "label_meta": None, "image": image}
            result = BinaryYDataset([base])[0]
            self.assertIsNot(result, base)
            self.assertIs(base["label"], labels)
            self.assertEqual(result["label"].dtype, torch.int64)
            self.assertIsNone(result["label_meta"])

    def test_validation_crop_randomstate_sequence(self):
        old = reference(CASCADE, {"validate_stage1_patch"})
        function = next(node for node in ast.parse(subprocess.check_output(
            ["git", "-C", str(ROOT), "show", "85a6f2d:trainer/" + CASCADE + ".py"], text=True,
        )).body if getattr(node, "name", None) == "validate_stage1_patch")
        case_loop = next(node for node in function.body if isinstance(node, ast.For))
        start = next(i for i, node in enumerate(case_loop.body) if isinstance(node, ast.Try))
        block = []
        for node in case_loop.body[start:]:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "preds" for t in node.targets):
                break
            block.append(node)
        for case_id in ("123", "s"):
            for foreground in (False, True):
                for probability in (0., 0.33, 1.):
                    namespace = dict(old, label=np.full((6, 7, 8), int(foreground)),
                                     case_id=case_id, pd=3, ph=4, pw=5, d=6, h=7, w=8,
                                     patches_per_case=9, random_crop_prob=probability)
                    exec(compile(ast.Module(body=block, type_ignores=[]), "reference-crops", "exec"), namespace)
                    self.assertEqual(namespace["positions"], sample_validation_patch_positions(
                        namespace["label"], case_id, (3, 4, 5), 9, probability))

    def test_cache_arrays_indices_and_reuse(self):
        old = reference(CASCADE, {"prepare_preprocessed_cache"})
        from topocnv.data.utils import build_preprocessed_cache
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            image = np.arange(120, dtype=np.float32).reshape(5, 6, 4)
            labels = (image % 6).astype(np.int16)
            paths = [root / "ct.nii.gz", root / "label.nii.gz"]
            for array, path in zip((image, labels), paths):
                nib.save(nib.Nifti1Image(array, np.diag([1.2, 1.1, 2.5, 1.])), path)
            cases = [("s", *paths)]
            for index, spacing in enumerate((None, (2.5, 1.2, 1.1))):
                left, right = root / f"old{index}", root / f"new{index}"
                old["_get_cache_dir"] = lambda _spacing: left
                with redirect_stdout(io.StringIO()):
                    old["prepare_preprocessed_cache"](cases, spacing)
                    build_preprocessed_cache(cases, spacing, right,
                        load_label_fn=old["_load_label_as_int3d"], resample_fn=old["pp"].resample_image_to_spacing)
                for field in ("image", "label"):
                    a, b = np.load(left / f"s_{field}.npy"), np.load(right / f"s_{field}.npy")
                    self.assertEqual(a.dtype, b.dtype)
                    np.testing.assert_array_equal(a, b)
                with np.load(left / "s_fg_indices.npz") as a, np.load(right / "s_fg_indices.npz") as b:
                    for field in ("shape", "indices"):
                        np.testing.assert_array_equal(a[field], b[field])
                with patch("nibabel.load", side_effect=AssertionError("缓存命中不读取原始数据")), redirect_stdout(io.StringIO()):
                    self.assertEqual(build_preprocessed_cache(cases, spacing, right,
                        load_label_fn=None, resample_fn=None), right)

    def test_sliding_validation_files_and_metric_equivalence(self):
        old = reference("npc_baseline", {"validate_with_sliding_window", "compute_dice_per_class"})
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            (root / "image").mkdir()
            (root / "label").mkdir()
            volume = (np.arange(120).reshape(5, 6, 4) % 6).astype(np.int16)
            nib.save(nib.Nifti1Image(volume, np.eye(4)), root / "image/s.nii.gz")
            nrrd.write(str(root / "label/s.nrrd"), volume)
            left, right = root / "old", root / "new"
            with patch.object(old["pp"], "sliding_window_inference", return_value=volume.transpose(2, 0, 1)), redirect_stdout(io.StringIO()):
                expected = old["validate_with_sliding_window"](nn.Identity(), str(root), 0, str(left), (3, 4, 5), 6, "cpu", num_folds=1)
                actual = segmentation.validate_with_sliding_window(nn.Identity(), str(root), 0, str(right), (3, 4, 5), 6, "cpu", num_folds=1)
            self.assertEqual(expected, actual)
            for filename in ("validation_metrics.csv", "cv_results.json"):
                self.assertEqual((left / filename).read_bytes(), (right / filename).read_bytes())
            np.testing.assert_array_equal(nib.load(left / "s_pred.nii.gz").get_fdata(), nib.load(right / "s_pred.nii.gz").get_fdata())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
