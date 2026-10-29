"""可复用指标的旧协议数值一致性及通用绘图回归测试。"""

import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from MIA.scripts import metrics_protocol as legacy  # noqa: E402
from topocnv import metrics, visualization  # noqa: E402
from topocnv.engine.evaluator import evaluate_segmentation  # noqa: E402
from topocnv.losses.seg_losses import multiclass_dice_score  # noqa: E402


class TestMetricsProtocol(unittest.TestCase):
    def assert_rows_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key, value in expected.items():
            if isinstance(value, str):
                self.assertEqual(actual[key], value, key)
            else:
                np.testing.assert_allclose(actual[key], value, rtol=0, atol=1e-12, equal_nan=True,
                                           err_msg=key)

    def test_protocol_matches_latest_legacy(self):
        rng = np.random.RandomState(42)
        gt = np.zeros((20, 24, 26), dtype=np.uint8)
        gt[3:6, 6:9, 4:7] = 1
        gt[7:10, 6:9, 4:7] = 2
        gt[11:14, 6:9, 4:7] = 3
        gt[15:18, 15:18, 12:15] = 4
        gt[0:2, 20:22, 20:22] = 5
        hu = rng.normal(30, 25, gt.shape).astype(np.float32)
        hu[gt == 4] = 900
        perfect = metrics.mask_Y(gt)
        for pred in (gt, np.zeros_like(gt), np.roll(gt, 2, axis=0), (perfect | (rng.rand(*gt.shape) < 0.01))):
            for raw_hu in (None, hu):
                for binary in (False, True):
                    kwargs = dict(hu=raw_hu, spacing_zyx=(2.5, 1.1, 0.9),
                                  pred_binary_Y=binary, vis_pred=np.where(gt == 5, 0, gt))
                    self.assert_rows_equal(metrics.score_protocol(pred, gt, **kwargs),
                                           legacy.score_protocol(pred, gt, **kwargs))
        posteriors = rng.rand(6, *gt.shape).astype(np.float32)
        partition = metrics.partition_Y_by_posterior(perfect, posteriors)
        np.testing.assert_array_equal(partition, legacy.partition_Y_by_posterior(perfect, posteriors))
        kwargs = dict(hu=hu, pred_binary_Y=True, secondary_pred=partition, secondary_source="s1_inY")
        self.assert_rows_equal(metrics.score_protocol(perfect, gt, **kwargs),
                               legacy.score_protocol(perfect, gt, **kwargs))

    def test_distance_crop_preserves_boundaries_and_spacing(self):
        rng = np.random.RandomState(8)
        for shape in ((1, 7, 9), (8, 8, 8), (4, 5, 6)):
            for _ in range(5):
                p, g = rng.rand(*shape) > 0.6, rng.rand(*shape) > 0.6
                spacing = (2.5, 0.9, 1.1)
                self.assertAlmostEqual(metrics.hd95(p, g, spacing), legacy._hd95(p, g, spacing))
                self.assertAlmostEqual(metrics.assd(p, g, spacing), legacy._assd(p, g, spacing))
                self.assertAlmostEqual(metrics.surface_dice(p, g, spacing), legacy._surface_dice(p, g, spacing))
        p = np.zeros((8, 9), bool)
        p[2, 3] = True
        g = np.roll(p, 2, axis=1)
        self.assertEqual(metrics.hd95(p, g, (3, 2)), 4)
        self.assertEqual(metrics.hausdorff_distance(p, g, (3, 2)), 4)

    def test_empty_masks_and_aggregation(self):
        gt = np.zeros((4, 5, 6), np.uint8)
        gt[2, 2, 2] = 1
        empty = np.zeros_like(gt)
        row = metrics.score_protocol(empty, gt, spacing_zyx=(2.5, 1, 1), pred_binary_Y=True)
        self.assertEqual(row["Dice_Y"], 0)
        self.assertEqual(row["Empty_pred_Y"], 1)
        self.assertAlmostEqual(row["HD95_Y"], np.sqrt(100 + 25 + 36))
        self.assertEqual(row["SurfDice_Y"], 0)
        self.assertTrue(np.isnan(metrics.dice_score(empty, empty)))
        self.assertEqual(metrics.dice_score(empty, empty, empty_value=1), 1)
        self.assertTrue(np.isnan(metrics.hd95(gt, empty)))
        self.assertTrue(np.isnan(metrics.assd(empty, gt)))
        summary = metrics.aggregate_metrics([row, {"HD95_Y": float("nan"), "Secondary_source": "s1"}])
        self.assertEqual(summary["HD95_Y"], row["HD95_Y"])
        self.assertNotIn("Secondary_source", summary)
        self.assertTrue(np.isinf(metrics.aggregate_metrics([{"HD": 2}, {"HD": np.inf}])["HD"]))
        self.assertTrue(np.isnan(metrics.aggregate_metrics([], keys=["missing"])["missing"]))

    def test_binary_scores_and_reexports(self):
        p = np.array([[1, 1], [0, 0]])
        g = np.array([[1, 0], [1, 0]])
        self.assertEqual(metrics.dice_score(p, g), 0.5)
        self.assertAlmostEqual(metrics.iou_score(p, g), 1 / 3)
        self.assertEqual(metrics.precision_score(p, g), 0.5)
        self.assertEqual(metrics.recall_score(p, g), 0.5)
        self.assertIs(metrics.evaluate_segmentation, evaluate_segmentation)
        self.assertIs(metrics.multiclass_dice_score, multiclass_dice_score)
        result = metrics.evaluate_segmentation(p, g, num_classes=3)
        self.assertEqual(result.mean_dice, 0.5)
        self.assertEqual(result.present, (True, False))

    def test_weighted_dice_preserves_false_positive_cost(self):
        gt = np.zeros((4, 4, 4), np.uint8)
        gt[0, 0, :4] = 3
        p = gt > 0
        p[2, 2, :4] = True
        for lam in (0.25, 0.5, 0.75, 1):
            self.assertAlmostEqual(metrics.weighted_dice_y(p, gt, lam), legacy._wdice(p, gt, lam))
        self.assertEqual(metrics.recall_gt3(p, gt), 1)

    def test_validation_prevents_broadcast_and_invalid_spacing(self):
        p = np.ones((2, 3, 4), bool)
        for call in (metrics.dice_score, metrics.hd95, metrics.surface_dice, metrics.score_protocol):
            with self.assertRaises(ValueError):
                call(p, np.ones((1, 3, 4), bool))
        for spacing in ((1, 1), (1, 0, 1), (1, np.nan, 1)):
            with self.assertRaises(ValueError):
                metrics.hd95(p, p, spacing)
        with self.assertRaises(ValueError):
            metrics.surface_dice(p, p, tol_mm=-1)
        with self.assertRaises(ValueError):
            metrics.partition_Y_by_posterior(p, np.ones((4, 2, 3, 4)))


class TestVisualizationArrays(unittest.TestCase):
    def test_overlay_alpha_colors_and_input_immutability(self):
        image = np.array([[0.2, 0.4], [0.6, 0.8]], np.float32)
        mask = np.array([[0, 1], [2, 17]])
        original = image.copy()
        overlay = visualization.make_overlay(image, mask, window=(0, 1), alpha=0.5,
                                             colors={1: "#FF0000", 2: (0, 1, 0)})
        self.assertEqual(overlay.shape, (2, 2, 3))
        self.assertEqual(overlay.dtype, np.float32)
        np.testing.assert_allclose(overlay[0, 0], [0.2] * 3)
        np.testing.assert_allclose(overlay[0, 1], [0.7, 0.2, 0.2])
        np.testing.assert_allclose(overlay[1, 0], [0.3, 0.8, 0.3])
        np.testing.assert_array_equal(image, original)
        self.assertTrue(np.all((overlay >= 0) & (overlay <= 1)))

    def test_normalization_handles_hu_mr_constant_and_nonfinite(self):
        image = np.array([[-160, 40, 240, 500, np.nan, np.inf]], np.float32)
        normalized = visualization.normalize_image(image, window=visualization.CT_WINDOW)
        np.testing.assert_allclose(normalized, [[0, 0.5, 1, 1, 0, 0]])
        np.testing.assert_array_equal(visualization.normalize_image(np.full((3, 3), 17)), 0)
        np.testing.assert_array_equal(visualization.normalize_image(np.full((3, 3), np.nan)), 0)
        np.testing.assert_allclose(visualization.normalize_image(np.arange(5), percentiles=(0, 100)),
                                   np.arange(5) / 4)

    def test_slice_selection_is_axis_aware_and_empty_safe(self):
        mask = np.zeros((7, 5, 9), np.uint8)
        self.assertEqual(visualization.select_slice(mask), 3)
        mask[1, :, :2] = 1
        mask[5, :, :2] = 2
        mask[4, 0, 0] = 1
        self.assertEqual(visualization.select_slice(mask), 5)
        self.assertEqual(visualization.select_slice(mask, axis=2), 0)

    def test_validation(self):
        image, mask = np.ones((3, 4)), np.ones((3, 4), bool)
        for kwargs in (dict(alpha=-1), dict(window=(1, 1)), dict(colors={1: "red"})):
            with self.assertRaises(ValueError):
                visualization.make_overlay(image, mask, **kwargs)
        for labels in (np.ones((1, 4)), np.full((3, 4), 0.5), np.full((3, 4), np.nan)):
            with self.assertRaises(ValueError):
                visualization.make_overlay(image, labels)

    def test_numpy_tools_do_not_import_matplotlib(self):
        import builtins
        original_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError("用于测试的缺失可选依赖")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guard):
            visualization.make_overlay(np.ones((2, 2)), np.zeros((2, 2)))
            with self.assertRaisesRegex(ImportError, "matplotlib"):
                visualization.plot_overlay(np.ones((2, 2)), np.zeros((2, 2)))


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "绘图测试需要 matplotlib")
class TestVisualizationFigures(unittest.TestCase):
    def test_segmentation_uses_one_grid_and_physical_aspect(self):
        image = np.arange(4 * 5 * 6).reshape(4, 5, 6)
        target = np.zeros_like(image)
        target[2, 1:3, 1:4] = 1
        prediction = target.copy()
        before = "matplotlib.pyplot" in sys.modules
        fig, axes = visualization.plot_segmentation(image, prediction, target, axis=0,
                                                    spacing=(2.5, 1.2, 0.8), window=(0, 120))
        self.assertEqual(len(axes), 3)
        self.assertAlmostEqual(axes[0].get_aspect(), 1.5)
        self.assertEqual(axes[0].images[0].get_array().shape, (5, 6, 3))
        self.assertEqual(before, "matplotlib.pyplot" in sys.modules)
        output = io.BytesIO()
        fig.savefig(output, format="png", dpi=50)
        self.assertTrue(output.getvalue().startswith(b"\x89PNG"))
        ax = visualization.plot_overlay(image[0], target[0], ax=axes[0])
        self.assertIs(ax, axes[0])

    def test_training_curves_group_and_smooth_unequal_histories(self):
        history = {"train_loss": [3, 2, 1], "val_loss": [4, 2], "val_dice": [0.2, np.nan, 0.8]}
        original = history["train_loss"].copy()
        fig, axes = visualization.plot_training_curves(history, smoothing=2)
        self.assertEqual(len(axes), 2)
        self.assertEqual(len(axes[0].lines), 2)
        np.testing.assert_allclose(axes[0].lines[0].get_ydata(), [3, 2.5, 1.5])
        self.assertTrue(np.isnan(axes[1].lines[0].get_ydata()[1]))
        self.assertEqual(history["train_loss"], original)
        fig.savefig(io.BytesIO(), format="svg")

    def test_explicit_epoch_validation_and_custom_groups(self):
        fig, axes = visualization.plot_training_curves({"dice": [0.3, 0.5]}, epochs=[5, 10],
                                                       groups={"Validation": ["dice"]})
        np.testing.assert_array_equal(axes[0].lines[0].get_xdata(), [5, 10])
        for kwargs in (dict(epochs=[1]), dict(groups={"Missing": ["loss"]}), dict(smoothing=0)):
            with self.assertRaises(ValueError):
                visualization.plot_training_curves({"dice": [0.3, 0.5]}, **kwargs)
        with self.assertRaises(ValueError):
            visualization.plot_segmentation(np.ones((2, 3, 4)), np.ones((2, 3, 4)), slice_index=5)

    def test_save_figure_delegates_without_closing(self):
        fig = Mock()
        output = REPO / "results" / "visualization_test" / "overlay.png"
        with patch.object(Path, "mkdir") as mkdir:
            self.assertEqual(visualization.save_figure(fig, output, dpi=100), output)
        mkdir.assert_called_once_with(parents=True, exist_ok=True)
        fig.savefig.assert_called_once_with(output, dpi=100, bbox_inches="tight")
        with self.assertRaises(ValueError):
            visualization.save_figure(fig, output, dpi=0)


if __name__ == "__main__":
    unittest.main()
