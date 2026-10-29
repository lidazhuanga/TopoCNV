"""v4 迁移回归：旧导出、训练配方、checkpoint 与真实滑窗数值兼容。"""

import copy
import importlib
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from topocnv.engine import v4_inference as inference
from topocnv.engine.checkpoint import save_ckpt
from topocnv.engine.trainer import make_optim, train_epoch, train_one_epoch
from topocnv.utils.seed import seed_everything


class Pointwise(torch.nn.Module):
    """确定性逐体素模型，滑窗融合不应改变其输出。"""

    def __init__(self, n_out, sequence=False):
        super().__init__()
        self.n_out = n_out
        self.sequence = sequence

    def forward(self, x):
        ct = x[:, :1]
        out = torch.cat([ct * (k + 1) - 0.4 * k for k in range(self.n_out)], dim=1)
        return (out,) if self.sequence else out


def reference_epoch(model, loader, optimizer, step_loss, device):
    """固定迁移前的更新顺序、非有限 loss 策略和按 batch 平均口径。"""
    model.train()
    sums, n = {}, 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda", dtype=inference.AMP_DTYPE or torch.float16,
            enabled=inference.AMP_DTYPE is not None and device.type == "cuda",
        ):
            loss, parts = step_loss(model, batch)
        if not torch.isfinite(loss):
            print("  [WARN] non-finite loss, batch skipped", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        for key, value in parts.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        n += 1
    return {key: value / max(n, 1) for key, value in sums.items()}


class TestCompatibility(unittest.TestCase):
    def test_old_exports_are_canonical_objects(self):
        targets = {
            "common": ("data.utils", "config.loader", "utils.box_ops"),
            "data": ("data.v4_dataset",),
            "engine": ("engine.trainer", "engine.checkpoint", "engine.v4_inference", "utils.seed"),
            "infer": ("engine.v4_inference",),
            "losses": ("losses.seg_losses", "losses.visibility_loss", "losses.skeleton_losses"),
            "model": ("models.builder",),
            "priors": ("priors.atlas", "priors.builders", "priors.icp"),
            "roi": ("priors.roi",),
            "splits": ("data.v4_splits",),
            "harmonize": ("data.harmonize",),
            "train_stage1": ("scripts.v4_train_stage1",),
            "train_stage2": ("scripts.v4_train_stage2",),
        }
        for legacy, module_names in targets.items():
            old = importlib.import_module(f"trainer.v4.{legacy}")
            modules = [importlib.import_module(f"topocnv.{name}") for name in module_names]
            for name in old.__all__:
                with self.subTest(module=legacy, name=name):
                    owners = [module for module in modules if hasattr(module, name)]
                    self.assertTrue(owners)
                    self.assertIs(getattr(old, name), getattr(owners[0], name))

    def test_public_package_exports(self):
        for name in ("config", "data", "engine", "losses", "models", "priors", "utils"):
            module = importlib.import_module(f"topocnv.{name}")
            for symbol in module.__all__:
                with self.subTest(module=name, symbol=symbol):
                    self.assertTrue(hasattr(module, symbol))

    def test_old_and_new_cli_help_match(self):
        for stage in (1, 2):
            outputs = []
            for prefix in ("trainer.v4.train_stage", "topocnv.scripts.v4_train_stage"):
                result = subprocess.run(
                    [sys.executable, "-m", f"{prefix}{stage}", "--help"],
                    cwd=ROOT, check=True, capture_output=True, text=True,
                )
                self.assertNotIn("RuntimeWarning", result.stderr)
                normalised = result.stdout.replace(f"v4_train_stage{stage}.py", f"train_stage{stage}.py")
                outputs.append(normalised.split())
            self.assertEqual(*outputs)

    def test_dynunet_factory_preserves_rng_and_state_dict(self):
        from monai.networks.nets import DynUNet
        from topocnv.models import build_dynunet, stage2_channels
        from topocnv.models.builder import build_dynunet as canonical_factory

        self.assertIs(build_dynunet, canonical_factory)
        self.assertEqual(stage2_channels({}), (1, 2))
        self.assertEqual(stage2_channels({"priors": ["a", "c"], "vis_head": {"enabled": True}}), (3, 7))
        torch.manual_seed(19)
        ref = DynUNet(
            spatial_dims=3, in_channels=1, out_channels=3,
            kernel_size=[[3, 3, 3]] * 6,
            strides=[[1, 1, 1]] + [[2, 2, 2]] * 5,
            upsample_kernel_size=[[2, 2, 2]] * 5,
            filters=[32, 64, 128, 256, 320, 320], res_block=True,
        )
        expected_rng = torch.get_rng_state()
        torch.manual_seed(19)
        model = build_dynunet(1, 3)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        self.assertEqual(list(model.state_dict()), list(ref.state_dict()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, ref.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(len(model._forward_hooks), len(ref._forward_hooks))

    def test_seed_preserves_numpy_and_torch_recipe(self):
        seed_everything(17)
        got_np, got_torch = np.random.rand(5), torch.rand(5)
        np.random.seed(17)
        torch.manual_seed(17)
        np.testing.assert_array_equal(got_np, np.random.rand(5))
        torch.testing.assert_close(got_torch, torch.rand(5), rtol=0, atol=0)

    def test_training_updates_metrics_and_resume(self):
        torch.manual_seed(8)
        model = torch.nn.Linear(3, 2)
        ref = copy.deepcopy(model)
        opt, sched = make_optim(model, 0.003, 4)
        ref_opt = torch.optim.Adam(ref.parameters(), lr=0.003)
        ref_sched = torch.optim.lr_scheduler.CosineAnnealingLR(ref_opt, T_max=4, eta_min=0.00003)
        batches = [(torch.randn(2, 3), torch.randn(2, 2), False),
                   (torch.randn(1, 3), torch.randn(1, 2), True),
                   (torch.randn(3, 3), torch.randn(3, 2), False)]

        def step(m, batch):
            x, y, bad = batch
            loss = (m(x) - y).square().mean() * (float("nan") if bad else 1.0)
            return loss, {"loss": loss.item(), "component": loss.item() * 0.3}

        for _ in range(2):
            with redirect_stdout(io.StringIO()) as new_log:
                got = train_epoch(model, batches, opt, step, torch.device("cpu"))
            with redirect_stdout(io.StringIO()) as old_log:
                want = reference_epoch(ref, batches, ref_opt, step, torch.device("cpu"))
            self.assertEqual(got, want)
            self.assertEqual(new_log.getvalue(), old_log.getvalue())
            sched.step()
            ref_sched.step()
            self.assertEqual(sched.state_dict(), ref_sched.state_dict())
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, ref.state_dict()[key], rtol=0, atol=0)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "latest_checkpoint.pth"
            state = dict(epoch=2, model_state=model.state_dict(), optimizer_state=opt.state_dict(),
                         scheduler_state=sched.state_dict(), best=0.8, best_epoch=1, no_improve=1)
            save_ckpt(destination, **state)
            self.assertFalse(Path(str(destination) + ".tmp").exists())
            loaded = torch.load(destination, map_location="cpu", weights_only=False)
            self.assertEqual(set(loaded), set(state))
            resumed = copy.deepcopy(model)
            resumed_opt, resumed_sched = make_optim(resumed, 0.003, 4)
            resumed.load_state_dict(loaded["model_state"])
            resumed_opt.load_state_dict(loaded["optimizer_state"])
            resumed_sched.load_state_dict(loaded["scheduler_state"])
            with redirect_stdout(io.StringIO()):
                train_epoch(model, batches, opt, step, torch.device("cpu"))
                train_epoch(resumed, batches, resumed_opt, step, torch.device("cpu"))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
            save_ckpt(destination, epoch=3, model_state=model.state_dict())
            self.assertEqual(set(torch.load(destination, weights_only=False)), {"epoch", "model_state"})

    def test_empty_and_nonfinite_epoch(self):
        model = torch.nn.Linear(1, 1)
        opt, _ = make_optim(model, 0.001, 2)
        self.assertEqual(train_epoch(model, [], opt, None, torch.device("cpu")), {})
        before = copy.deepcopy(model.state_dict())
        with redirect_stdout(io.StringIO()):
            got = train_epoch(model, [None], opt,
                              lambda m, _: (m(torch.ones(1, 1)).sum() * float("inf"), {"loss": float("inf")}),
                              torch.device("cpu"))
        self.assertEqual(got, {})
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_shared_loop_still_rejects_nonfinite_gradient(self):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()
        before = model.weight.detach().clone()
        opt, _ = make_optim(model, 0.001, 2)
        result = train_one_epoch(model, [None], opt,
                                lambda m, _: m(torch.ones(1, 1)).sqrt().sum(), "cpu")
        self.assertEqual(result.optimizer_steps, 0)
        self.assertEqual(result.skipped_batches, 1)
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)

    def test_sliding_window_and_stage1_quantisation(self):
        image = np.random.default_rng(7).random((9, 10, 11), dtype=np.float32)
        device = torch.device("cpu")
        for sequence in (False, True):
            model = Pointwise(3, sequence)
            got = inference.sliding_window_logits(model, image[None], (4, 4, 4), 0.5, device)
            want = Pointwise(3)(torch.from_numpy(image)[None, None])[0]
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
            with patch.object(inference, "load_image", return_value=image):
                p1, p4 = inference.predict_stage1(model, "sample", (4, 4, 4), 0.5, device)
            probabilities = (torch.softmax(got, dim=0)[1:] * 255).round().clamp(0, 255).to(torch.uint8).numpy()
            np.testing.assert_array_equal(p1, probabilities[0])
            np.testing.assert_array_equal(p4, probabilities[1])

    def test_prior_channels_and_shuffle(self):
        shape = (9, 10, 11)
        image = np.zeros(shape, np.float32)
        box = [[2, 6], [1, 7], [3, 8]]
        prior = np.random.default_rng(1).integers(0, 256, (3, 4, 6, 5), dtype=np.uint8)
        full = np.zeros((3,) + shape, np.uint8)
        full[(slice(None),) + tuple(slice(a, b) for a, b in box)] = prior
        region = [[1, 8], [2, 10], [0, 9]]
        sl = tuple(slice(a, b) for a, b in region)
        for shuffle in (False, True):
            cfg = {"priors": ["c", "a"], "prior_shuffle": shuffle}
            with patch.object(inference, "load_prior", return_value=(prior, box)):
                got = inference.stage2_input("sample", 0, cfg, region, image)
            expected = full
            if shuffle:
                from topocnv.utils.box_ops import case_shift
                expected = np.roll(full, case_shift("sample", shape), axis=(1, 2, 3))
            np.testing.assert_array_equal(got[0], image[sl])
            np.testing.assert_array_equal(got[1:], expected[(np.array([2, 0]),) + sl].astype(np.float32) / 255)

    def test_stage2_roi_visibility_and_full_volume_fallback(self):
        image = np.random.default_rng(9).random((9, 10, 11), dtype=np.float32)
        box = [[3, 5], [4, 6], [2, 5]]
        device = torch.device("cpu")
        for use_roi, found in ((False, False), (True, False), (True, True)):
            rec = {"sides": [{"boxes": {"25": box if found else None}}]}
            cfg = {"train": {"patch": [4, 4, 4], "step_size": 0.5},
                   "roi": {"enabled": use_roi, "margin_mm": 25}}
            for n_out in (2, 7):
                model = Pointwise(n_out)
                with patch.multiple(inference, load_image=lambda _: image, load_roi_record=lambda *_: rec):
                    pred, vis, roi_used = inference.predict_stage2(model, "sample", 0, cfg, device)
                logits = model(torch.from_numpy(image)[None, None])[0]
                expected = (torch.softmax(logits[:2], dim=0)[1] > 0.5).numpy()
                expected_vis = logits[2:].argmax(0).numpy().astype(np.uint8) if n_out > 2 else None
                if use_roi and found:
                    mask = np.zeros(image.shape, bool)
                    mask[tuple(slice(a, b) for a, b in box)] = True
                    expected &= mask
                    if expected_vis is not None:
                        expected_vis[~mask] = 0
                self.assertEqual(roi_used, use_roi and found)
                np.testing.assert_array_equal(pred, expected)
                if expected_vis is None:
                    self.assertIsNone(vis)
                else:
                    np.testing.assert_array_equal(vis, expected_vis)


if __name__ == "__main__":
    unittest.main()
