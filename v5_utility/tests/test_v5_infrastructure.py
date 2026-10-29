"""任务26只提供测试，交给独立 Verify 执行；所有数据均为临时合成数组。"""
from __future__ import annotations

import copy
import importlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List
from unittest import mock

from v5_utility.protocol import (BASE_PLANS_SHA256, DATASETS, RECIPE, ProtocolError,
                                 build_cohort_split, canonical_json, digest, freeze_protocol,
                                 guard_child, guard_root, load_protocol, run_directory,
                                 validate_cohort_split)
from v5_utility.randomness import (initialize_seed_runtime, initialize_worker,
                                   runtime_environment, seed_cpu, seed_sequence)


class SyntheticDataset:
    """可 spawn 序列化的内存数据集；没有真实患者和文件读取。"""
    def __init__(self, channels: int = 3) -> None:
        self.channels = channels

    def keys(self) -> List[str]:
        return [f"synthetic_{i}" for i in range(10)]

    def load_case(self, key: str) -> Any:
        import numpy as np
        index = int(key.rsplit("_", 1)[1])
        ct = np.arange(14 ** 3, dtype=np.float32).reshape(14, 14, 14) / (14 ** 3) + index / 20
        image = ct[None] if self.channels == 1 else np.stack((ct, ct[::-1].copy(), np.ones_like(ct)))
        seg = np.zeros((1, 14, 14, 14), dtype=np.int16)
        seg[:, 4:10, 4:10, 4:10] = 1
        return image, seg, {"class_locations": {1: np.array([[0, 6, 6, 6]], dtype=np.int64)}}


class RNGProbeLoader:
    stream_fold = 2
    stream_epoch = 3
    stream_training = True

    def set_thread_id(self, worker: int) -> None:
        self.worker = worker
        initialize_worker(self.stream_fold, self.stream_epoch, worker, "train_sampling")

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Dict[str, Any]:
        import numpy as np
        import torch
        return {"worker": self.worker, "python": random.random(), "numpy": float(np.random.random()),
                "torch": float(torch.rand(())), "cuda_initialized": torch.cuda.is_initialized(),
                "threads": torch.get_num_threads()}


class StopAfterEpoch(Exception):
    pass


class QueueTrace:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.consumed = 0
        self.finished = False

    def __next__(self) -> Any:
        batch = next(self.inner)
        self.consumed += 1
        return batch

    def _finish(self) -> None:
        self.inner._finish()
        self.finished = True


def split_fixture() -> Any:
    cohort = [f"synthetic_{i}" for i in range(10)]
    report, dev = [], []
    for k in range(5):
        val = cohort[2 * k:2 * k + 2]
        train = [case for case in cohort if case not in val]
        report.append({"train": train, "val": val})
        dev.append({"train": train[:-2], "val": train[-2:]})
    return cohort, build_cohort_split(cohort, report, dev)


def tiny_plans() -> Dict[str, Any]:
    return {"dataset_name": "SyntheticOnly", "plans_name": "nnUNetPlans", "label_manager": "LabelManager",
            "transpose_forward": [0, 1, 2], "transpose_backward": [0, 1, 2],
            "configurations": {"3d_fullres": {"data_identifier": "synthetic", "batch_size": 2,
                "patch_size": [12, 12, 12], "spacing": list(RECIPE["spacing"]), "batch_dice": True,
                "normalization_schemes": ["NoNormalization"], "use_mask_for_norm": [False],
                "architecture": {"network_class_name": "SyntheticOnly", "arch_kwargs": {
                    "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]]}, "_kw_requires_import": []}}}}


def tiny_network(architecture: str, kwargs: dict, imports: list, channels: int,
                 heads: int, deep_supervision: bool) -> Any:
    import torch

    class TinyNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv3d(channels, heads, 1)
            self.drop = torch.nn.Dropout3d(0.1)
            self.decoder = SimpleNamespace(deep_supervision=deep_supervision)

        def forward(self, image: Any) -> Any:
            out = self.conv(self.drop(image))
            return [out, torch.nn.functional.avg_pool3d(out, 2)] if self.decoder.deep_supervision else out

    return TinyNetwork()


@contextmanager
def tiny_environment() -> Iterator[Any]:
    """仅测试内 mock 来源和大网络；保留真实 constructor/loader/step/checkpoint 接线。"""
    with tempfile.TemporaryDirectory(prefix="v5-utility-test-") as temporary, ExitStack() as stack:
        root = Path(temporary)
        environment = runtime_environment(root, True)
        environment.update({"V5_UTILITY_BOOTSTRAPPED": str(root), "CUDA_VISIBLE_DEVICES": "",
                            "MPLCONFIGDIR": str(root / ".matplotlib")})
        stack.enter_context(mock.patch.dict(os.environ, environment))
        initialize_seed_runtime(root=root, fold=0, test_root=True, cuda=False)
        trainer_module = importlib.import_module("v5_utility.trainer")
        stock_module = importlib.import_module("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")
        stack.enter_context(mock.patch.object(stock_module, "nnUNet_preprocessed", str(root / "nnUNet_preprocessed")))
        stack.enter_context(mock.patch.object(stock_module, "nnUNet_results", str(root / "nnUNet_results")))
        stack.enter_context(mock.patch.dict(RECIPE, {"epochs": 2, "train_iterations": 2,
                                                    "validation_iterations": 1, "iteration_budget": 4}))
        stack.enter_context(mock.patch("v5_utility.protocol._base_plans", return_value=tiny_plans()))
        stack.enter_context(mock.patch("v5_utility.protocol.upstream_fingerprint", return_value={"synthetic": True}))
        stack.enter_context(mock.patch.object(trainer_module, "validate_training_inputs"))
        stack.enter_context(mock.patch.object(trainer_module, "verify_sources"))
        traces = []
        original_factory = trainer_module.ordered_augmenter

        def traced_factory(*args: Any, **kwargs: Any) -> Any:
            queue = QueueTrace(original_factory(*args, **kwargs))
            traces.append(queue)
            return queue

        stack.enter_context(mock.patch.object(trainer_module, "ordered_augmenter", side_effect=traced_factory))
        prepared = stack.enter_context(mock.patch.object(trainer_module, "prepare_modalities",
                                                          wraps=trainer_module.prepare_modalities))
        trainer_class = trainer_module.V5Trainer
        stack.enter_context(mock.patch.object(trainer_class, "_assert_recipe"))
        build = stack.enter_context(mock.patch.object(trainer_class, "build_network_architecture", side_effect=tiny_network))
        stack.enter_context(mock.patch.object(trainer_class, "get_tr_and_val_datasets",
                                               return_value=(SyntheticDataset(), SyntheticDataset())))
        cohort, splits = split_fixture()
        split_path = root / "synthetic_splits.json"
        split_path.write_text(canonical_json(splits), encoding="utf-8")
        protocol = freeze_protocol(arm="F", fold=0, mode="report", cohort=cohort,
                                   splits_file=split_path, gt_sha256="0" * 64, root=root,
                                   test_root=True, plans_source=root / "mock_source.json")
        # 首次导入后再初始化，消除测试进程模块缓存造成的非训练随机消费差别。
        initialize_seed_runtime(root=root, fold=0, test_root=True, cuda=False)
        yield SimpleNamespace(root=root, protocol=protocol, cls=trainer_class, module=trainer_module,
                              build=build, traces=traces, prepared=prepared)


def construct(fixture: Any, resume: bool = False) -> Any:
    import torch
    plans = fixture.protocol.plans()
    plans["v5_utility"] = {"protocol": fixture.protocol.envelope(), "train_authorized": True, "resume": resume}
    return fixture.cls(plans, "3d_fullres", fixture.protocol.split_index,
                       fixture.protocol.to_dict()["dataset_json"], unpack_dataset=False, device=torch.device("cpu"))


class V5InfrastructureTests(unittest.TestCase):
    def assertTreeEqual(self, first: Any, second: Any) -> None:
        import torch
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assertTreeEqual(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assertTreeEqual(a, b)
        else:
            self.assertEqual(first, second)

    def test_split_contract_rejects_missing_out_of_range_duplicate_overlap_and_coverage(self) -> None:
        cohort, splits = split_fixture()
        self.assertEqual(validate_cohort_split(cohort, splits, 2, "dev"),
                         (splits[7]["train"], splits[7]["val"]))
        for candidate in (None, [], splits[:5]):
            with self.assertRaises(ProtocolError):
                validate_cohort_split(cohort, candidate)
        for fold in (-1, 5, "all", True):
            with self.assertRaises(ProtocolError):
                validate_cohort_split(cohort, splits, fold)
        broken = []
        for key, operation in (("train", "duplicate"), ("val", "overlap"),
                               ("train", "missing"), ("val", "unknown")):
            case = copy.deepcopy(splits)
            if operation == "duplicate":
                case[0][key].append(case[0][key][0])
            elif operation == "overlap":
                case[0][key].append(case[0]["train"][0])
            elif operation == "missing":
                case[0][key].pop()
            else:
                case[0][key].append("nonexistent_synthetic")
            broken.append(case)
        for candidate in broken:
            with self.assertRaises(ProtocolError):
                validate_cohort_split(cohort, candidate)

    def test_actual_upstream_source_pins_and_baseline_hash(self) -> None:
        from v5_utility.protocol import BASE_PLANS, file_sha256, upstream_fingerprint
        fingerprint = upstream_fingerprint()
        self.assertEqual(fingerprint["versions"]["nnunetv2"], "2.5.2")
        self.assertEqual(file_sha256(BASE_PLANS), BASE_PLANS_SHA256)

    def test_protocol_roundtrip_hash_and_source_gate(self) -> None:
        with tiny_environment() as f:
            self.assertEqual(load_protocol(f.protocol.envelope()).sha256, f.protocol.sha256)
            edited = f.protocol.envelope()
            edited["protocol"]["recipe"]["training_seed"] = 43
            edited["sha256"] = digest(edited["protocol"])
            with self.assertRaises(ProtocolError):
                load_protocol(edited)
            edited = f.protocol.envelope()
            edited["sha256"] = "f" * 64
            with self.assertRaises(ProtocolError):
                load_protocol(edited)
            from v5_utility.protocol import verify_sources
            with mock.patch("v5_utility.protocol.local_fingerprint", return_value={"changed": "0" * 64}):
                with self.assertRaises(ProtocolError):
                    verify_sources(f.protocol)
            self.assertEqual(f.protocol.to_dict()["plans_source_sha256"], BASE_PLANS_SHA256)

    def test_path_and_constructor_guards_precede_stock_side_effects(self) -> None:
        for root in ("/", "/tmp", "/usr", "/cfs/zzkj/home/dkq/cnv/nnunet_cnv",
                     "/cfs/zzkj/home/dkq/cnv/nnunet_cnv/v5-utility-test-x"):
            with self.assertRaises(ProtocolError):
                guard_root(root, True)
        with tiny_environment() as f:
            outside = f.root / "escaped"
            outside.symlink_to("/tmp", target_is_directory=True)
            with self.assertRaises(ProtocolError):
                guard_child(f.root, outside / "file")
            import torch
            with mock.patch.object(f.module.nnUNetTrainer, "__init__", side_effect=AssertionError("stock called")) as stock:
                with self.assertRaises(ProtocolError):
                    f.cls({}, "3d_fullres", 0, {}, device=torch.device("cpu"))
                stock.assert_not_called()
            self.assertFalse(run_directory(f.protocol).exists())

    def test_missing_split_never_calls_stock_fallback(self) -> None:
        with tiny_environment() as f:
            trainer = object.__new__(f.cls)
            trainer.preprocessed_dataset_folder_base = str(f.root)
            trainer.preprocessed_dataset_folder = str(f.root / "missing")
            trainer.v5 = f.protocol.to_dict()
            with self.assertRaisesRegex(ProtocolError, "split"):
                trainer.do_split()
            self.assertFalse((f.root / "splits_final.json").exists())

    def test_cpu_network_initialization_repeats_with_seed42(self) -> None:
        import torch
        with tiny_environment():
            states = []
            for _ in range(2):
                seed_cpu(seed_sequence(0, 0, 0, "main"))
                model = tiny_network("mock", {}, [], 3, 2, True)
                states.append(copy.deepcopy(model.state_dict()))
            self.assertTreeEqual(states[0], states[1])
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.backends.cudnn.benchmark)

    def test_ordered_multiworker_python_numpy_torch_streams(self) -> None:
        with tiny_environment():
            from v5_utility.augmentation import ordered_augmenter
            runs = []
            for _ in range(2):
                queue = ordered_augmenter(RNGProbeLoader(), 4)
                try:
                    runs.append([next(queue) for _ in range(12)])
                finally:
                    queue._finish()
            self.assertEqual(runs[0], runs[1])
            self.assertEqual([item["worker"] for item in runs[0]], [0, 1, 2, 3] * 3)
            self.assertEqual(len({item["python"] for item in runs[0][:4]}), 4)
            self.assertTrue(all(not item["cuda_initialized"] and item["threads"] == 1 for item in runs[0]))

    def test_real_loader_common_augmentation_and_f_extra_rng_are_isolated(self) -> None:
        import numpy as np
        import torch
        with tiny_environment() as f:
            from v5_utility.augmentation import SeededDataLoader3D, SeededTransforms, ordered_augmenter, prepare_modalities
            label_manager = SimpleNamespace(all_labels=[0, 1], has_ignore_label=False)
            outputs = []
            for arm in ("C", "MM", "F"):
                transform = f.module.nnUNetTrainer.get_training_transforms(
                    [12, 12, 12], (-0.1, 0.1), None, (0, 1, 2), False,
                    use_mask_for_norm=[False] * 3, foreground_labels=[1])
                loader = SeededDataLoader3D(SyntheticDataset(1 if arm == "C" else 3), 2,
                    [14, 14, 14], [12, 12, 12], label_manager, oversample_foreground_percent=0.33,
                    transforms=SeededTransforms(transform, 0, 0, True), stream_fold=0, stream_epoch=0, stream_training=True)
                queue = ordered_augmenter(loader, 4)
                batches = []
                try:
                    for _ in range(8):
                        batch = next(queue)
                        batches.append((list(batch["keys"]), batch["data"][:, :1].clone(), batch["target"].clone()))
                        if arm == "F":
                            rng = np.random.default_rng(seed_sequence(0, 0, 0, "F_dropout"))
                            for _ in range(3):
                                prepare_modalities(batch["data"], "F", training=True, rng=rng)
                finally:
                    queue._finish()
                outputs.append(batches)
            self.assertTreeEqual(outputs[0], outputs[1])
            self.assertTreeEqual(outputs[1], outputs[2])
            self.assertTrue(all(isinstance(row[1], torch.Tensor) for row in outputs[0]))

    def test_valid_and_fixed_f_dropout(self) -> None:
        import numpy as np
        import torch
        from v5_utility.augmentation import draw_drop, keep_mask, prepare_modalities
        image = torch.ones((2, 3, 16, 12, 12))
        image[:, 2, :8] = 0.4
        masked = prepare_modalities(image, "MM")
        self.assertEqual(masked[:, 1:, :8].count_nonzero().item(), 0)
        self.assertTrue(torch.equal(masked[:, 0], image[:, 0]))
        self.assertEqual(prepare_modalities(image, "F", no_mri=True)[:, 1:].count_nonzero().item(), 0)
        rng = np.random.default_rng(42)
        whole = draw_drop(rng, [16, 12, 12], 1, 1, 1)
        self.assertEqual(whole, {"case": True})
        self.assertEqual(keep_mask(whole, [16, 12, 12], RECIPE["spacing"]).sum().item(), 0)
        shaped = draw_drop(rng, [16, 12, 12], 0, 1, 1)
        self.assertGreaterEqual(shaped["trunc"]["tilt"], 0)
        self.assertLessEqual(shaped["trunc"]["tilt"], np.deg2rad(20))
        self.assertIn(len(shaped["slabs"]), (1, 2))
        self.assertTrue(all(4 <= end - start <= 10 for start, end in shaped["slabs"]))
        result = prepare_modalities(torch.ones_like(image), "F", training=True, rng=rng)
        self.assertTrue(torch.equal(result[:, 1], result[:, 2]))
        self.assertTrue(torch.equal(result[:, 0], image[:, 0]))

    def test_amp_counts_actual_optimizer_step_and_does_not_extend_budget(self) -> None:
        import torch
        with tiny_environment():
            from v5_utility.trainer import OptimizerUpdates
            weight = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.SGD([weight], lr=0.01)
            counter = OptimizerUpdates()
            hook = optimizer.register_step_post_hook(counter.hook)
            scaler = torch.amp.GradScaler("cpu", init_scale=8, growth_interval=100)
            try:
                for overflow in (False, True, False):
                    before, scale = counter.successful_steps, scaler.get_scale()
                    optimizer.zero_grad(set_to_none=True)
                    loss = weight.sum() * (float("inf") if overflow else 1.0)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    counter.finish_iteration(before, scaler, scale)
                self.assertEqual(counter.successful_steps, 2)
                self.assertEqual(counter.amp_skipped_steps, 1)
                self.assertEqual(counter.amp_overflows, 1)
                self.assertEqual(counter.successful_steps + counter.amp_skipped_steps, 3)
            finally:
                hook.remove()

    def test_actual_trainer_epoch_resume_matches_uninterrupted(self) -> None:
        import torch
        from v5_utility.randomness import capture_rng
        snapshots = []
        for interrupted in (False, True):
            with tiny_environment() as f, redirect_stdout(io.StringIO()):
                trainer = construct(f)
                original_epoch_end = f.cls.on_epoch_end
                if interrupted:
                    def stop_after_first(instance: Any) -> None:
                        original_epoch_end(instance)
                        if instance.current_epoch == 1:
                            raise StopAfterEpoch()
                    with mock.patch.object(f.cls, "on_epoch_end", stop_after_first):
                        with self.assertRaises(StopAfterEpoch):
                            trainer.run_training()
                    self.assertIsNone(trainer.dataloader_train)
                    self.assertIsNone(trainer.dataloader_val)
                    checkpoint = run_directory(f.protocol) / "checkpoint_latest.pth"
                    saved = f.module.read_checkpoint(checkpoint, f.protocol)
                    self.assertEqual(saved["next_epoch"], 1)
                    self.assertEqual(saved["completed_epoch"], 0)
                    self.assertEqual(saved["global_iteration"], 2)
                    for mutation in ("schema", "hash", "rng", "partial"):
                        bad = copy.deepcopy(saved)
                        if mutation == "schema":
                            bad["schema"] = "old-R0u"
                        elif mutation == "hash":
                            bad["identity"]["plans_sha256"] = "f" * 64
                        elif mutation == "rng":
                            del bad["rng"]["F_generator"]
                        else:
                            bad["global_iteration"] -= 1
                        with self.assertRaises(ProtocolError):
                            f.module.validate_checkpoint(bad, f.protocol)
                    initialize_seed_runtime(root=f.root, fold=0, test_root=True, cuda=False)
                    trainer = construct(f, resume=True)
                    trainer.load_checkpoint(str(checkpoint))
                trainer.run_training()
                f.build.assert_called()
                self.assertEqual(f.build.call_args.args[3], 3)
                self.assertEqual(trainer.global_iteration, 4)
                self.assertEqual(trainer.updates.successful_steps, 4)
                self.assertEqual([queue.consumed for queue in f.traces], [3, 2, 3, 2])
                self.assertTrue(all(queue.finished and not queue.inner._processes for queue in f.traces))
                self.assertEqual([queue.inner.num_processes for queue in f.traces], [4, 1, 4, 1])
                self.assertEqual(sum(call.kwargs.get("training", False) for call in f.prepared.call_args_list), 4)
                self.assertEqual(f.prepared.call_count, 6)
                self.assertIsNone(trainer.dataloader_train)
                self.assertIsNone(trainer.dataloader_val)
                final = run_directory(f.protocol) / "checkpoint_final.pth"
                saved = f.module.read_checkpoint(final, f.protocol)
                self.assertEqual(saved["identity"]["checkpoint_selection"], "final")
                with self.assertRaises(ProtocolError):
                    trainer.save_checkpoint(str(final))
                trainer._at_epoch_boundary = False
                with self.assertRaises(ProtocolError):
                    trainer.checkpoint_state()
                snapshots.append(copy.deepcopy({"network": trainer.network.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(), "scheduler": trainer.lr_scheduler.state_dict(),
                    "rng": f.module._safe_state(capture_rng(trainer.drop_rng)),
                    "losses": trainer.logger.my_fantastic_logging["train_losses"]}))
                self.assertTrue(all(torch.isfinite(t).all() for t in trainer.network.parameters()))
        self.assertTreeEqual(snapshots[0], snapshots[1])

    def test_cli_import_and_help_have_no_directory_or_environment_side_effects(self) -> None:
        parent = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="v5-utility-test-") as temporary:
            root = Path(temporary)
            env = dict(os.environ)
            env.update({"PYTHONPATH": str(parent), "PYTHONDONTWRITEBYTECODE": "1",
                        "HOME": str(root), "MPLCONFIGDIR": str(root / "mpl"), "PYTHONHASHSEED": "42"})
            commands = [
                ["-c", "import os,sys; before=dict(os.environ); import v5_utility, v5_utility.launch; "
                       "assert 'torch' not in sys.modules; assert 'nnunetv2' not in sys.modules; assert before==dict(os.environ)"],
                ["-m", "v5_utility.launch", "--help"],
                ["-m", "v5_utility.launch", "train", "--help"],
            ]
            for arguments in commands:
                result = subprocess.run([sys.executable, "-B", *arguments], cwd=root, env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
