"""显式导入的独立 V5 trainer；保留上游网络/损失/优化算法，仅替换基础设施。"""
from __future__ import annotations

import copy
import fcntl
import importlib
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time
from typing import Any, Dict, List, Mapping, Tuple, Union

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from .augmentation import SeededDataLoader3D, SeededTransforms, ordered_augmenter, prepare_modalities
from .protocol import (CHECKPOINT_SCHEMA, DATASETS, RECIPE, STREAM_SCHEME, ProtocolError,
                       V5Protocol, canonical_json, file_sha256, guard_child, load_protocol,
                       read_json, run_directory, validate_cohort_split, validate_training_inputs,
                       verify_sources)
from .randomness import (capture_rng, cpu_rng_scope, require_runtime_environment,
                         restore_rng, seed_sequence)


@dataclass
class OptimizerUpdates:
    successful_steps: int = 0
    amp_skipped_steps: int = 0
    amp_overflows: int = 0

    def hook(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        self.successful_steps += 1

    def finish_iteration(self, before: int, scaler: Any, old_scale: Any) -> None:
        delta = self.successful_steps - before
        if delta not in (0, 1) or (scaler is None and delta != 1):
            raise ProtocolError("optimizer 实际 step hook 计数异常")
        if scaler is not None:
            self.amp_skipped_steps += int(delta == 0)
            self.amp_overflows += int(float(scaler.get_scale()) < float(old_scale))


def _safe_state(value: Any) -> Any:
    """checkpoint 仅存 Torch weights_only 支持的原语，不需要任意 pickle 执行。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _safe_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_safe_state(item) for item in value)
    if isinstance(value, list):
        return [_safe_state(item) for item in value]
    if isinstance(value, torch.device):
        return str(value)
    return value


def runtime_signature(device: torch.device) -> Dict[str, Any]:
    signature = {"torch": str(torch.__version__), "device_type": device.type,
                 "cuda_version": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
                 "deterministic": torch.are_deterministic_algorithms_enabled()}
    if device.type == "cuda":
        signature.update({"device_name": torch.cuda.get_device_name(device),
                          "capability": list(torch.cuda.get_device_capability(device)),
                          "cudnn": torch.backends.cudnn.version(), "device_count": torch.cuda.device_count()})
    return signature


def validate_checkpoint(checkpoint: Mapping[str, Any], protocol: V5Protocol) -> None:
    required = {"schema", "identity", "protocol", "stream_scheme", "boundary", "network_weights",
                "optimizer_state", "scheduler_state", "grad_scaler_state", "logging", "_best_ema",
                "current_epoch", "completed_epoch", "next_epoch", "global_iteration", "updates",
                "rng", "init_args", "trainer_name", "inference_allowed_mirroring_axes", "runtime"}
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != required:
        raise ProtocolError("未知或旧 checkpoint schema；不能声明可复现恢复")
    if checkpoint["schema"] != CHECKPOINT_SCHEMA or checkpoint["boundary"] != "before_next_epoch_warmup":
        raise ProtocolError("仅支持 V5 已完成 epoch 边界 checkpoint")
    if checkpoint["identity"] != protocol.identity() or checkpoint["protocol"] != protocol.envelope():
        raise ProtocolError("checkpoint 协议/配置/来源哈希不匹配")
    if checkpoint["stream_scheme"] != STREAM_SCHEME or checkpoint["trainer_name"] != "V5Trainer":
        raise ProtocolError("checkpoint 子流或 trainer 不匹配")
    epoch = checkpoint["next_epoch"]
    if type(epoch) is not int or not 1 <= epoch <= RECIPE["epochs"]:
        raise ProtocolError("checkpoint epoch 越界")
    if checkpoint["current_epoch"] != epoch or checkpoint["completed_epoch"] != epoch - 1:
        raise ProtocolError("checkpoint 当前/下一 epoch 语义不一致")
    iteration = checkpoint["global_iteration"]
    if type(iteration) is not int or iteration != epoch * RECIPE["train_iterations"]:
        raise ProtocolError("checkpoint 不是完整 epoch，拒绝 step 级等价恢复")
    updates = checkpoint["updates"]
    if not isinstance(updates, dict) or set(updates) != set(asdict(OptimizerUpdates())):
        raise ProtocolError("checkpoint 缺少实际更新计数")
    if any(type(v) is not int or v < 0 for v in updates.values()):
        raise ProtocolError("checkpoint 更新计数无效")
    if (updates["successful_steps"] + updates["amp_skipped_steps"] != iteration
            or updates["amp_overflows"] > updates["amp_skipped_steps"]):
        raise ProtocolError("训练迭代/成功更新/AMP skip 计数不一致")
    rng = checkpoint["rng"]
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda", "F_generator"}:
        raise ProtocolError("checkpoint 缺少完整 RNG 状态")
    if not isinstance(rng["torch_cpu"], torch.Tensor) or not isinstance(rng["torch_cuda"], list):
        raise ProtocolError("checkpoint Torch RNG 类型无效")
    if not isinstance(rng["F_generator"], dict) or rng["F_generator"].get("bit_generator") != "PCG64":
        raise ProtocolError("checkpoint F Generator 类型不匹配")
    if any(not isinstance(checkpoint[key], dict) for key in
           ("network_weights", "optimizer_state", "scheduler_state", "logging", "runtime")):
        raise ProtocolError("checkpoint 训练状态不完整")


def read_checkpoint(path: Union[str, Path], protocol: V5Protocol) -> Dict[str, Any]:
    file = guard_child(run_directory(protocol), path)
    if file.parent != run_directory(protocol) or not file.is_file():
        raise ProtocolError("恢复文件必须来自当前 V5 run")
    try:
        checkpoint = torch.load(str(file), map_location="cpu", weights_only=True)
    except Exception:
        raise ProtocolError("checkpoint 安全解包失败；旧/未知 pickle 不作为可复现恢复") from None
    validate_checkpoint(checkpoint, protocol)
    return checkpoint


class V5Trainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device("cuda")):
        # 不能新增 seed 参数：上游按 self.__init__ 签名查找其自身 locals。
        extension = plans.get("v5_utility")
        if not isinstance(extension, dict) or set(extension) != {"protocol", "train_authorized", "resume"}:
            raise ProtocolError("必须通过 V5 显式启动协议构造 trainer")
        self.protocol = load_protocol(extension["protocol"])
        self.v5 = self.protocol.to_dict()
        if extension["train_authorized"] is not True or type(extension["resume"]) is not bool:
            raise ProtocolError("trainer 构造需要显式训练授权")
        clean_plans = {k: v for k, v in plans.items() if k != "v5_utility"}
        if (clean_plans != self.v5["plans"] or dataset_json != self.v5["dataset_json"]
                or configuration != "3d_fullres" or type(fold) is not int or fold != self.protocol.split_index):
            raise ProtocolError("构造参数偏离 V5 协议")
        if device.type not in ("cpu", "cuda") or (device.type == "cpu" and not self.v5["test_root"]):
            raise ProtocolError("生产训练仅支持单 GPU；CPU 仅供显式合成测试")
        if torch.distributed.is_initialized():
            raise ProtocolError("V5-v1 不支持 DDP")
        require_runtime_environment(Path(self.v5["root"]), self.v5["test_root"])
        module = importlib.import_module("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")
        for name in ("nnUNet_preprocessed", "nnUNet_results"):
            if getattr(module, name) != str(Path(self.v5["root"]) / name):
                raise ProtocolError("nnunet 已绑定其他根；必须使用新进程")
        if (not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()
                or torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic
                or torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1):
            raise ProtocolError("必须先 initialize_seed_runtime，禁止非确定性回退")
        validate_training_inputs(self.protocol)
        output = run_directory(self.protocol)
        self._resume_authorized = extension["resume"]
        self._resume_loaded = False
        if self._resume_authorized:
            stamp = guard_child(output, output / "v5_protocol.json")
            if not stamp.is_file() or read_json(stamp) != self.protocol.envelope():
                raise ProtocolError("已有 run 的协议来源不匹配，不能恢复")
            if (output / "checkpoint_final.pth").exists():
                raise ProtocolError("该 V5 run 已有 final，禁止重复训练/覆盖")
        else:
            if output.exists():
                raise ProtocolError("V5 run 已存在；仅同协议显式 --resume 可继续")
            output.mkdir(parents=True, exist_ok=False)
            with (output / "v5_protocol.json").open("x", encoding="utf-8") as stream:
                stream.write(canonical_json(self.protocol.envelope()))
        lock_path = guard_child(output, output / ".v5-training.lock")
        self._run_lock = lock_path.open("a")
        try:
            fcntl.flock(self._run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._run_lock.close()
            raise ProtocolError("当前 V5 run 已被训练进程占用") from None
        # 上游第一次 mkdir/log 只能在上述守卫全部完成以后发生。
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = RECIPE["epochs"]
        self.num_iterations_per_epoch = RECIPE["train_iterations"]
        self.num_val_iterations_per_epoch = RECIPE["validation_iterations"]
        self.global_iteration = 0
        self.updates = OptimizerUpdates()
        self._step_hook = None
        self._at_epoch_boundary = True
        self.drop_rng = np.random.default_rng(seed_sequence(self.v5["fold"], 0, 0, "F_dropout"))
        self._assert_recipe()

    def _do_i_compile(self) -> bool:
        return False

    def _assert_recipe(self) -> None:
        config = self.configuration_manager
        if (self.num_epochs, self.num_iterations_per_epoch, self.num_val_iterations_per_epoch,
            self.batch_size, self.initial_lr, self.weight_decay, self.oversample_foreground_percent,
            self.enable_deep_supervision) != (300, 250, 50, 2, 0.01, 3e-5, 0.33, True):
            raise ProtocolError("trainer 超参数偏离冻结配方")
        if (list(config.patch_size) != RECIPE["patch_size"] or list(config.spacing) != RECIPE["spacing"]
                or list(self.plans_manager.transpose_forward) != [0, 1, 2]
                or self.is_cascaded or self.label_manager.has_regions or self.label_manager.has_ignore_label
                or self.label_manager.num_segmentation_heads != 2):
            raise ProtocolError("plans/标签/空间方向偏离冻结基线")

    def initialize(self) -> None:
        self._assert_recipe()
        super().initialize()
        self._step_hook = self.optimizer.register_step_post_hook(self.updates.hook)

    def do_split(self) -> Tuple[List[str], List[str]]:
        folder = Path(self.preprocessed_dataset_folder_base)
        path = folder / "splits_final.json"
        if not path.is_file():
            raise ProtocolError("缺少 split，禁止 stock 自动生成或 80:20 fallback")
        splits = read_json(path)
        if splits != self.v5["splits"]:
            raise ProtocolError("split 与冻结协议不同")
        actual = sorted(p.stem for p in Path(self.preprocessed_dataset_folder).glob("*.npz"))
        if actual != sorted(self.v5["cohort"]):
            raise ProtocolError("病例覆盖与协议不符")
        return validate_cohort_split(actual, splits, self.v5["fold"], self.v5["mode"])

    def get_dataloaders(self) -> Tuple[Any, Any]:
        patch = self.configuration_manager.patch_size
        scales = self._get_deep_supervision_scales()
        rotation, dummy2d, initial_patch, mirrors = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        train_transform = self.get_training_transforms(
            patch, rotation, scales, mirrors, dummy2d, use_mask_for_norm=[False] * 3,
            is_cascaded=False, foreground_labels=self.label_manager.foreground_labels,
            regions=None, ignore_label=None)
        val_transform = self.get_validation_transforms(
            scales, is_cascaded=False, foreground_labels=self.label_manager.foreground_labels,
            regions=None, ignore_label=None)
        datasets = self.get_tr_and_val_datasets()
        queues = []
        try:
            for training, dataset, transform, size, workers in (
                    (True, datasets[0], train_transform, initial_patch, 4),
                    (False, datasets[1], val_transform, patch, 1)):
                loader = SeededDataLoader3D(
                    dataset, self.batch_size, size, patch, self.label_manager,
                    oversample_foreground_percent=self.oversample_foreground_percent,
                    sampling_probabilities=None, pad_sides=None,
                    transforms=SeededTransforms(transform, self.v5["fold"], self.current_epoch, training),
                    stream_fold=self.v5["fold"], stream_epoch=self.current_epoch, stream_training=training)
                queues.append(ordered_augmenter(loader, workers, pin_memory=self.device.type == "cuda"))
            # 每次创建队列固定各消费一批，恢复时同样消费；这些批次不计训练迭代。
            for queue in queues:
                next(queue)
            return queues[0], queues[1]
        except BaseException:
            for queue in queues:
                queue._finish()
            raise

    def _finish_queues(self) -> None:
        error = None
        for name in ("dataloader_train", "dataloader_val"):
            queue = getattr(self, name, None)
            if queue is not None:
                try:
                    queue._finish()
                except Exception as failure:
                    error = failure
                finally:
                    setattr(self, name, None)
        if error is not None:
            raise error

    def close(self) -> None:
        try:
            self._finish_queues()
        finally:
            self._run_lock.close()

    def on_train_start(self) -> None:
        self._assert_recipe()
        if self._resume_authorized and not self._resume_loaded:
            raise ProtocolError("--resume 尚未成功载入完整 V5 checkpoint")
        if not self.was_initialized:
            self.initialize()
        self.set_deep_supervision_enabled(True)
        # 不调用 stock on_train_start：它会解包写 cache、复制文件并绘图消费 RNG。
        self.print_to_log_file("V5 start", canonical_json(self.protocol.identity()))

    def on_epoch_start(self) -> None:
        if not self._at_epoch_boundary:
            raise ProtocolError("上一个 epoch 未完成，不能在当前实例继续")
        self._finish_queues()
        # 构造/启动队列可能引入主进程 RNG 消费，隔离它以保持恢复契约。
        with cpu_rng_scope():
            self.dataloader_train, self.dataloader_val = self.get_dataloaders()
        self.drop_rng = np.random.default_rng(seed_sequence(self.v5["fold"], self.current_epoch, 0, "F_dropout"))
        self._at_epoch_boundary = False
        super().on_epoch_start()

    def train_step(self, batch: dict) -> dict:
        before = self.updates.successful_steps
        scale = self.grad_scaler.get_scale() if self.grad_scaler is not None else None
        prepared = dict(batch)
        prepared["data"] = prepare_modalities(batch["data"].to(self.device, non_blocking=True),
                                              self.v5["arm"], training=True, rng=self.drop_rng)
        result = super().train_step(prepared)
        self.updates.finish_iteration(before, self.grad_scaler, scale)
        self.global_iteration += 1
        return result

    def validation_step(self, batch: dict) -> dict:
        prepared = dict(batch)
        prepared["data"] = prepare_modalities(batch["data"].to(self.device, non_blocking=True), self.v5["arm"])
        return super().validation_step(prepared)

    def on_epoch_end(self) -> None:
        if self.global_iteration != (self.current_epoch + 1) * self.num_iterations_per_epoch:
            raise ProtocolError("epoch 迭代数量不完整，拒绝保存边界 checkpoint")
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)
        self._finish_queues()
        self.current_epoch += 1
        self._at_epoch_boundary = True
        self.print_to_log_file("V5 epoch complete", canonical_json({
            **self.protocol.identity(), "next_epoch": self.current_epoch,
            "global_iteration": self.global_iteration, **asdict(self.updates)}))
        # 不写 best、不依据验证结果选择模型；每 epoch 可恢复，最终仅使用 final。
        self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_latest.pth"))

    def on_train_end(self) -> None:
        self._finish_queues()
        if self.current_epoch != self.num_epochs or self.global_iteration != RECIPE["iteration_budget"]:
            raise ProtocolError("未完成固定 75000 次训练迭代，不能标记 final")
        final = Path(self.output_folder) / "checkpoint_final.pth"
        self.save_checkpoint(str(final))
        self.print_to_log_file("V5 final", canonical_json({**self.protocol.identity(),
                               "checkpoint": final.name, "checkpoint_sha256": file_sha256(final),
                               "global_iteration": self.global_iteration, **asdict(self.updates)}))

    def run_training(self) -> None:
        try:
            super().run_training()
        finally:
            self.close()

    def checkpoint_state(self) -> Dict[str, Any]:
        if not self._at_epoch_boundary or self.current_epoch == 0:
            raise ProtocolError("仅支持已完成 epoch 边界 checkpoint，不支持任意 step 恢复")
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA, "identity": self.protocol.identity(),
            "protocol": self.protocol.envelope(), "stream_scheme": copy.deepcopy(STREAM_SCHEME),
            "boundary": "before_next_epoch_warmup", "network_weights": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(), "scheduler_state": self.lr_scheduler.state_dict(),
            "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            "logging": self.logger.get_checkpoint(), "_best_ema": self._best_ema,
            "current_epoch": self.current_epoch, "completed_epoch": self.current_epoch - 1,
            "next_epoch": self.current_epoch, "global_iteration": self.global_iteration,
            "updates": asdict(self.updates), "rng": capture_rng(self.drop_rng),
            "init_args": self.my_init_kwargs, "trainer_name": "V5Trainer",
            "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
            "runtime": runtime_signature(self.device),
        }
        checkpoint = _safe_state(checkpoint)
        validate_checkpoint(checkpoint, self.protocol)
        return checkpoint

    def save_checkpoint(self, filename: str) -> None:
        file = guard_child(run_directory(self.protocol), filename)
        if file.parent != run_directory(self.protocol) or file.name not in ("checkpoint_latest.pth", "checkpoint_final.pth"):
            raise ProtocolError("checkpoint 只能写当前 V5 run 的 latest/final")
        if file.name == "checkpoint_final.pth" and file.exists():
            raise ProtocolError("禁止覆盖已有 final checkpoint")
        checkpoint = self.checkpoint_state()
        # 未完成的临时文件不作为恢复来源；中断残留也不阻挡下一次边界保存。
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".v5-checkpoint-", suffix=".pending",
                                         dir=file.parent, delete=False) as stream:
            temporary = stream.name
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(file))
        self.print_to_log_file("V5 checkpoint", file.name, file_sha256(file))

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self._resume_authorized:
            raise ProtocolError("恢复必须显式 --resume，不接受静默加载旧权重")
        if not self._at_epoch_boundary or self._resume_loaded:
            raise ProtocolError("只能在新 trainer 的 epoch 边界加载一次")
        checkpoint = (read_checkpoint(filename_or_checkpoint, self.protocol)
                      if isinstance(filename_or_checkpoint, (str, Path)) else filename_or_checkpoint)
        validate_checkpoint(checkpoint, self.protocol)
        verify_sources(self.protocol)
        if checkpoint["runtime"] != runtime_signature(self.device):
            raise ProtocolError("软硬件运行签名变化，不声明跨环境位级一致恢复")
        if (checkpoint["grad_scaler_state"] is None) != (self.grad_scaler is None):
            raise ProtocolError("AMP scaler 配置不一致")
        if not self.was_initialized:
            self.initialize()
        # 不调用存在 dict 分支缺陷的 super.load_checkpoint。
        self.network.load_state_dict(checkpoint["network_weights"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
        if self.grad_scaler is not None:
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]
        self.inference_allowed_mirroring_axes = checkpoint["inference_allowed_mirroring_axes"]
        self.current_epoch = checkpoint["next_epoch"]
        self.global_iteration = checkpoint["global_iteration"]
        for key, value in checkpoint["updates"].items():
            setattr(self.updates, key, value)
        restore_rng(checkpoint["rng"], self.drop_rng)
        self._resume_loaded = True
        self.print_to_log_file("V5 resume epoch boundary", self.current_epoch, self.protocol.sha256)

    def perform_actual_validation(self, save_probabilities: bool = False) -> None:
        raise ProtocolError("真实推理、几何恢复和评分由任务19独立入口处理")
