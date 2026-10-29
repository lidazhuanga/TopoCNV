"""有序 CPU 增强及固定 MMd1；不导入旧自定义 trainer。"""
from __future__ import annotations

import multiprocessing
import threading
from queue import Queue as ThreadQueue
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter, producer, results_loop
from threadpoolctl import threadpool_limits
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D

from .protocol import ProtocolError, RECIPE
from .randomness import cpu_rng_scope, initialize_worker, seed_sequence

INTENSITY = {"GammaTransform", "MultiplicativeBrightnessTransform", "ContrastTransform",
             "GaussianBlurTransform", "SimulateLowResolutionTransform", "GaussianNoiseTransform"}


def draw_drop(rng: np.random.Generator, shape: Sequence[int], p_case: float = 0.2,
              p_trunc: float = 0.3, p_slab: float = 0.2) -> Dict[str, Any]:
    """纯算法移植自 protocol.LEGACY_ALGORITHM 指定的源码哈希。"""
    d: Dict[str, Any] = {"case": bool(rng.random() < p_case)}
    if d["case"]:
        return d
    if rng.random() < p_trunc:
        tilt = np.deg2rad(rng.uniform(0.0, 20.0))
        az = rng.uniform(0.0, 2 * np.pi)
        d["trunc"] = {"z0": float(rng.uniform(0.0, shape[0])), "tilt": float(tilt),
                      "az": float(az), "side": "cranial" if rng.random() < 0.5 else "caudal"}
    if rng.random() < p_slab:
        n = 1 if rng.random() < 0.5 else 2
        slabs = []
        for _ in range(n):
            t = int(rng.integers(4, 11))
            t = min(t, shape[0])
            s = int(rng.integers(0, shape[0] - t + 1))
            slabs.append((s, s + t))
        d["slabs"] = slabs
    return d


def keep_mask(d: Dict[str, Any], shape: Sequence[int], spacing: Sequence[float],
              device: Any = "cpu") -> torch.Tensor:
    """按 mm 计算倾斜平面；轴0语义沿用冻结 plans 和旧纯算法。"""
    keep = torch.ones(tuple(shape), device=device)
    if d.get("case"):
        return keep * 0
    if "trunc" in d:
        t = d["trunc"]
        n0 = np.cos(t["tilt"])
        n1 = np.sin(t["tilt"]) * np.cos(t["az"])
        n2 = np.sin(t["tilt"]) * np.sin(t["az"])
        z = (torch.arange(shape[0], device=device, dtype=torch.float32) - t["z0"]) * spacing[0]
        a = (torch.arange(shape[1], device=device, dtype=torch.float32) - (shape[1] - 1) / 2) * spacing[1]
        b = (torch.arange(shape[2], device=device, dtype=torch.float32) - (shape[2] - 1) / 2) * spacing[2]
        dist = z[:, None, None] * n0 + a[None, :, None] * n1 + b[None, None, :] * n2
        drop = dist > 0 if t["side"] == "cranial" else dist < 0
        keep = keep * (~drop).float()
    for s, e in d.get("slabs", ()):
        keep[s:e] = 0
    return keep


def prepare_modalities(data: torch.Tensor, arm: str, *, training: bool = False,
                       rng: Optional[np.random.Generator] = None, no_mri: bool = False,
                       spacing: Sequence[float] = tuple(RECIPE["spacing"])) -> torch.Tensor:
    """训练/验证/推理共用；输入 B,C,Z,Y,X，MRI 全零实验必须同时清零 valid。"""
    if arm not in ("C", "MM", "F") or data.ndim != 5 or data.shape[1] != (1 if arm == "C" else 3):
        raise ProtocolError("V5 输入维度或通道数量不匹配")
    if arm == "C":
        return data
    result = data.clone()
    result[:, 2] = (result[:, 2] > 0.5).to(result.dtype)
    result[:, 1] *= result[:, 2]
    if no_mri:
        result[:, 1:] = 0
    elif training and arm == "F":
        if rng is None:
            raise ProtocolError("F 训练必须提供独立 Generator")
        for i in range(result.shape[0]):
            d = draw_drop(rng, result.shape[2:])
            mask = keep_mask(d, result.shape[2:], spacing, result.device).to(result.dtype)
            result[i, 1:] *= mask
    return result


class FirstTwoChannels:
    def __init__(self, transform: Any) -> None:
        self.transform = transform

    def __call__(self, **data: Any) -> Dict[str, Any]:
        image = data["image"]
        data["image"] = image[:2]
        result = self.transform(**data)
        result["image"] = torch.cat((result["image"], image[2:]), dim=0)
        return result


class SeededTransforms:
    """逐样本/逐变换子流；C 补两个零通道以匹配 MM/F 的随机消费。"""
    def __init__(self, pipeline: Any, fold: int, epoch: int, training: bool) -> None:
        self.fold, self.epoch, self.training = fold, epoch, training
        self.worker = 0
        self.sample = 0
        self.transforms = []
        count = 0
        for transform in pipeline.transforms:
            inner = getattr(transform, "transform", transform)
            intensity = type(inner).__name__ in INTENSITY
            if type(inner).__name__ == "GaussianBlurTransform":
                inner.benchmark = False
            if intensity:
                transform = FirstTwoChannels(transform)
                count += 1
            purpose = ("train_intensity" if intensity else "train_spatial") if training else "val_transform"
            self.transforms.append((transform, purpose))
        if training and count != 7:
            raise ProtocolError("上游增强结构变化，必须重新审核强度通道适配")

    def set_worker(self, worker: int) -> None:
        self.worker, self.sample = worker, 0

    def __call__(self, **data: Any) -> Dict[str, Any]:
        image = data["image"]
        channels = image.shape[0]
        if channels not in (1, 3) or image.device.type != "cpu":
            raise ProtocolError("增强 worker 只接受单/三通道 CPU 张量")
        if channels == 1:
            data["image"] = torch.cat((image, torch.zeros_like(image), torch.zeros_like(image)), 0)
        for index, (transform, purpose) in enumerate(self.transforms):
            sequence = seed_sequence(self.fold, self.epoch, self.worker, purpose, self.sample, index)
            with cpu_rng_scope(sequence):
                data = transform(**data)
        self.sample += 1
        if channels == 1:
            data["image"] = data["image"][:1]
        else:
            data["image"] = prepare_modalities(data["image"][None], "MM")[0]
        return data


class SeededDataLoader3D(nnUNetDataLoader3D):
    def __init__(self, *args: Any, stream_fold: int, stream_epoch: int,
                 stream_training: bool, **kwargs: Any) -> None:
        self.stream_fold = stream_fold
        self.stream_epoch = stream_epoch
        self.stream_training = stream_training
        super().__init__(*args, **kwargs)
        self.indices = sorted(self.indices)
        self.list_of_keys = list(self.indices)
        # 上游 infinite=True 不消费 rs，仍禁止保留 seed=None 的随机源。
        self.rs = np.random.RandomState(42)

    def set_thread_id(self, thread_id: int) -> None:
        super().set_thread_id(thread_id)
        purpose = "train_sampling" if self.stream_training else "val_sampling"
        initialize_worker(self.stream_fold, self.stream_epoch, thread_id, purpose)
        self.rs = np.random.RandomState(int(seed_sequence(
            self.stream_fold, self.stream_epoch, thread_id, purpose).generate_state(1)[0]))
        if self.transforms is not None:
            self.transforms.set_worker(thread_id)

    def generate_train_batch(self) -> Dict[str, Any]:
        try:
            return super().generate_train_batch()
        except Exception:
            # 上游 producer 会直接打印异常；不要让患者文件名出现在控制台。
            raise RuntimeError("V5 CPU worker 数据/确定性增强失败；请独立核对输入与 kernel") from None


class OrderedAugmenter(MultiThreadedAugmenter):
    """保留上游轮询队列顺序；显式 spawn，CPU 模式不触碰 CUDA。"""
    def _start(self) -> None:
        if self.was_initialized:
            return
        self._finish()
        self.abort_event.clear()
        self._queue_ctr = self._end_ctr = 0
        if hasattr(self.generator, "was_initialized"):
            self.generator.was_initialized = False
        context = multiprocessing.get_context("spawn")
        with threadpool_limits(limits=1, user_api="blas"):
            for i in range(self.num_processes):
                self._queues.append(context.Queue(self.num_cached_per_queue))
                process = context.Process(target=producer, args=(
                    self._queues[i], self.generator, self.transform, i, self.seeds[i], self.abort_event))
                process.daemon = True
                self._processes.append(process)
                process.start()
        # stock _start 即使 CPU 模式也查询 CUDA；这里只在主进程要求 pin 时查询。
        gpu = torch.cuda.current_device() if self.pin_memory else None
        self.pin_memory_queue = ThreadQueue(max(3, self.num_cached_per_queue * self.num_processes // 2))
        self.pin_memory_thread = threading.Thread(target=results_loop, args=(
            self._queues, self.pin_memory_queue, self.abort_event, self.pin_memory, gpu,
            self.wait_time, self._processes), daemon=True)
        self.pin_memory_thread.start()
        self.was_initialized = True

    def _finish(self, timeout: float = 10) -> None:
        processes = list(getattr(self, "_processes", ()))
        super()._finish(timeout)
        for process in processes:
            process.join(timeout)
            if process.is_alive():
                raise RuntimeError("V5 epoch worker 未能停止，拒绝带旧队列继续")


def ordered_augmenter(loader: SeededDataLoader3D, workers: int, *, pin_memory: bool = False) -> OrderedAugmenter:
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        raise ProtocolError("有序增强必须使用 spawn")
    purpose = "train_sampling" if loader.stream_training else "val_sampling"
    seeds = [int(seed_sequence(loader.stream_fold, loader.stream_epoch, i, purpose).generate_state(1)[0])
             for i in range(workers)]
    return OrderedAugmenter(data_loader=loader, transform=None, num_processes=workers,
                            num_cached_per_queue=2, seeds=seeds, pin_memory=pin_memory,
                            timeout=10, wait_time=0.02)
