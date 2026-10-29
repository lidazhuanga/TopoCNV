"""独立 seed 子流、进程启动环境及可序列化 RNG 状态；模块导入无副作用。"""
from __future__ import annotations

import copy
import multiprocessing
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .protocol import ProtocolError, ROOT_NAMES, STREAM_SCHEME, guard_root


def runtime_environment(root: Path, test_root: bool = False) -> Dict[str, str]:
    base = guard_root(root, test_root)
    values = {name: str(base / name) for name in ROOT_NAMES}
    values.update({"PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                   "nnUNet_compile": "false", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                   "nnUNet_n_proc_DA": "4", "nnUNet_keep_files_open": "false",
                   "PYTHONDONTWRITEBYTECODE": "1"})
    return values


def require_runtime_environment(root: Path, test_root: bool = False) -> None:
    if any(os.environ.get(k) != v for k, v in runtime_environment(root, test_root).items()):
        raise ProtocolError("必须先在新进程导入 Torch/nnunet 前设置 V5 启动环境")
    if os.environ.get("V5_UTILITY_BOOTSTRAPPED") != str(guard_root(root, test_root)):
        raise ProtocolError("缺少新进程启动标记；使用 V5 显式入口")


def seed_sequence(fold: int, epoch: int, worker: int, purpose: str,
                  sample: int = 0, transform: int = 0) -> Any:
    import numpy as np
    coords = (fold, epoch, worker, sample, transform)
    if any(type(x) is not int or x < 0 for x in coords):
        raise ProtocolError("子流坐标必须是非负整数")
    code = STREAM_SCHEME["purposes"][purpose]
    return np.random.SeedSequence([42, fold, epoch, worker, code, sample, transform])


def seed_cpu(sequence: Any) -> None:
    """仅 CPU RNG；不调用 torch.manual_seed 或任何 CUDA API。"""
    import numpy as np
    import torch
    values = sequence.generate_state(4)
    random.seed(int(values[0]) | (int(values[1]) << 32))
    np.random.seed(int(values[2]))
    torch.random.default_generator.manual_seed(int(values[3]))


def initialize_seed_runtime(*, root: Path, fold: int, test_root: bool = False,
                            cuda: bool = False) -> Dict[str, Any]:
    """新进程环境已经设置后显式调用；确定性 kernel 不支持时直接报错。"""
    require_runtime_environment(root, test_root)
    import torch
    method = multiprocessing.get_start_method(allow_none=True)
    if method is None:
        multiprocessing.set_start_method("spawn")
    elif method != "spawn":
        raise ProtocolError("必须在独立 spawn 进程运行，不能沿用 fork 上下文")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    seq = seed_sequence(fold, 0, 0, "main")
    seed_cpu(seq)
    if cuda:
        if not torch.cuda.is_available():
            raise ProtocolError("请求 CUDA 但设备不可用")
        torch.cuda.manual_seed_all(int(seq.generate_state(1)[0]))
    return {"training_seed": 42, "fold": fold, "start_method": "spawn", "cpu_threads": 1,
            "deterministic": True, "cross_hardware_bitwise_guarantee": False}


def initialize_worker(fold: int, epoch: int, worker: int, purpose: str) -> None:
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    seed_cpu(seed_sequence(fold, epoch, worker, purpose))


def capture_cpu_rng() -> Dict[str, Any]:
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone()}


def restore_cpu_rng(state: Dict[str, Any]) -> None:
    import numpy as np
    import torch
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch_cpu"].cpu())


@contextmanager
def cpu_rng_scope(sequence: Optional[Any] = None) -> Iterator[None]:
    """增强隔离：保存并恢复外层采样流，绝不读写 CUDA RNG。"""
    state = capture_cpu_rng()
    try:
        if sequence is not None:
            seed_cpu(sequence)
        yield
    finally:
        restore_cpu_rng(state)


def capture_rng(drop_generator: Any) -> Dict[str, Any]:
    import torch
    result = capture_cpu_rng()
    result["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    result["F_generator"] = copy.deepcopy(drop_generator.bit_generator.state)
    return result


def restore_rng(state: Dict[str, Any], drop_generator: Any) -> None:
    import torch
    restore_cpu_rng(state)
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ProtocolError("CUDA RNG 设备数不匹配，不能声明等价恢复")
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])
    elif torch.cuda.is_initialized():
        raise ProtocolError("CPU checkpoint 不能作为 CUDA 可复现恢复")
    drop_generator.bit_generator.state = copy.deepcopy(state["F_generator"])
