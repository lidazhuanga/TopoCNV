"""V5 专用启动器；import/--help 不导入 Torch、不设置环境、不创建目录。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from .protocol import (ProtocolError, canonical_json, guard_child, load_protocol,
                       run_directory, validate_training_inputs)
from .randomness import initialize_seed_runtime, runtime_environment


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="独立 V5-Utility-v1；固定 training_seed=42，仅 epoch 边界恢复")
    commands = result.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="显式启动训练；只允许已准备好的独立 V5 数据")
    train.add_argument("--protocol", required=True, type=Path, help="冻结协议 envelope JSON")
    train.add_argument("--resume", type=Path, help="同 run 的 V5 epoch checkpoint，禁止旧 R0u")
    train.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="CPU 仅供合成 Verify")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(arguments)
    protocol = load_protocol(args.protocol)
    p = protocol.to_dict()
    if args.device == "cpu" and not p["test_root"]:
        raise ProtocolError("CPU 仅允许显式合成临时根")
    # 正常入口总是 exec 新进程，保证 PYTHONHASHSEED 在解释器启动前生效。
    # PID 标记不能被父进程遗留环境误当作本次已引导；spawn worker 不调用 main。
    marker = f"{os.getpid()}:{protocol.sha256}"
    if os.environ.get("V5_UTILITY_EXEC_MARKER") != marker:
        env = dict(os.environ)
        env.update(runtime_environment(Path(p["root"]), p["test_root"]))
        env["V5_UTILITY_BOOTSTRAPPED"] = p["root"]
        env["V5_UTILITY_EXEC_MARKER"] = marker
        env["MPLCONFIGDIR"] = str(guard_child(Path(p["root"]), run_directory(protocol).parent / ".matplotlib"))
        os.execve(sys.executable, [sys.executable, "-B", "-m", "v5_utility.launch", *arguments], env)
        return 0
    # 在 import nnunet 及其 constructor 的副作用之前完成所有只读预检。
    validate_training_inputs(protocol)
    output = run_directory(protocol)
    if args.resume is None and output.exists():
        raise ProtocolError("V5 run 已存在，拒绝覆盖；恢复必须显式 --resume")
    if args.resume is not None:
        checkpoint_path = guard_child(output, args.resume)
        if checkpoint_path.parent != output or not checkpoint_path.is_file():
            raise ProtocolError("--resume 必须指向当前 V5 run 的既有 epoch checkpoint")
    initialize_seed_runtime(root=Path(p["root"]), fold=p["fold"], test_root=p["test_root"],
                            cuda=args.device == "cuda")
    import torch
    from .trainer import V5Trainer, read_checkpoint
    checkpoint = read_checkpoint(args.resume, protocol) if args.resume is not None else None
    plans = protocol.plans()
    plans["v5_utility"] = {"protocol": protocol.envelope(), "train_authorized": True,
                           "resume": args.resume is not None}
    trainer = V5Trainer(plans, "3d_fullres", protocol.split_index, p["dataset_json"],
                        unpack_dataset=False, device=torch.device(args.device))
    try:
        if checkpoint is not None:
            trainer.load_checkpoint(checkpoint)
        print("V5 launch " + canonical_json(protocol.identity()))
        trainer.run_training()
    finally:
        trainer.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # 不打印未知异常文本/traceback，避免数据读取异常暴露患者文件名。
        message = str(error) if isinstance(error, ProtocolError) else type(error).__name__
        print("V5 启动/训练失败: " + message, file=sys.stderr)
        raise SystemExit(2) from None
