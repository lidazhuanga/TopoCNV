#!/usr/bin/env python3
"""v4 Stage-1 训练入口兼容导出，仍支持旧模块 CLI。"""

from topocnv.scripts.v4_train_stage1 import PATCH, main

__all__ = ["PATCH", "main"]

if __name__ == "__main__":
    main()
