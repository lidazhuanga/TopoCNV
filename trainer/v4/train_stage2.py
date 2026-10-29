#!/usr/bin/env python3
"""v4 Stage-2 训练与报告入口兼容导出，仍支持旧模块 CLI。"""

from topocnv.scripts.v4_train_stage2 import main, score_cases

__all__ = ["main", "score_cases"]

if __name__ == "__main__":
    main()
