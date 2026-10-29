# results/ 入库说明

生产环境里 `results/` 约 470 GB（预测 NIfTI、checkpoint、缓存），整体被 `.gitignore` 排除。为了复核论文与 V4/V5 决策，只有体积小、不含逐病例行的汇总表用 `git add -f` 强制跟踪；逐病例 CSV 一律不入库，需要时在本地重算。

## 已跟踪

| 路径 | 内容 |
|---|---|
| `p0_binary_y_eval_v2/P0_main_table.md` | P0 六臂 OOF 主表（n = 154，protocol v0.3，L1 scoring v2），含 v0.4 端点 |
| `p0_stats/P0_paper_table.md`、`P0_paper_table.tex` | 论文用 P0 表（bootstrap 95% CI、Holm 校正） |
| `p0_stats/P0_stats_report.md` | P0 统计报告 |
| `p0_stats/descriptives.csv`、`per_fold.csv`、`paired_families.csv` | 按臂/折聚合的描述统计与配对检验（无 case 行） |
| `p0_stats/status.json` | 表状态与版本 |
| `sota_binary_y_eval/SOTA_main_table.md` | SOTA 对照主表（含 nnU-Net v2 5 折 0.6579） |
| `analysis_mech/summary.md` | 机制分析条件汇总（原图 / 去 memory / 擦除 E1、E2） |
| `DEV_main_table.md` | V4 开发阶段按臂汇总（harm_v1 评价，含逐折 Dice_Y） |
| `v4_dev_eval_harm_v1/GT.json` | 开发评测所用 GT 版本与缓存哈希 |
| `v4_select42/fold*/d4.json`、`v4_select43/fold*.json` | 每折选择记录（规则、候选、哈希、选择结果；无 case 行） |

## 未跟踪（需要时本地重算）

- 大体积产物：`*/fold*/checkpoint*.pth`、`*_label_predictions/`、`preprocess_cache/`、`v4_harm_v1/`、`v4_priors/`、`npc_vesselFM3D_*_cv_predictions/` 等。
- 逐病例评分 CSV：`p0_binary_y_eval_v2/*.csv`、`v4_dev_eval_harm_v1/*/fold*.csv`、`analysis_mech/cases/`，含伪匿名 case_id，只在本地保留。

## 重算约定

- 汇总表由 `MIA/scripts/eval_protocol_metrics.py`（指标）+ 各评估/统计驱动脚本生成；口径见 `MIA/scripts/metrics_protocol.py`。
- `DEV_main_table.md` 的聚合口径：Dice_Y / SurfDice_Y 用 mean ± 总体 sd，HD95 用中位数 [IQR]，逐折与整体分别汇总，不跨臂做推断。
