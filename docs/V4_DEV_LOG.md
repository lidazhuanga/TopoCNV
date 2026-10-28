# V4 开发日志

规则（协议 v0.4-draft §5）：开发阶段只用 fold 0、fold 1 的内部验证集 V_0、V_1 做选择；所有试过的配置都记在这里，没选上的也记。判据见 `docs/V4_PLAN.md` §5：V_0 ∪ V_1 合并后的平均 Dice_Y，与最好的候选差 < 0.003 时取更简单的那个；HD95_Y 只作参考。

## 报告折访问记录

开发阶段唯一一次看报告折，是 R0 的复现核对（主会话 2026-10-22 同意）：

- 看什么：R0 在报告折 fold 0 上的 Dice_Y、HD95_Y，和 cnv-sota 的 DynUNet fold 0（0.6423 / 12.6 mm）比。
- 为什么不影响选择：R0 是固定基线，不参与 D1–D4 的任何一项选择。R0 的配方在看之前就定了（`configs/v4_dev/r0.yaml`），看完也不改。
- 什么时候看的：见下面「R0 复现核对」一节，以及 `log/v4/report_access.log` 里的时间戳。

其他臂冻结前一律不看报告折。

## 开发前的准备

| 日期 | 事项 | 结果 |
|---|---|---|
| 2026-10-22 | GT 路标框的 ROI 覆盖率上限（154 例，只读） | 只用类别 4：m = 35 均值 99.75%，最低 96.3%；类别 1∪4：m = 25 均值 99.97%，m = 35 为 100%。Stage 1 因此学类别 1 和 4 |
| 2026-10-22 | 单元测试 `tests/test_v4_*.py` | 16 项全部通过 |
| 2026-10-23 | 端到端冒烟 `tests/v4_smoke.sh`（GPU 4，2 epoch） | Stage 1 → 先验 → R0、Full-v4 → OOF 标签都跑通，`eval_protocol_metrics.py --pred_binary_Y` 能读。数值无意义（未训练） |
| 2026-10-23 | 骨架预计算（154 例） | 管状骨架占 Y 体素的 72%（最低 53%）。Y 本身只有 1–3 体素粗，所以骨架召回率很接近体素召回率 |
| 2026-10-23 | 单个 run 显存 | Stage 1、R0 训练中约 6.5 GB |
| 2026-10-23 | **ROI 清理参数：连接距离 30 → 45 mm** | 用 GT 路标代替 Stage 1 检查 V_0 的 8 例，m = 25 的覆盖率只有 65–100%，原因是清理时把真实路标的连通域当成远处假阳性去掉了。在 T_0 − V_0 的 105 例训练病例上量 GT 路标（类别 1∪4）：每侧连通域从最大块单连接生长所需的最大间距，中位数 31.7 mm，P90 34.7 mm，P99 39.5 mm，最大 54.6 mm；73/105 例超过 30 mm。改为 45 mm（只依据训练病例的 GT 几何，没有看任何 Stage 2 结果）。改后这 8 例 m = 25 的覆盖率为 99.4–100%。commit bceed15 |
| 2026-10-23 | 先验 (c) 的合理性（GT 路标对齐，V_0 的 8 例） | ICP RMSE 0.9–1.9 mm，缩放 0.89–1.06（没有碰到 0.8/1.25 的边界）。图谱单独当分割用，最佳阈值下 Dice 0.41–0.56；Y 内平均 0.29–0.36，Y 外（先验框内）0.003–0.005 |
| 2026-10-23 | 头部中线与类别 4 中点的偏差（T_0 − V_0，全体积算中线） | 平均 1.7 体素，SD 4.0，最大 27.5 体素。实际流程只在路标所在的 z 段上算中线，待 Stage 1 出来后在训练病例上复查分侧是否出错 |

| 2026-10-23 | 主线改为 nnU-Net v2（主会话通知，协议 §3 修订；用户同日确认"baseline 用 nnU-Net v2，在此基础上改进"） | DynUNet 版 R0 fold 0 继续跑完，留作附录骨干对照；DynUNet 版 R0 fold 1 已从队列撤掉 |
| 2026-10-23 | fold 0 先验构建（`tools/v4_build_priors.py`） | 154 例全部有 ROI，没有一侧需要镜像；模板 2201540520，图谱 123 例，用时 4 分钟 |
| 2026-10-23 | **真实 Stage 1 下的 ROI 覆盖率（协议原定义：路标外接框 + m）** | V_0（18 例）：m = 15/25/35 均值 67.9% / 79.7% / 87.6%，最低 46.5% / 64.4% / 78.4%；T_0 − V_0（105 例）：66.1% / 78.2% / 86.0%。达不到 99% |
| 2026-10-23 | 原因：Stage 1 偏弱（T_0 − V_0 内层 OOF） | 类别 1 Dice 0.35，类别 4 Dice 0.47；预测路标体积约为 GT 的 60%；清理步骤几乎没去掉东西（保留比例中位数 1.00）。预测框在 z 低端比 GT 框短中位数 55 mm（P90 67 mm），其他方向短 1–17 mm |
| 2026-10-23 | 候选方案 B：图谱定义的 ROI（开发选项，只在 V_0 和训练病例上算） | 预测类别 4 点做 ICP，把留一图谱的支撑区（阈值 0/0.02/0.05）映到当前病例，每侧外接框 + m。阈值 0.02：V_0 m = 15/25/35 覆盖率 99.96% / 100% / 100%（最低 99.3% / 100% / 100%），ROI 体积 3.3% / 4.7% / 6.3%；T_0 − V_0：99.65% / 99.98% / 100%（最低 93.6% / 99.5% / 100%）。阈值 0：m = 25 体积 6.8%；阈值 0.05：m = 25 的 T_0 − V_0 最低 98.5%。是否采用待主会话确认（改的是协议 §4 的 ROI 定义） |

| 2026-10-23 | 主会话确认三件事 | (1) 选模型用最后一个 epoch：开发阶段在 T_k − V_k 上训练、在 V_k 上比较，报告阶段在 T_k 上重训，R0 直接用 cnv-sota 的 nnU-Net v2 5 折；开发阶段的参照是同样在 T_k − V_k 上训练的原版 nnU-Net v2（R0-dev），不和 5 折报告数比。(2) ROI 改为图谱支撑区（阈值 0.02）经 ICP 映射 + m；图谱和模板只能由当前训练集构建（开发 T_k − V_k，报告 T_k）；报告阶段覆盖率 < 99% 的病例要列出。(3) Stage 1 保持 DynUNet |
| 2026-10-23 | 先验按两套重建：`fold{k}_dev`（图谱 = T_k − V_k，105 例）、`fold{k}_report`（图谱 = T_k，123 例），k = 0、1 | 4 套都是 154 例全部有 ROI，没有一侧需要镜像。模板：fold0_dev 03956794，fold0_report 2201540520，fold1_dev 2300312143，fold1_report 02960410 |
| 2026-10-23 | **D1：开发图谱下的 ROI 覆盖率，V_0 ∪ V_1（36 例）** | m = 15 / 25 / 35：均值 99.48% / 99.84% / 99.93%，最低 82.6% / 94.2% / 97.6%，≥ 99% 的病例占 94.4% / 97.2% / 97.2%，ROI 体积 3.8% / 5.3% / 7.0%。低于 99% 的：2500258351（82.6% / 94.2% / 97.6%）、03765432（m = 15 为 98.6%）。训练病例（T_k − V_k，210 例，留一）：99.73% / 99.97% / 100%，最低 93.7% / 97.9% / 99.6% |
| 2026-10-23 | 2500258351 覆盖率低的原因 | Stage 1 只找到一小块类别 4（每侧约 660 体素，左右方向只有 15 体素宽），两侧 ICP 缩放都碰到上界 1.25，映过来的图谱缩小了 20%，前后方向漏掉 Y 的前端。正在训练病例上比较 ICP 缩放范围（0.8–1.25 / 0.9–1.1 / 刚性），不在 V_k 上调 |

## D0 基线（R0）

## D1 ROI 外扩 m 与 ROI 清理参数

## D2 先验组合

## D3 类别 5 在辅助头里的处理

## D4 骨架损失权重

## 冻结

| 2026-10-24 | kimi k3.1-pro 只读审查 V4_PLAN（报告：~/cnv/v4_kimi_review_final_20261024.md） | 3 个 P0（同一套五折既调参又报告；开发期 Stage 1 未嵌套且 V_k 推理口径与报告不一致；nnU-Net 路径无真实冒烟）+ 6 个 P1。用户决定采用 nested CV 补救 |
| 2026-10-24 | nested CV 落地（V4_PLAN §5–§9 修订） | splits.py 加 tier（开发层内层折只分 T_k−V_k）；train_stage1.py 加 --tier（dev 层预测 V_k 到 val_pred/）；build_priors.py dev 变体改读 fold{k}_dev（V_k 三模型平均）；新增 tools/v4_select_config.py（每折选择 + 完整性硬校验 + configs/v4_frozen/fold{k}.json）；train --variant report 按折门控。tests：+4（test_v4_splits_devtier），20+6 全过。修复 tools/v4_nnunet.py 一处 f-string 语法错误（prep 曾因此从未运行） |
| 2026-10-24 | 旧开发产物作废标记 | 2026-10-23 构建的 fold0/fold1 _dev 先验基于报告层 Stage 1（含 V_k），按新设计作废待重建；r0dev fold0/f1 结果（V_0 Dice 0.634、V_1 Dice 0.674）仍有效（R0 无级联不依赖 Stage 1），作为各折参照 |
| 2026-10-24 | 用户决策落地 | (1) D1 m 规则改下尾：训练病例最低覆盖率 ≥99%；报告折只描述。(2) 空预测 HD95 = 病例物理对角线惩罚计入汇总，逐例 Empty_pred_Y + 空预测率（metrics_protocol._hd95 / score_protocol、eval_protocol_metrics 已改）。(3) 统计预注册：主比较 Full-v4 vs R0 配对 ΔDice/ΔHD95 + bootstrap CI，其余 Holm；不做重复 seed。(4) 强度增强只作用 CT 通道 0（nnUNetTrainerV4.Channel0Only，含 Gamma 反转在内的 7 类 stock 强度变换被包装）。(5) 冒烟按一般要求，不做吞吐实测 |
| 2026-10-24 | sota_binary_y_nnunet fold0 修复重跑完成 | 原版配方（300ep + sliding-window 选模 + crop0.33）：fold0 Dice_Y 0.6475 / HD95 12.79mm（n=31），与 10-22 原版 0.6423/12.64 一致；10-24 上午因丢 flag 产生的垃圾结果及覆盖事故已恢复 |
| 2026-10-24 | 主干曝光的处理方向（用户问询后确认不重跑） | 重跑同批 154 例无法消除“设计用过全量五折”的信息曝光；方向定为锁定确认集：2026-10-24 新到的 126 例（120 例 CT 未标注已隔离 + MR）标注到位后作锁定测试集，默认不并入训练（V4_PLAN §7.1） |
| 2026-10-24 | P0 §17.1 七臂 fold0 重跑确认放弃 | 协议 v0.4：P0 仅作历史记录不进新稿主表、GPU 优先级低于 v4；P0 5/5 表已完整。已完成 3 臂中 full/b0（500ep，fold0，4/7 臂缺失）标记 ABANDONED，a_mem/a_gate/c1/a2 取消不排队；nnunet 臂坏结果已修复（fold0 0.6475/12.79 与 10-22 序列一致）。详见 docs/P0_RUNBOOK.md |
| 2026-10-24 | 复审（qoder DeepSeek-Flash，~/cnv/qoder_review_20261024.log）修复 | P0-1 Channel0Only 首版键/轴错误（生产键是 image、无 batch 维、通道轴 0）→ 生产中为空操作，已按生产约定重写并加真实管线集成测试；P0-2 dev 先验/prep 病例集合改为 T_k（原 154 会在报告折病例上崩溃）；P1 报告阶段门控改为实际校验 chosen_config 的 margin/trainers、移除旧 yaml 旁路；sota/p0 汇总 REQUIRED 加 Empty_pred_Y（旧 CSV 自动触发重评）；Empty_pred_Y 仅在 GT 非空时计 1。受影响需作废的 Stage-2 run：无（v4 候选尚未开跑；r0dev 用 stock trainer 不受 DA 改动影响） |
| 2026-10-25 | 数据范围定界（用户确认） | 绿皮只用医生标注/修改的 154 例（稿件口径：178 筛查→157 有 MRI→154 标注 + 3 例剂量学）；约 250 例存量与 10-24 新到 120 例无标签一律不用；§7.1 锁定确认集方向取消，主干曝光按局限声明处理。120 例 CT 维持隔离，其 MR 不在 selection 清单、不参与配准 |
| 2026-10-25 | R1+P+V 消融链采纳（用户确认） | 开发期每折 +1 个 R1+P+V run（先验组合/类别 5 处理随本折 D2/D3 的选择），共 55 个开发 run（新启动 53）；预算 ~350 GPU 小时。同日：4 个 0 字节 MR（00894361/04504013/2201024214/2300669554）从 ztest 的原始 DICOM（export_args 记录的 series UID）重导并覆盖，配准循环自动接手 |
| 2026-10-25 | 真实 nnU-Net 路径冒烟通过（SMOKE_OK） | 全链路：dev 层 Stage1(3例x2ep) → 4例先验 → Dataset520 小集(fingerprint 修复后正常生成) → SMOKE_Pabc_Vign_S10 2ep(先验+可见度+骨架全开) → 导出(ROI 贴回+_vis) → 评价器读回(Empty_pred_Y 列在)。冒烟共修 4 个从未暴露的集成 bug：(1) prep 缺 dataset_fingerprint(首次训练必崩)；(2) smoke 包装器 *args 签名破坏 nnUNetTrainer 调用帧内省(KeyError a)；(3) label_manager 6 头 vs 网络 7 通道 → _NOutLabelManager 代理；(4) predictor.manual_initialization 重建 6 头 label_manager → 训练后代理传递。qoder DeepSeek-Flash 两轮复审(报告 ~/cnv/qoder_review2_20261025.log)另修 5 P0(fingerprint/标记污染/陈旧先验作废改名/评价器参数/MEAN 行计数)+4 P1 |
| 2026-10-25 | v4 队列启动（15:24） | s1dev x15(5折x3内层)、r0dev f2-4、priors dev/report 各5、s1 report f0/f1(标记即完)。前 4 个 s1dev 上 GPU 4/1/7/6；priors_report_f0/f1 并行重建(带 stage1_dir 溯源)。jobs.json/queue_state 已归档重建 |
| 2026-10-25 | D1 覆盖率实测（新开发层先验，训练病例留一） | f1=25(min 0.991)、f2=35(0.990)、f3=15(1.000) 按原规则可选；f0 无 m（唯一例 2200498367 双侧 ICP 收敛但图谱框完全错过 Y，cov=0，RMSE 3.5mm 正常；该例在 f2/f4 图谱下 cov=1.0）；f4 无 m（06019086=0.977、2300115954=0.988、2500258351=0.988 @m35）。逐例 CSV: results/v4_roi_coverage/dev_train_folds*.csv |
| 2026-10-25 | D1 修订咨询 kimi k3.1-pro（报告 ~/cnv/kimi_d1_answer_20261025.log，会话 3b2fb650） | 建议 D1-AM1：各折 m 取 f0=35/f1=25/f2=35/f3=15/f4=35（不可行折取上限，规则闭合而非降阈值）；全折统一 GT 无关安全门=双侧清洁c4质心左右间距 D，按本折 105 训练例 Q1/Q3±3IQR 冻结（f0 [39.759,59.782]...触发例仅 2200498367，D=347mm vs 其余最大 58.3mm）；触发→全体积+abc先验置零+导出不清零；f4 三例不特判、如实披露 97.67%。修订属训练失败促成的预注册闭合，冻结后不回调。**待用户确认后实施**（m 值本身按原规则，无争议；安全门为新增） |
| 2026-10-25 | D0 基线（R0-dev）五折全部完成 | V_0=0.6344、V_1=0.6735、V_2=0.6613（20:49）、V_3=0.6650（21:29）、V_4=0.6490（21:11）；均值 0.6566，与全量数据 R0 五折 OOF 0.6579 一致（少 15% 训练数据的代价可忽略）。各折候选 run 的参照就绪。队列：31 完成，priors_report_f2-4 等报告层（正常） |
| 2026-10-25 | D1-AM1 实施启动（用户确认） | 代码：priors.bilateral_c4_distance_mm（真实质心 D）；tools/v4_qc_d1.py（每折 m 规则+阈值冻结 → configs/v4_qc/fold{k}.json）；build_priors --qc（结构检查+D 越界→全体积回退，boxes 空/先验置零/记录 qc 块）；v4_jobs priors 挂 --qc；报告门控加 qc 文件校验。待 qoder 复审后执行 |
| 2026-10-25 | D1-AM1 执行完成 | 阈值冻结=QC 工具输出与 kimi 决议逐值一致（f0 [39.759,59.782]...触发仅 2200498367 D=347.27mm）；7 套先验带 --qc 重建（旧目录归档 _PRE_AM1_20261025）；回退验证：2200498367 boxes 全空+先验全零+reason=D_out_of_bounds，其余 5 折 dev 0 触发、report f0/f1 0 触发。中途修复自身 bug：target_transforms 返回 (mid, list) 未解包导致首轮全量误回退 |
| 2026-10-25 | qoder 第三轮复审修复 | P0：队列 done 契约（归档+重加 jobs 重建路径）、候选绑定旧先验的顺序风险（prep 断言 AM1 溯源+margin）、结构检查改用 ICP how==side（bilateral 分支）；P1：QC 配置 write-once+sha256+逐例D、nnprep margin 默认取冻结值、折绑定校验；P2×5 |
| 2026-10-25 | 并发 unpack 竞态事故与修复 | 22:25 两个训练（r1+r1p_abc 同折）同时首解包 Dataset520/521 互相截断 .npy（EOFError），队列自动重试的修复循环救回（全量校验 0 损坏）；修复=prep 末尾预解包 + train 启动前按数据集 flock 串行化验证性解包；Dataset524 手动预解包 248 npy 验证通过 |
| 2026-10-25 | 候选第一波上线（22:23） | nnprep×5（每折 m_k 从冻结 qc 配置自动取）+ R1×5 + D2 组合×20（Pabc/Pbc/Pac/Pab）共 25 个 GPU run；首批 4 个 r1 已在训练（22:30 起），预计 ~2h/波，全部 25 个约 24-28h |
| 2026-10-25 | 队列提速（用户要求利用空闲资源） | 每卡 1 个放宽为每卡 2 个（--max_per_gpu，按卡显存记账守住 ≥40GB 红线），总并发 4→8；报告层 Stage 1 fold2-4（9 个）提前入队（不依赖开发结果，解锁 priors_report_f2-4）。当前 8 个并行：r1×5 + r1p_abc×3；波次从 ~7 压到 ~4，第一波+报告层 Stage 1 预计明早完成 |
| 2026-10-26 | V4-AM-C3W-01 实施启动（用户确认仅评价端） | metrics_protocol 新增 _wdice/_recall_gt3；score_protocol 输出 wDSC_Y_{0.25,0.50,0.75}+Recall_GT3；eval/summary 联动 REQUIRED 触发旧表重算；测试 13 项全过（λ=1 退化、类3漏检折减、FP 永不折减、完美预测=1、空例约定）。训练/选择零改动 |
| 2026-10-26 | 双队列事故与修复（11:30） | 昨晚切换队列时 kill 了 bash 包装而非 python 主进程，新旧两队列并行 6 小时：abc_f3/f4 被双启、r1_f4 完成后残留野实例、state 被互踩（abc_f0 被误标 failed 实际在 Epoch 275）。处置：杀旧队列+去重（保留各组较老实例）+ state 修复为 running_orphan（真实 pid/gpu）+ 单队列重启（同并发参数）。完成作业的有效性：r1_f1-f4/abc_f1/f2 在冲突期内单向运行无重复，结果有效；abc_f3/f4 的存活实例自启动起独立训练（checkpoint_latest 交错但各自内存态完整，最终产物以存活者为准） | 

## D2 候选首批统一重评（2026-10-26 15:00）

- 已完成 14 折（r0_dev×5、r1×5、r1p_abc f0-2、r1p_bc f0）用现行 eval_protocol_metrics.py 统一重评，补齐 wDSC_Y_{0.25,0.50,0.75} 与 Recall_GT3 列（此前 6 个 CSV 为旧列版本，数值不变）。聚合脚本 tools/../agg_dev_results.py。
- 每折 V_k 18 例（fold4 19 例），eval.csv 末行 MEAN 为汇总行非病例。

| run | 已评折 | Dice_Y 均值 | HD95_Y 均值 | wDSC_Y_0.50 | Recall_GT3 | 空预测 |
|---|---|---|---|---|---|---|
| r0_dev | 5 | 0.6566 | 5.28 | 0.6394 | 0.6819 | 0 |
| r1 | 5 | 0.6411 | 5.09 | 0.6239 | 0.6595 | 0 |
| r1p_abc | 3 | 0.6359 | 4.34 | 0.6189 | 0.6480 | 0 |

- 级联代价（R1 − R0-dev，逐例配对）：五折 −0.94/−1.44/−1.45/−1.86/−2.07 pt，R1 逐例胜率仅 1–5/19；HD95 五折中三折改善（均值 5.09 vs 5.28）。
- abc 先验（vs R1，匹配折 f0-2 逐例配对）：Dice −1.19/−0.74/−0.40 pt，胜率 5–7/19；HD95 f0 改善 −0.45 mm，f1/f2 持平略差。先验暂未见收益，待 bc/ac/ab。
- wDSC_Y_0.50 与 Recall_GT3 排序与 Dice 一致（级联与先验均降低 class3 类覆盖）。
- 最差病例（r1 Dice 0.27–0.44）在 r0_dev 上同样最差（0.26–0.46），属病例本身困难，非级联特有失败。
- 注意：dev 层候选训练于 T_k−V_k（75% 数据），级联两端同时缺数据，代价或高于报告层（全 T_k）；报告层复跑将给出最终口径。
- 运行态：abc_f3/f4、bc_f1 Ep299（收尾）；bc_f2 251、bc_f3 151、bc_f4 109、ac_f0 110、ac_f1 101；ab×5 + ac f2-4 排队。tools/eval_watcher.sh 已挂（EXPORTED→自动评，25 折评完自退）。
- C3W 历史臂重算进行中：sota_summary 第~4 臂、p0_summary 第~3 臂（子进程单核逐例 30–60 min/臂，预计今晚—明晨完成）。

## V4.1 重大转向（2026-10-27 12:40-13:30）

- **PI 三项指令**：Stage 1 改 nnU-Net v2 主干；标签本体修正（1=CT可见/2=MRI引导/3=可能走行/4=路标，仅 5(与0)为背景——class 2 作负例是本体错误）；Stage 2 只保留二值 Y 输出。相关实验重跑。
- **止损**：queue STOP 文件生效；8 个 noP run 杀掉并标 cancelled；ab_f4 保留收尾（完成后旧设计 25/25 完整存档作 pre-redesign 证据）；R0-dev 5 折可复用。
- **kimi k3.1-pro 重审计完成**（`~/cnv/kimi_audit_v41_answer_20261027.log`，会话 a95446ef）：签发 V4-AM-20261027-01；Stage1=nnU-Net 5类（bg{0,5}+1/2/3/4）、ROI 保留图谱-ICP、先验 a=p_Y/b/c4 距离/c 图谱、D1-AM2 重冻、Stage2 二值+删 V 删 D3、D4'∈{0,0.5,1.0}、链 R0→R1'→R1'+P*→Full-v4'、预算 750 GPU 时（90 run，5-8 天）。
- **§10 V4.1 修正案已并入 V4_PLAN.md（状态：待 PI 确认）**；开放问题 Q1（class3 训练权重 1.0？）/Q2（不补 2×2 交互？）待拍板。
- 审计亦确认：C3W 完全不受影响；Dice_tier1-4 随 V 头删除而消失（其余评价不变）；旧 25 run 标 V4.0 exploratory。
- 后续门：新链冒烟 → fold0 Stage1 质量报告（含与旧 Stage1 配对比较）→ qoder 审查新代码 → 放开生产队列。

- **PI 确认 V4.1（2026-10-27）**：Q1=class3 训练权重 1.0（等权）；Q2=不补开发期 2×2 交互。§10 状态改为已确认。下一步：新链冒烟 → fold0 Stage1 质量报告 → 新代码 qoder 审查 → 放开生产队列。

## V4.2 草案（2026-10-27 晚）

以下分析都是 0 GPU、只读，没有接触报告折。

| 日期 | 事项 | 结果 |
|---|---|---|
| 2026-10-27 | **qoder（DeepSeek-Flash，max）只读数据分析**。报告：`~/cnv/qoder_analysis_20261027/REPORT.md`；在 qoder 服务端排队约 25 分钟 | • MRI–CT：115 例已配准，pass 58 / review 48 / fail 9，平面内残余中位 1.13 mm。<br>• label–CT：完全对齐（16 例独立重读，Dice 1.000）。<br>• 裁剪与 patch：padding 中位 22.6%，但与 Dice 无关，**plans 失配的假设不成立**（本会话先前提出，已撤回）。<br>• ROI 外 FP 占 0.04% |
| 2026-10-27 | **复核：r0_dev / r1 的 FP 和漏检按 ROI 内外拆分**（91 例） | • r0 的 ROI 外 FP 共 109 体素，都在 1 例里；ROI 外清零后 Dice −0.0002。<br>• r1 的损失来自召回：TP 少 19,349 体素，类 1–4 召回降 2.3–3.3 个点；FP 反而少 8,218。<br>• r0 的 FP 有 64% 在 GT 2.3 mm 内，94% 在 5 mm 内 |
| 2026-10-27 | **标注厚度漂移** | • 45 例厚度中位 6.43，其余 11.37；骨架长度相同（825–873）。<br>• 薄标注集中在 CT 日期 2024-12 至 2025-09（与厚度规则一致率 95.5%），两段时间的 CT 扫描协议相同。<br>• r0_dev：薄标注 0.515（P/Y 1.5–1.8），其余 0.716。<br>• Dice 与厚度 ρ = 0.80，P/Y 与厚度 ρ = −0.82。<br>• 按时期对 r0 输出做平面内腐蚀 / 膨胀：−0.015 / −0.051，粗形态学不可行，**未采用** |
| 2026-10-27 | **薄标注的形态** | • z 范围相同（50 vs 49 层），类别比例相同。<br>• 平面内行程 LR 2.77 vs 4.12、AP 4.15 vs 5.65；每层面积是厚标注的 0.55 倍。<br>• 伪标签批次（P001–P050 及 14 例）不在 154 例内，与此无关 |
| 2026-10-27 | **用户决定 V4.2 方向** | • 放弃级联。<br>• 重采样统一厚度。<br>• Skeleton Recall 并入基线。<br>• 主方法 = 图谱-ICP（改作先验配准引擎）+ 先验 a/b/c + 打乱先验对照。<br>• D2′ 候选含 B。<br>已写入 V4_PLAN §11（草案） |
| 2026-10-27 | **harm_v1 冻结与构建**（`tools/v4_harmonize_labels.py`） | • t = 0.19（网格 0.10–0.30：0.18 → 12.77，0.19 → 11.59，0.20 → 10.77；目标 11.37）。<br>• 45 例共新增 161,320 体素。<br>• QA 全部通过，详见 V4_PLAN §11.3。<br>• 单元测试 `tests/test_v4_harmonize.py`：nnU-Net venv 4 项通过；topocnv 环境 3 项通过，1 项需要 skimage，跳过 |
| 2026-10-27 | **冻结后核对**（r0_dev，未重训，V_k） | • 薄标注 24 例：Dice 0.517 → 0.587，P/Y 1.52 → 0.90。<br>• 厚标注 67 例不变（0.707）。<br>• 全部 91 例：0.6565 → 0.6752 |
| 2026-10-27 | **第 0 步：V4.2 全图数据集**（`tools/v4_nnunet.py prep42`，CPU，约 10 分钟） | • Dataset540 = 缓存 CT + 二值 Y（R0u、B、D4′ 用）；Dataset541 = 同一 CT（原始图像软链接到 540）+ 类别 0–4（Stage 1 用，两层内层折）。标签都来自 harm_v1，构建前核对 manifest 的 sha256。<br>• 逐例核对 154 例：540 的预处理 CT 与 Dataset501 逐字节相同，裁剪框一致；厚标注 109 例的标签与 501 完全相同；薄标注 45 例只多出 manifest 记录的新增体素，合计 161,320；541 的前景与 540 的 Y 相同，类别与 harm_v1 相同。<br>• 两个数据集的 plans 与 Dataset501 只差 dataset_name；splits_final 与 `v42_splits` 一致（540 共 10 条，541 共 30 条）；fingerprint 和预解包都已生成。<br>• 导出链路（stock 路径）：用 `export42` 从 Dataset519 的验证输出重导 r0_dev 5 折共 91 例，与现有导出逐体素相同。V4 训练器路径要等 GPU 冒烟验证。<br>• 未用 GPU，生产队列仍关闭 |
| 2026-10-27 | **qoder 代码审查第 6 轮**（DeepSeek-Flash，max，只读，约 18 分钟；`~/cnv/qoder_review6_20261027.log`，脚本和输出在 `~/cnv/qoder_review6_20261027/`）。审查对象是 8954da3 | • 没有代码级 P0。流程上的 P0：harm_v1 的方向还没确认，确认前不能开始任何 V4.2 训练（包括 D4′）。确认时另写记录文件，不改 harm_v1.json，因为它的 sha256 已写进 manifest 和两个数据集。<br>• 独立核验：3 例薄标注从原始缓存重算，与 harm_v1 逐体素一致；其中 2 例用不裁剪的全体积实现复算，也一致；154 例标签哈希、图像链接、9 项不变量全部通过。最近类别按物理间距和按体素算，会让 66/3835、11/626 个新增体素的类别不同，所以按物理间距是必要的。<br>• B 与 R0u 只差 Skeleton Recall：epoch、采样、增强、深监督权重、滑窗参数都相同。<br>• P1：train42 不查 fold/inner 越界，越界时 nnU-Net 会退回随机 80:20 划分，报告折也会进训练；harm_cache 只查配置哈希，不查标签内容；Stage 1 的完成标记不检查 npz；队列前部还有 8 个 V4.0 遗留作业（`nn_r1v_*`，视觉头已按 §10.5 删除）；真实冒烟还没做。<br>• P2：先验臂可以在单通道数据集上静默训练；`--ref` 默认值不随 `--gt` 变；空 HD95 数组会进 bootstrap；541 构建不检查 540 的完成标记；写入不是原子的；已有标签复用时不校验；运行记录缺代码哈希和库版本；验证后没有清 Gaussian 缓存 |
| 2026-10-27 | **第 6 轮意见的修复**（提交见 git log） | • `v42_split_index` 只接受 fold 0–4 和 dev/report，binY 不接受 inner；train42 还要求数据集自己的 splits_final.json 里有可用条目。<br>• `harm_cache` 逐例核对标签数组的 sha256，核对 `trainer/v4/harmonize.py` 自校准后没改，返回标签摘要；摘要写进 v42_meta.json、完成标记和每次运行记录（`<fold 目录>/v42_runs.jsonl`，另含 git HEAD、已安装训练器文件的哈希、库版本）。<br>• Stage 1：验证结果缺 npz 时自动带 `--npz` 重做验证；内层验证病例都有 npz 才写完成标记；冒烟训练器不写标记。<br>• 先验臂遇到通道数不对，在构建网络时就报错；train42 也拒绝在 Dataset540 上用先验臂。<br>• prep42：标签每次按核验过的缓存重写，CT 和标签都先写临时文件再改名；`--cases` 遇到原始目录里已有其他病例时拒绝；541 要求 540 已完成。<br>• v4_dev_summary：`--ref` 默认随 `--gt`（orig → r0_dev，harm_v1 → v42_r0u）；评价目录带 GT.json，参考标准不符时拒绝；空数组不进 bootstrap。<br>• V4 验证结束清 Gaussian 缓存，与 stock 一致。<br>• 新增 stock 冒烟训练器 `nnUNetTrainer_SMOKE_2x5` 和脚本 `tests/v4_nnunet_smoke42.sh`。<br>• 测试新增 8 项：fold 越界、train42 拒绝条件、缓存哈希失配、导出方向往返（两种验证格式）、队列完成路径、评价目录标记、先验臂通道检查、`--cases` 防护。nnU-Net 环境 31 项通过；topocnv 环境 15 项通过，1 项需要 skimage，跳过。<br>• 队列：备份 `log/v4/jobs.json`（`jobs.json.bak_20261027_v42`）后删掉 8 个 `nn_r1v_*`，删掉的条目存在 `jobs_removed_20261027_v42.json`；81 → 73 条；STOP 保留。<br>• 未修：harm 构建脚本的符号链接目标检查和 QA 里复核哈希。缓存只建一次，已经两轮独立核验；改这个脚本会让它与 harm_v1.json 记录的脚本哈希不一致 |
| 2026-10-27 | **两个数据集用 9e01dd8 重建并复核**（CPU；540 用时 20:17–20:21，541 用时 20:21–20:25） | • prep42 先核对 154 例标签的 sha256，全部与 manifest 一致。两个数据集的 v42_meta.json 和完成标记都写入了配置哈希 1ef730c30545 和标签摘要 3a0abb6f98e1。<br>• 逐例复核结果与第一次相同：CT 与 Dataset501 逐字节相同；厚标注 109 例标签不变；薄标注 45 例只多出 161,320 个体素。plans、splits、fingerprint 和预解包都正常。<br>• 记录的库版本：nnunetv2 2.5.2、torch 2.6.0+cu124、numpy 1.26.4、scipy 1.13.1、scikit-image 0.24.0、nibabel 5.3.3。这是 nnU-Net venv 的版本。harm_v1 构建时没有记录库版本；构建要用 skimage，topocnv 环境里没有 |
| 2026-10-27 | **qoder 第 7 轮复核**（审查 9e01dd8，约 33 分钟；`~/cnv/qoder_review7_20261027.log`） | • 第 6 轮的 P1-1 至 P1-4 已关闭；P1-5（GPU 冒烟）还没做；没有新的代码级 P1。5 例抽查通过。变异测试也通过：删掉越界检查或标签哈希校验后，对应的测试会失败。<br>• 新的 P2，待修：子集或冒烟导出也会写 EXPORTED；train42 训练前不核对数据集的标签摘要；重建时没有刷新 fingerprint（缺 `--clean`）；图像不强制重写，也没有图像摘要；元数据写入不是原子的；npz 只查是否存在；stock 判定靠类名前缀；train42 的负向测试用的是正式 NNUNET_HOME；V4_PLAN 第 15 行还写着"已冻结"。<br>• 排产前要做：PI 确认方向并另写记录；跑冒烟；删 STOP；先加两个 nnprep42 作业，再加训练作业，否则训练作业会一直等依赖 |
| 2026-10-27 | **V4.2 确认**（主会话转达用户原话："确认按薄→厚统一（harm_v1，t=0.19），可以放开 v4 生产队列。"） | • V4_PLAN §11 状态改为已确认，方向一栏注明用户确认；医生复核/补勾的备选方案仍在 §11.10，不阻塞。<br>• 确认记录 `configs/v4_harm/harm_v1_direction.json`（写入前完整核对 154 例标签；配置 1ef730c30545，标签摘要 3a0abb6f98e1）。harm_v1.json 不改。<br>• 主会话要求：第 0 步做完后按 V4.2 放开队列，先跑开发阶段，R0u 在统一标签上重训；每张卡最多 1 个本会话的 run，空闲显存 ≥ 40 GB 才启动，不向别人的进程发信号；开发阶段选择定下来、准备冻结时先汇报，确认后才进报告阶段 |
| 2026-10-27 | **GPU 冒烟第 1 次**（用户授权；`tests/v4_nnunet_smoke42.sh`，GPU 6，代码 8a8f07b；日志 `log/v4_smoke42.log`） | • 通过（SMOKE42_OK）。4 例小数据集（训练用 3 个内层折各 1 例，验证用 V_0 的 1 例薄标注）。<br>• V4 训练器（S1.0）→ validation_v4 → export42，stock 训练器 → validation/*.nii.gz → export42，两条路径都导出到缓存网格，评价器各算出 1 例。<br>• Stage 1 `--npz`：probabilities 形状 (5, 154, 512, 512)、float32，argmax 与转置后的 NIfTI 100% 一致，即 npz 已是缓存轴序。<br>• 没有写任何正式完成标记。旧代码给子集导出写了 EXPORTED（第 7 轮 P2-1），名字是 smoke42_*，不影响正式结果 |
| 2026-10-27 | **第 7 轮 P2 的修复**（提交见 git log） | • train42 启动前核对方向确认记录和数据集元数据，两者都要对上当前缓存的配置哈希和标签摘要（`check_dataset42`）。<br>• stock / V4 的输出路径按 nnU-Net 实际解析到的类判断，不再看名字前缀；运行记录哈希该类继承链上的全部 nnunetv2 文件（R0u 的 300epochs 和 stock 基类也在内）。<br>• Stage 1 的 npz 按表头核对：能读、有 probabilities、形状 (5, *缓存形状)。<br>• 子集或冒烟训练器的导出只写 EXPORTED_PARTIAL，不写队列和开发汇总等的 EXPORTED。<br>• prep42：已有 CT 先读回比对，不一致就重写；v42_meta 和完成标记加 CT 摘要；指纹带 `--clean` 重算；dataset.json、v42_meta、plans、splits 先写临时文件再改名；重建开始就删旧完成标记；541 的软链接也原子替换。<br>• harm_cache 同时核对构建脚本的哈希。<br>• 测试：新增方向记录与元数据核对、npz 表头、按类解析训练器；train42 和 prep42 的负向测试改用哨兵和临时目录，守卫失效时也不会写正式目录或启动训练。nnU-Net 环境 34 项通过；topocnv 环境 17 项通过，1 项需要 skimage，跳过。<br>• 未改：harm 构建脚本本身（缓存只建一次）；冒烟只覆盖开发路径，报告路径的索引由单元测试覆盖 |
| 2026-10-27 | **数据集第 3 次构建 + GPU 冒烟第 2 次**（修复后的代码；21:52–22:09；日志 `log/v4_final42.log`、`log/v4_smoke42b.log`） | • 两个数据集用新的 prep42 重建，指纹已重算。完成标记写入配置 1ef730c30545、标签 3a0abb6f98e1、CT 837023fa8b0d。逐例复核全部通过，结果与前两次相同。<br>• 冒烟第 2 次（GPU 6）通过：子集导出只写 EXPORTED_PARTIAL；运行记录里有训练器类继承链上各文件的哈希；npz 检查同第 1 次。没有写任何正式完成标记。<br>• 行尾说明：这次部署的 `tools/v4_nnunet.py` 在本地拼接时被写成了 CRLF，提交前已改回 LF，内容逐字节相同（去掉 CR 后）。所以两个数据集 v42_meta.json 里的 tool_sha256 是 dbe52c28cc86（CRLF 版），已提交文件是 d8d66f5c8c43 |
| 2026-10-27 | **放开 V4.2 生产队列，先跑开发阶段**（22:12；`tools/v4_queue.py --max_per_gpu 1 --max_gpu_jobs 4 --min_free_mb 40960`；排队脚本 `~/cnv/claude_check_20261027/queue42.sh`） | • 新增 47 个作业。2 个 nnprep42（标记已在，直接算完成）；按折交错的 Stage 1 dev（每折 3 个内层模型）和 D4′（v42_r0u = stock R0u、v42_s05、v42_s10）共 30 个；最后是 Stage 1 report 层 15 个。<br>• 报告阶段的臂（R0u 报告、B、M 等）没有排，等冻结后向主会话汇报、确认了再排。<br>• 备份 `log/v4/jobs.json.bak_20261027_v42q` 和 `queue_state.json.bak_20261027_v42q`；queue_state 里删掉 8 条过期的 nn_r1v_*；STOP 已删除。cnv-sota 队列 35 个作业都已完成，不用让卡。<br>• 第一批 4 个，每卡 1 个：s142_dev_f0_i0/i1/i2 在 GPU 6/4/1，nn42_v42_r0u_dev_f0 在 GPU 7，启动前每张卡空闲显存都在 57 GB 以上。第一个 epoch 约 38 秒，一个 run 估计 3–3.5 小时。运行记录的 git HEAD 是 0e867ce，标签和 CT 摘要与数据集一致。<br>• D2′ 之前还缺：用 Stage 1 概率在统一标签上建先验（D1-AM2）→ 4 通道全图数据集（542–556）→ D2′ 作业和 V4.2 选择器。这部分代码还没写 |
| 2026-10-28 | **D2′ 链：代码、qoder 第 8 轮、排队**（规则 5944b98 先于任何 D4′/D2′ 结果提交；代码 046a2dc，修复见下一个提交） | • 新代码：`v4_nnunet.py predict42 / prep42p / train42 --dataset prior`、`tools/v4_build_priors42.py`（D1-AM2）、`tools/v4_select42.py`（D4′/D2′ 选择与 freeze）、`v4_jobs.py d2chain42 / smoke42p`、`tests/test_v4_d2.py`、`tests/v4_nnunet_smoke42p.sh`。<br>• qoder 第 8 轮（`~/cnv/qoder_review8_20261027.log`）：无 P0。3 条 P1 已修：先验臂预测时 label_manager 按臂的头数设置（否则 7 头 smoke 臂会崩）；Stage 1 对 V_k 的推理也等 smoke 通过；d2chain42 缺 smoke42p 作业时拒绝。P2 已修 8 条：续跑的输出绑定检查点和工具哈希；部分导出不能与完整导出共存，选择器同时拒绝；freeze 只能五折一起；D1-AM2 的 write-once 键加 Stage 1 检查点和触发病例；smoke 开始先删旧标记；残留文件检查覆盖全部通道和标签；Stage 1 概率检查网格和有限值；先验构建并行数 24 → 12。P2-6（smoke 标记比较）在审查前已改。<br>• CPU 对比（smoke Stage 1 模型，1 例）：predict42 与 nnU-Net 验证输出最大差 0.20，argmax 一致 99.82%。差异来自 CPU 全精度和 GPU 半精度的不同，模型只训了 2 个 epoch、概率接近均匀，所以这次不算结论；逐体素一致性由 GPU 冒烟判定（同卡，容差 1e-3）。<br>• 单元测试：nnU-Net venv 43 OK，topocnv 26 OK（1 跳过）。 |
| 2026-10-28 | **用户 2026-10-28 叫停 D2′**（用户原话「D2′ 先验实验 先停掉」，主会话转达） | • 16:23 撤掉 jobs.json 里所有还没完成的相关作业，共 100 个，名单见 `log/v4/d2_stop_removed_20261028.json`。撤前备份 `log/v4/jobs.json.bak_20261028_d2stop` 和 `queue_state.json.bak_20261028_d2stop`。<br>• 撤掉的作业：<br>&nbsp;&nbsp;– D2′ 链：smoke42p（已经 3 次失败，见下）、s1p42_dev ×15、pri42_dev ×5、nnprep42p_dev ×5、nn42_v42_d2{ab,ac,bc,abc} ×20、sc42_v42_d2* ×20、d2sel42 ×5；<br>&nbsp;&nbsp;– 只为先验服务的 Stage 1：s142_dev_f3_i2、s142_dev_f4 ×3、s142_report ×15；<br>&nbsp;&nbsp;– V4.3 用不到的 D4′ 剩余折：nn42_v42_s05/s10_dev_f3/f4 共 4 个、对应 sc42 共 4 个、d4sel42_f3/f4。<br>• 停掉 3 个正在跑、本会话队列起的进程（16:25，TERM 发给各自的进程组，rc = −15）：s142_dev_f3_i2（GPU 7）、nn42_v42_s05_dev_f3（GPU 1）、nn42_v42_s10_dev_f3（GPU 4）。没有动别人的进程。<br>• 保留：<br>&nbsp;&nbsp;– R0u 开发层 fold 3（在跑）、fold 4（16:25 起跑），以及它们的 sc42_v42_r0u 打分（V4.3 选择要用）；<br>&nbsp;&nbsp;– 新排报告层 R0u：nn42_v42_r0u_report_f0–f4，作为多模态的对照，优先级排在 V4.3 开发 run 之后。<br>• 已有结果原样保留：<br>&nbsp;&nbsp;– Stage 1 dev fold 0–2 全部 9 个模型，fold 3 的 i0、i1；<br>&nbsp;&nbsp;– D4′ fold 0–2 的 run 和 `results/v4_select42/fold{0,1,2}/d4.json`，三折都选 S0：fold 0 0.6640 / 0.6557 / 0.6354，fold 1 0.7035 / 0.6952 / 0.6745，fold 2 0.6830 / 0.6777 / 0.6645；<br>&nbsp;&nbsp;– D2′ 代码（529dd0c）。先验还没建，没有 D2′ 结果。<br>• smoke42p 的状态：GPU 冒烟里 predict42 与 nnU-Net 验证输出的对比 3 次都没过容差（最大差 0.21–0.24，argmax 一致 99.99%，平均差 5e-5，p99.99 为 1.4e-3，超过 0.05 的体素 40M 里只有 52 个，散在全图）。这像是 2 epoch 未训练模型上的半精度噪声，不像流程错误。另起了一项 CPU 对比，用训练好的 Stage 1 模型核实。D2′ 叫停后这件事不再挡路，但 Stage 1 推理代码如果以后要用，要先把这一项查清。 |
| 2026-10-28 | **V4.3（路线 B）：规则、截断方向统计、代码** | • §12 在 66f3d73 写定，早于任何多模态结果。§12.8（零影响测试、种子噪声）随后补入，选择规则不变。<br>• 截断方向（`tools/v4_mm43_trunc_stats.py`，只看冻结的 MRI 视野）：<br>&nbsp;&nbsp;– 覆盖率 < 95% 的有 51 例，缺失的 Y 体素颅侧 30,721、尾侧 19,115、侧方 0；<br>&nbsp;&nbsp;– 按主导侧算：颅侧 25 例，尾侧 18 例，混合 8 例；<br>&nbsp;&nbsp;– 定为两侧各 0.5，截断面倾斜 0–20°。<br>• 代码 dcbe4b9：`trainer/v4/nnunet/nnUNetTrainerV43.py`、`tools/v4_mm43.py`、`tools/v4_select43.py`、`v4_jobs.py mm43`、`tests/test_v4_mm43.py`（venv 17 OK，topocnv 17 OK，其中 8 跳过）、`tests/v4_nnunet_smoke43.sh`。<br>• qoder 第 9 轮审查在跑；smoke43 16:25 在 GPU 4 启动。Dataset557 和开发 run 等审查通过后再排。 |
