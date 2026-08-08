# 悠悠球模型与追踪报告：2026-08-08

本报告只记录当前有效结论，覆盖悠悠球检测、绳线分割、三分类方向识别，以及连续视频中的悠悠球/绳线追踪。训练与评估均在 `.venv` 中执行；详细 checkpoint、manifest 和逐帧结果保留在 `runs/`。

## 当前默认方案

| 任务 | 当前方案 | 默认行为 |
| --- | --- | --- |
| 悠悠球检测 | YOLO11s，约 9.4M 参数 | `imgsz=1024`，常规单次推理 |
| 绳线分割 | 三权重 LR-ASPP 路由 | 普通域双路融合，一般弱域高分辨率双路，超弱域单主模型 |
| 绳线追踪 | 语义分割 + 概率门控颜色/Hough + 按需光流 | 新鲜观测优先，仅缺帧时传播 |
| 方向识别 | 仅悠悠球 ROI 的三分类模型 | 5/25 FPS 自适应采样 + EMA/滞回 |
| RTMPose | 可选审核分支 | 默认关闭，不参与上述四项主任务输出 |

Workbench 默认模型和参数与 `config.yaml`、`config.py` 一致。

## 悠悠球检测

生产权重：

`runs/candidates/yoyo_unified_396ce5fa8e73_detection_yolo11s_s-current-capacity60/weights/best.pt`

SHA-256：`ac76000388bc81f442e860b6aac68487406205f27e89f31aab16e0c52e82f705`。

65 张扩展 test（57 个正样本）的最终结果：Precision `0.9522`、Recall `0.8596`、mAP50 `0.9415`、mAP50-95 `0.5425`。1024 输入下 mAP50 `0.9240`、mAP50-95 `0.5168`、Recall `0.8269`；单图推理约 `8.9 ms`，因此生产输入由 1280 降为 1024。

454 帧、五段 reviewed 连续区间的 detector-only 结果为：

| 指标 | 结果 |
| --- | ---: |
| Precision / Recall / F1 | `1.000000 / 0.990544 / 0.995249` |
| Mean IoU | `0.837197` |
| Detector-only 吞吐 | `5.9737 FPS` |

运行时低置信 TTA 已移除：它只净增 1 个 TP，却引入 3 个 FP，并将速度降至 `5.7466 FPS`。训练 manifest 位于候选目录，最佳 epoch 为 16。

## 绳线分割

生产候选：

`runs/candidates/yoyo_unified_42086e82249d_semantic_string_degradation-aug-lr5e6-a80-v1/`

| 权重 | 用途 | SHA-256 |
| --- | --- | --- |
| `weights/primary.pt` | 普通域主模型 | `72bfa24275261248f69ada0325f81876067909468c30105a8f93c92bada508f3` |
| `weights/secondary.pt` | 固定副模型 | `640c4ac5b59c2f70aee1c45ebca78774b78983e4251d03d065267576310223df` |
| `weights/adaptive.pt` | 退化增强弱域主模型 | `ce6e07e9abf689adbd49fdc74702809d24c433cc827e3535ee0970efd16c9a0f` |

普通域以 `960x544` 输入，将主模型和副模型相对各自阈值校准后按 `0.7/0.3` 融合。最近 12 次观测同时满足颜色候选通过数为 0、平均 confidence `<0.82`、平均悠悠球距离比例 `>0.018` 时，从下一帧单向进入弱域：

- 一般弱域：`1440x816`，弱域主/副模型按 `0.5/0.5` 融合。
- 超弱域：触发窗口平均 confidence `<0.30` 时，只运行弱域主模型，阈值 `0.55`，最多保留 2 个语义组件。

65 张 canonical test（59 张有绳、6 张无绳）的代表性结果：

| 方案 | Pixel Dice | Tolerant F1@3 | Presence F1 | 负图平均误检 |
| --- | ---: | ---: | ---: | ---: |
| 单 LR-ASPP | 0.583385 | 0.859735 | 0.983333 | 14.167 px |
| 普通域校准双路 | 0.592413 | 0.868790 | 0.983333 | 10.000 px |
| 弱域权重选择结果 | **0.599546** | **0.877097** | 0.983333 | **4.333 px** |

“弱域权重选择结果”使用 `alpha=0.80`，只用于 checkpoint 选择；生产连续视频仍使用上面的分域参数。Workbench 仅在选择该候选主权重时自动加载匹配的副权重和弱域权重。

## 连续视频追踪

评估使用 `datasets/1Ayoyo_consecutive` 的 reviewed yoyo box，以隔离绳线策略和 detector 误差。当前流程包含：分域 LR-ASPP、语义支持邻域内的饱和色/亮脊 Hough、沿线概率门控、观测优先状态更新，以及只在缺少新鲜观测时执行的 Lucas-Kanade 前后向光流。

最终 8 组、757 帧结果：

| 序列 | 帧数 | Precision@8 | Recall@8 | F1@8 | Chamfer | Presence F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 周博文 | 72 | 0.678371 | 0.512807 | 0.584083 | 18.8245 | 1.000000 |
| 唐浩翔 | 100 | 0.799403 | 0.619221 | 0.697869 | 23.2869 | 1.000000 |
| DSCF7145 | 95 | 0.584312 | 0.429525 | 0.495102 | 81.9063 | 1.000000 |
| 池高宇，67 帧 | 67 | 0.753447 | 0.277451 | 0.405558 | 65.4543 | 1.000000 |
| 池高宇前场 | 120 | 0.702356 | 0.405545 | 0.514192 | 24.6535 | 1.000000 |
| 华南赛 | 98 | 0.806788 | 0.606642 | 0.692544 | 11.6624 | 0.983784 |
| 池高宇 f4200-f4299 | 100 | 0.449351 | 0.128170 | 0.199450 | 446.0609 | 0.868571 |
| namdongxun | 105 | 0.889906 | 0.518684 | 0.655378 | 22.7764 | 0.990385 |

Pooled Precision/Recall/F1@8 为 `0.706441/0.441152/0.543133`，帧均 Chamfer 为 `88.4406 px`。最终结果位于：

`runs/experiments/semantic_adaptive_ultraweak_single_production_verified_757/summary.json`

### 关键消融

低饱和亮脊补线是弱场景的主要几何增益。加入亮脊前后，7 组 652 帧 pooled Precision/Recall/F1@8 从 `0.676094/0.382865/0.488881` 提升至 `0.697974/0.434477/0.535570`，帧均 Chamfer 从 `92.1143` 降至 `86.3637 px`；七组均无指标回退。

超弱单模型路由只命中 `池高宇 f4200-f4299`，其他七组严格不变。该组变化如下：

| 指标 | 一般弱域双路 | 超弱单主模型 |
| --- | ---: | ---: |
| Precision / Recall / F1@8 | 0.318856 / 0.077319 / 0.124459 | **0.449351 / 0.128170 / 0.199450** |
| Chamfer | 532.2312 px | **446.0609 px** |
| Presence F1 | 0.767296 | **0.868571** |
| 零预测帧 | 37 | **21** |

该区间仍是当前最弱场景，绝对 Recall@8 只有 `0.128170`。主要限制是语义概率只覆盖局部白绳；后续应增加同域独立训练样本，而不是继续放宽几何门控。

### 运行时优化

已保留的优化都通过了输出或指标等价检查：

| 优化 | 代表性验证结果 |
| --- | --- |
| 按需光流与延迟灰度转换 | 新鲜观测帧不再做无效光流；552 帧指标/几何不变 |
| GPU 概率校准融合 + CUDA Graph | 静态阈值图不变；552 帧指标不变 |
| 语义支持 ROI 内执行颜色/Hough | 552 帧 F1/Recall 提升，Chamfer 降低；墙钟约降低 20% |
| 饱和色/亮脊两轮复用 ROI、支持图、HSV 和 mask | 757 帧逐帧 SHA-256 与全部指标相同 |
| 检测期间并行 semantic letterbox | 默认 30 帧逐帧与视频 SHA-256 相同 |
| 异步有界 MP4 写入 | 输出视频和 JSONL 逐字节相同 |
| 无可消费锚点时跳过语义前向 | 60 帧输出相同，语义前向 `60 -> 0` |

最新两轮颜色搜索缓存的 100 帧生产 A/B 中，tracking loop 中位数 `13.7619 -> 12.8433 s`（缩短 `6.68%`），吞吐 `7.2668 -> 7.7947 FPS`；四次 `frames.jsonl` 和 `tracked.mp4` 均分别逐字节一致。证据位于 `runs/experiments/semantic_color_twopass_cache_candidate_757/` 和 `runs/experiments/color_search_cache_ab_*/`。

## 三分类方向识别

数据视图：`datasets/1Ayoyo_dataset/orientation`，训练/验证/test 为 810/65/65 张，只使用悠悠球 ROI。

生产权重：

`runs/candidates/yoyo_unified_2b0cfca8743a_orientation_roi_9cd9d9361ab5_best_yoyo-only-final-warm-freeze10-lr1e4-v1/weights/best.pt`

SHA-256：`f00e3766c05d9ae7dc3fe13a9cd45faf3507aab4c9a9acfa6df73b155ff7cd91`。

| 方案 | Top-1 | Macro recall | horizontal | normal | not_applicable |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧上下文 ROI | 0.8000 | 0.6562 | 0.5000 | 0.9130 | 0.5556 |
| 最终仅悠悠球 ROI | **0.9231** | **0.8818** | **0.8000** | **0.9565** | **0.8889** |

连续帧采用三类概率 EMA `alpha=0.4`、切换 margin `0.05`、连续 3 次确认；高置信大幅领先时允许快速切换。稳定状态以 5 FPS 推理，不稳定状态提升至 25 FPS，连续 4 次稳定后恢复低频。

6 组、552 帧 reviewed 结果：

| 策略 | Accuracy | Macro recall | 输出切换 | 超额切换 |
| --- | ---: | ---: | ---: | ---: |
| 固定 5 FPS + carry | 0.882246 | 0.850290 | 13 | 9 |
| **5/25 FPS 自适应 + EMA/滞回** | **0.942029** | **0.922041** | **5** | **1** |

真实 detector 框的 DSCF7145 95 帧 A/B 中，Accuracy `0.947368 -> 0.968421`，平均/最大边界延迟 `2.5/3 -> 1.5/2` 帧，吞吐差约 `0.45%`。证据位于 `runs/experiments/orientation_temporal_adaptive_final_20260807/metrics.json` 和 `runs/experiments/orientation_runtime_ab/`。

## RTMPose

RTMPose-m WholeBody 与 YOLOX-m 仍作为可选人体/手部审核分支保留，但不参与当前检测、绳线或仅悠悠球 ROI 的方向预测。严格开关 A/B 中，关闭 RTMPose 后四项主输出不变，tracking loop 从 `0.9989` 提升至 `1.6117 FPS`，因此 Workbench 与 CLI 默认关闭 pose；需要 133 点姿态审核时可显式启用 `--pose`。

## 验证状态

- 上次完整验证：`pytest -q`，164 项通过。
- `compileall` 与 `git diff --check` 通过。
- 两个数据集共 914 个 canonical JSON，pose/手部键残留数为 0。

## 复现命令

```powershell
.\.venv\Scripts\python.exe -m training_v3.evaluate runs\candidates\yoyo_unified_2b0cfca8743a_orientation_roi_9cd9d9361ab5_best_yoyo-only-final-warm-freeze10-lr1e4-v1 --device 0
.\.venv\Scripts\python.exe -m cli.training.train_semantic --dataset-dir datasets\1Ayoyo_dataset\string_segmentation --project runs\experiments --name semantic_degradation_aug_lr5e6_v1 --epochs 20 --input-width 960 --input-height 544 --batch 2 --lr 0.000005 --architecture lraspp_mobilenet_v3 --initial-weights runs\candidates\yoyo_unified_f5775b248d3b_semantic_string_lraspp_soup-a25-v1\weights\best.pt --degradation-augment --early-stopping-patience 5 --early-stopping-min-epochs 6 --device cuda
.\.venv\Scripts\python.exe -m cli.tracking.evaluate_orientation --output-dir runs\experiments\orientation_temporal_adaptive_20260807 --device 0
.\.venv\Scripts\python.exe -m pytest -q
```
