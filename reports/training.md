# 悠悠球模型与追踪报告：2026-08-08

本报告只记录当前有效结论，覆盖悠悠球检测、绳线分割、三分类方向识别，以及连续视频中的悠悠球/绳线追踪。训练与评估均在 `.venv` 中执行；详细 checkpoint、manifest 和逐帧结果保留在 `runs/`。

## 当前默认方案

| 任务 | 当前方案 | 默认行为 |
| --- | --- | --- |
| 悠悠球检测 | YOLO11s，约 9.4M 参数 | `imgsz=1024`，常规单次推理 |
| 绳线分割 | 单 MobileNetV3-FPN，约 3.0M 参数 | `960x544` 单次推理，验证阈值 `0.6471` |
| 绳线追踪 | 语义分割 + 概率门控颜色/Hough + 按需光流 | 新鲜观测优先，仅缺帧时传播 |
| 方向识别 | 仅悠悠球 ROI 的三分类模型 | 5/25 FPS 自适应采样 + EMA/滞回 |
| RTMPose | 可选审核分支 | 默认关闭，不参与上述四项主任务输出 |

Workbench 默认模型和参数与 `config.yaml`、`config.py` 一致。

## 悠悠球检测

生产权重（本机发布产物，GitHub 提交不包含二进制文件）：

`runs/candidates/yoyo_unified_1f05c056cb5d_detection_yolo11s_prod-ft-lr1e5-v1/weights/best.pt`

SHA-256：`05fe3d46cfc175034b3e3e4e69b6af2cf7596863839cbe91a7bfb792b1f6f894`。部署机器需将该本机权重放在上述路径；仓库只提交 manifest 与指标记录。

候选从上一生产权重出发，在当前 v5 训练集以 `lr=1e-5` 微调；训练上限 15 epoch，patience 触发后实际完成 7 epoch，最佳 checkpoint 来自 epoch 2。初始化 lineage 与当前 val/test source group 无交集。固定 65 张 test（57 个正样本）、`imgsz=1024` 的同口径对比如下：

| 权重 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| 旧生产 YOLO11s | 0.994193 | 0.807018 | 0.921067 | 0.512976 |
| **当前微调 YOLO11s** | **1.000000** | **0.824142** | **0.947232** | **0.551444** |

未参与该微调的 221 张 stride-3 reviewed 连续帧上，Recall `0.859818 -> 0.892425`、mAP50 `0.886691 -> 0.906043`、mAP50-95 `0.592889 -> 0.596493`；模型结构、参数量和生产输入不变。

周博文 72 帧完整 pipeline A/B 只替换检测权重：悠悠球 Presence F1 `0.985915 -> 1.000000`、Mean IoU `0.830264 -> 0.847169`、中心误差 `7.1239 -> 4.8022 px`；更稳定的锚点也使绳线 F1@8 `0.649541 -> 0.664403`、Chamfer `20.2388 -> 17.0302 px`。同架构推理耗时持平，首次真实 pipeline 顺序 A/B 墙钟为 `36.79 -> 30.62 s`。

YOLO26s、运动模糊回放和连续帧回放候选均未晋升：YOLO26s 当前 test mAP50-95 仅 `0.476189`；运动模糊回放降低 Recall；连续帧回放虽提高 Recall，但 test mAP50-95 低于当前候选。运行时低置信 TTA 继续保持移除。

## 绳线分割

生产候选（本机发布产物，GitHub 提交不包含二进制文件）：

`runs/candidates/yoyo_unified_b36a77f2e354_semantic_string_mobilenetv3-fpn-single-v1/weights/best.pt`

SHA-256：`8b953763c6102e767d060fc7af7d13dcff353fa8ef5c73d03cdf36ee4dda4468`。

模型以 ImageNet MobileNetV3-Large 为编码器，用 `/2、/4、/8、/16、/32` 特征构建轻量 FPN。训练前 5 个 epoch 冻结编码器；随后解冻，编码器学习率为解码器的 `0.05x`。小 batch 训练期间固定编码器 BatchNorm 统计，避免随机解码头扰动预训练分布。最佳 epoch 为 36，参数量约 `3.02M`。

65 张 canonical test（59 张有绳、6 张无绳）的代表性结果：

| 方案 | Pixel Dice | Tolerant F1@3 | Presence F1 | 负图平均误检 |
| --- | ---: | ---: | ---: | ---: |
| 单 LR-ASPP | 0.583385 | 0.859735 | 0.983333 | 14.167 px |
| 普通域校准双路 | 0.592413 | 0.868790 | 0.983333 | 10.000 px |
| 旧弱域权重选择结果 | 0.599546 | 0.877097 | 0.983333 | **4.333 px** |
| **单 MobileNetV3-FPN** | **0.713177** | **0.925824** | 0.983333 | 43.333 px |

FPN 在像素和容差几何上有大幅提升，59 张正图的 presence recall 为 `1.0`。6 张负图中旧方案和新方案都误判 2 张；新方案误检面积更大，因此负图像素误检是明确局部回退。生产流程仍要求悠悠球/历史绳线锚点，并在连续集上单独验证误检与几何。

## 连续视频追踪

评估使用 `datasets/1Ayoyo_consecutive` 的 reviewed yoyo box，以隔离绳线策略和 detector 误差。当前流程为：单 FPN 语义分割、语义支持邻域内的饱和色/亮脊 Hough、沿线概率门控、观测优先状态更新，以及只在缺少新鲜观测时执行的 Lucas-Kanade 前后向光流。不再加载副模型、弱域模型或执行分域路由。

最终 8 组、757 帧结果：

| 序列 | 帧数 | 旧 F1@8 | 新 F1@8 | 旧 Chamfer | 新 Chamfer |
| --- | ---: | ---: | ---: | ---: | ---: |
| 周博文 | 72 | 0.584083 | **0.657621** | 18.8245 | **17.3310** |
| 唐浩翔 | 100 | 0.697869 | **0.703379** | **23.2869** | 33.1537 |
| DSCF7145 | 95 | **0.495102** | 0.479661 | **81.9063** | 128.3117 |
| 池高宇，67 帧 | 67 | 0.413072 | **0.485520** | 64.7856 | **43.1967** |
| 池高宇前场 | 120 | 0.526797 | **0.562186** | **24.0786** | 29.3198 |
| 华南赛 | 98 | 0.692544 | **0.744656** | **11.6624** | 11.6659 |
| 池高宇 f4200-f4299 | 100 | 0.199450 | **0.470934** | 446.0609 | **24.3131** |
| namdongxun | 105 | 0.655378 | **0.735113** | 22.7764 | **13.5036** |

Pooled Precision/Recall/F1@8 从 `0.711532/0.443424/0.546359` 提升至 `0.810104/0.459328/0.586252`；帧均 Chamfer/HD95 约从 `88.29/195.68 px` 降至 `37.30/118.46 px`。最弱的池高宇 f4200-f4299 区间 F1 提升 `136%`，Chamfer 降低 `94.5%`。

DSCF7145 的 F1 和 Chamfer 明确回退；唐浩翔、池高宇前场的 Chamfer 也有局部回退。保留候选的依据是 pooled 五项指标全部提升、最弱域大幅改善、6/8 组 F1 提升，以及模型数和运行成本同时下降。结果位于 `runs/experiments/semantic_mobilenetv3_fpn_single_production_757/summary.json`。

### 运行时优化

`960x544` 的直接 GPU 基准中，单 FPN 为 `7.05 ms`，两个 LR-ASPP 前向合计约 `8.85 ms`。`namdongxun` 105 帧真实 detector Workbench A/B 中，旧三模型为 `8.5255/8.5335 FPS`，单 FPN 为 `10.0651/8.8566 FPS`；两次候选输出 MP4 SHA-256 均为 `e40854b86e9651c73e7f027b7e76e441cf4aca8d2ba248b32dfe349ce9125241`。

继续保留的通用优化包括：检测期间并行 semantic letterbox、语义支持 ROI 内颜色/Hough、饱和色/亮脊缓存复用、按需光流、异步有界 MP4 写入，以及无锚点时跳过语义前向。旧 LR-ASPP 专用的 GPU 融合、双路 CUDA Graph、弱域门控和三档校准不再进入默认流程。

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

- 本轮完整验证：`pytest -q`，164 项及 2 个子测试通过。
- `compileall` 与 `git diff --check` 通过。
- 两个数据集共 914 个 canonical JSON，pose/手部键残留数为 0。

## 复现命令

```powershell
.\.venv\Scripts\python.exe -m training_v3.evaluate runs\candidates\yoyo_unified_2b0cfca8743a_orientation_roi_9cd9d9361ab5_best_yoyo-only-final-warm-freeze10-lr1e4-v1 --device 0
.\.venv\Scripts\python.exe -m cli.training.train_semantic --dataset-dir datasets\1Ayoyo_dataset\string_segmentation --project runs\experiments --name semantic_degradation_aug_lr5e6_v1 --epochs 20 --input-width 960 --input-height 544 --batch 2 --lr 0.000005 --architecture lraspp_mobilenet_v3 --initial-weights runs\candidates\yoyo_unified_f5775b248d3b_semantic_string_lraspp_soup-a25-v1\weights\best.pt --degradation-augment --early-stopping-patience 5 --early-stopping-min-epochs 6 --device cuda
.\.venv\Scripts\python.exe -m cli.tracking.evaluate_orientation --output-dir runs\experiments\orientation_temporal_adaptive_20260807 --device 0
.\.venv\Scripts\python.exe -m pytest -q
```
