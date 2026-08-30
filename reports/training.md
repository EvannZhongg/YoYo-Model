# 悠悠球识别与追踪模型报告

## 当前默认方案

| 模块 | 当前模型与权重 | 默认输入/参数 |
| --- | --- | --- |
| 悠悠球检测 | YOLO11s，约 9.4M 参数；`runs/experiments/det_replay_soup_a25/weights/best.pt` | `imgsz=1024`，置信度 `0.15`，IoU `0.7` |
| 绳线分割 | MobileNetV3-FPN，约 3.0M 参数；`runs/experiments/semantic_soup_centerline_w005_a050/weights/best.pt` | `960x544`，配置阈值 `0.40` |
| 绳线追踪 | 单语义模型 + 颜色/亮脊候选 + 按需光流 | 每帧语义推理，最多传播 12 帧 |
| 方向识别 | 悠悠球 ROI 三分类模型；`runs/candidates/yoyo_unified_2b0cfca8743a_orientation_roi_9cd9d9361ab5_best_yoyo-only-final-warm-freeze10-lr1e4-v1/weights/best.pt` | 5 FPS 稳态、25 FPS 突发，EMA 与滞回 |
| 姿态审核 | RTMPose-m WholeBody（可选） | 默认关闭，不参与主输出 |

Workbench 和 CLI 都从 `config.yaml`、`config.py` 读取以上默认值。训练数据身份由
`datasets/1Ayoyo_dataset/string_segmentation/manifest.json` 固定，SHA-256 为
`f2c1d0269f6c3cb8ce5177b5a269d374179ab72e42baf39f88c591cec5439195`。

## 悠悠球检测模型

当前权重 SHA-256：
`2d5a0e45b9da1aa88609c79015ce7b651e86fb8206d9ae6463f0fa72cf4a0e00`。

候选在生产 YOLO11s 权重上以训练集 48 张 Telea 去除悠悠球的合成负样本 replay 两轮，
`epochs=4`、`imgsz=1024`、AdamW `lr0=1e-6` 微调，再与基线按 `alpha=0.25` 做
参数 soup；数据 manifest SHA-256 为
`ab2af76133f7791c91613008f6ea997e29d6d26ef3233462fbee8699e5aef80e`。

在统一数据集的独立 test split（108 张记录、103 个检测样本）上，replay-soup
模型相对上一生产模型的结果为：

| 指标 | 上一模型 | 当前模型 |
| --- | ---: | ---: |
| Precision | 0.976794 | 0.957737 |
| Recall | 0.898990 | 0.924984 |
| mAP50 | 0.968342 | 0.981065 |
| mAP50-95 | 0.589861 | 0.606387 |

连续帧数据集 `1Ayoyo_consecutive` 的 856 帧直接 A/B（固定 reviewed yoyo 框、
`conf=0.15`、`IoU=0.7`）显示 pooled Presence F1 为 `0.977387`，Mean IoU
为 `0.800496`，中心误差为 `17.2435 px`；上一生产模型分别为 `0.978056`、
`0.799332` 和 `16.2794 px`。误检由 `10` 降至 `9`，Precision 由 `0.987342`
升至 `0.988564`。最弱的邬聪聪来源组 F1 由 `0.933333` 升至 `0.944444`，
误检由 `10` 降至 `9`，TP/FN 由 `84/2` 变为 `85/1`；2023 华南和 namdongxun
两组召回各有小幅下降（F1 分别 `0.956044 -> 0.944444`、`0.928571 -> 0.923077`）。

在同一环境的邬聪聪 99 帧完整 pipeline 复核中，检测器吞吐为 `7.335 -> 7.179 FPS`，
完整 pipeline 为 `9.292 -> 8.652 FPS`，约下降 `6.9%`；模型结构和参数量不变。

权重 lineage、test 指标和连续集复核分别保留在：

- `runs/experiments/yoyo_detection_replay_20260830_detection_best_replay48x2/run_manifest.json`
- `runs/experiments/det_replay_soup_a25/test_metrics.json`
- `tmp/detection_replay_soup_consecutive_20260830.json`
- `datasets/experiments/detection_replay_20260830_r2/manifest.json`

## 绳线分割模型

当前模型是生产 clDice 模型与 current-manifest centerline 模型的参数 soup，保持
MobileNetV3-Large 编码器和轻量 FPN 不变。centerline 模型训练期使用正样本拓扑损失
权重 `0.05`，soup 以 `alpha=0.50` 插值；推理仍只有一次语义前向，不增加推理头。

当前权重 SHA-256：
`e0328d24303dc76dd86772aae39f95bf0104026792319827ccc81eb72d325c8f`。

在固定阈值 `0.40` 的独立 test（128 张）上，相对生产绳线模型：

| 指标 | 生产模型 | 晋升模型 |
| --- | ---: | ---: |
| Pixel Dice | 0.731826 | 0.731940 |
| Tolerant F1@3 | 0.938868 | 0.938851 |
| Presence F1 | 0.983333 | 0.983333 |
| 负图平均误检像素 | 37.000 | 37.444 |

在 `1Ayoyo_consecutive` 的 856 帧、相同颜色/亮脊/时序协议和阈值 `0.40` 下，pooled
F1@8 从 `0.624796` 提升至 `0.630483`，加权 Chamfer 从 `28.4521` 降至
`27.9382 px`；9 个来源组 F1 均提升。邬聪聪组 F1@8 从 `0.799919` 提升至
`0.816028`，Chamfer 从 `15.4579` 降至 `15.3484 px`。

当前检测器下邬聪聪 99 帧真实 pipeline 的同阈值复核：绳线 F1@8 `0.809101 -> 0.810528`，
Presence F1 和悠悠球 Presence F1 保持不变。三次测速均值为 `7.8445 -> 7.7843 FPS`
（约下降 `0.8%`），模型结构和参数量不变。

复现证据：

- `runs/experiments/semantic_soup_centerline_w005_a050/run_manifest.json`
- `runs/experiments/semantic_soup_centerline_w005_a050/test_semantic_metrics_threshold_0p4.json`
- `runs/experiments/semantic_soup_centerline_w005_a050/promotion_comparison.json`
- `tmp/semantic_soup_centerline_w005_a050_consecutive/summary.json`
- `tmp/semantic_soup_centerline_w005_a050_pipeline_wu99/metrics.json`

## 绳线追踪流程

追踪器使用当前语义模型生成概率图，在语义支持邻域内融合饱和色和亮脊 Hough 候选，
再用组件级概率门控形成多段中心线。新鲜观测优先；仅在缺少新观测时使用
Lucas-Kanade 前后向光流，传播上限为 12 帧，前后向误差上限为 `4 px`。

默认配置：

- `string_confidence=0.40`
- `string_color_probability_augment=true`
- `string_bright_line_augment=true`
- `string_color_semantic_prefilter=true`
- `string_inference_fps=0`（每帧语义推理）

该流程不依赖姿态模型、第二语义模型或场景路由。连续帧评估使用 reviewed
`yoyo` box 隔离检测器误差，绳线指标以源图像像素中心线计算。

## 方向识别模型

模型只读取悠悠球 ROI。训练标签改为画面朝向四分类
`frontal`、`edge_horizontal`、`edge_vertical`、`unknown`，推理时聚合为原有三分类：
`frontal/edge_vertical -> normal`、`edge_horizontal -> horizontal`、`unknown -> not_applicable`。
四分类 ROI 视图 manifest 为
`datasets/1Ayoyo_dataset/orientation_roi/manifest.json`，当前训练集计数为
`frontal=422`、`edge_horizontal=50`、`edge_vertical=2`、`unknown=17`（验证/测试不重采样）。

新结构基线权重为
`runs/experiments/yoyo_unified_57935af9dc69_orientation_roi_405cce7c77e0_yolo11n-cls_presentation4_baseline/weights/best.pt`，
SHA-256：`db2b517aca834996fb2982fc6925168451a08c5da8437dfa012773fb7ab4ef8`。
独立 test（103 张）四分类 Top-1 为 `0.902913`，连续集 856 帧四分类 Accuracy 为
`0.912383`；映射后三分类在 5 FPS + 自适应突发 + EMA/滞回协议下 Accuracy 为
`0.942757`、Macro Recall 为 `0.826033`。

当前默认部署权重仍为下列已验证的三分类模型，直到新结构在所有弱来源组上完成对比：
权重 SHA-256：`f00e3766c05d9ae7dc3fe13a9cd45faf3507aab4c9a9acfa6df73b155ff7cd91`。

在 65 张独立 test 上，Top-1 为 `0.9231`，Macro Recall 为 `0.8732`；三类 recall
分别为 `0.8462`、`0.9552`、`0.8182`。连续集使用概率 EMA `alpha=0.5`、切换 margin
`0.05`、连续 4 次确认和强切换置信度 `0.9`。856 帧离线回放 Accuracy 为 `0.903037`，
Macro Recall 为 `0.864272`。

## 推理性能与验证

当前语义模型与上一 FPN 模型结构、参数量和推理流程相同；clDice 不增加推理开销。
关闭可选 RTMPose 后，主追踪输出不变，审核 pipeline 速度更高。

统一验证命令：

```powershell
.\.venv\Scripts\python.exe -m cli.training.evaluate_semantic `
  --weights runs\experiments\semantic_cldice_w010_r1\weights\best.pt `
  --dataset-dir datasets\1Ayoyo_dataset\string_segmentation `
  --split test --threshold 0.40 --device cuda

.\.venv\Scripts\python.exe -m string_segmentation.evaluate_consecutive `
  --weights runs\experiments\semantic_cldice_w010_r1\weights\best.pt `
  --dataset-dir datasets\1Ayoyo_consecutive `
  --output-dir tmp/semantic_cldice_consecutive `
  --threshold 0.40 --color-augment --color-semantic-prefilter `
  --bright-line-augment --bright-line-min-mean 0.70 `
  --temporal --max-propagation-frames 12 --max-forward-backward-error 4.0

.\.venv\Scripts\python.exe -m pytest -q
```

最近一次完整验证为 `162 passed`；`compileall` 和 `git diff --check` 均通过。
