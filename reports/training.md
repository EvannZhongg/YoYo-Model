# 悠悠球识别与追踪模型报告

## 当前默认方案

| 模块 | 当前模型与权重 | 默认输入/参数 |
| --- | --- | --- |
| 悠悠球检测 | YOLO11s，约 9.4M 参数；`runs/experiments/det_replay_soup_a25/weights/best.pt` | `imgsz=1024`，置信度 `0.15`，IoU `0.7` |
| 绳线分割 | MobileNetV3-FPN，约 3.0M 参数；`runs/experiments/semantic_ablation_nomorph_foundation_r1/weights/best.pt` | `960x544`，验证阈值 `0.9204` |
| 绳线追踪 | 单语义模型 + 颜色/亮脊候选 + 按需光流 | 每帧语义推理，最多传播 12 帧 |
| 方向识别 | 悠悠球 ROI 三分类模型；`runs/candidates/yoyo_unified_2b0cfca8743a_orientation_roi_9cd9d9361ab5_best_yoyo-only-final-warm-freeze10-lr1e4-v1/weights/best.pt` | 5 FPS 稳态、25 FPS 突发，EMA 与滞回 |
| 姿态审核 | RTMPose-m WholeBody（可选） | 默认关闭，不参与主输出 |

Workbench 和 CLI 都从 `config.yaml`、`config.py` 读取以上默认值。训练数据身份由
`datasets/1Ayoyo_dataset/string_segmentation/manifest.json` 固定，SHA-256 为
`f79c9805dae3c91df2ad49eb61f96db31a3236291e505c0925e3aad31f307964`。

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

当前模型使用 MobileNetV3-Large 编码器和轻量 FPN。训练损失由 Focal 与 Dice
组成，并对 reviewed 空 mask 样本额外加入 `0.2 × hard-negative` 项；训练采样器
对空 mask 样本使用 `negative sampling ×4`（仅影响 train loader，不改变 val/test）。
不对标注 mask 做膨胀；推理仍只有一次语义前向，不增加推理头。

当前权重 SHA-256：
`5bd3b22175317cc09ff0e160888643b856213944fb008f05a7da0e9ec2de7dc4`。
训练 manifest SHA-256 为
`f79c9805dae3c91df2ad49eb61f96db31a3236291e505c0925e3aad31f307964`，验证选择的
冻结阈值为 `0.9204`。

| split | 样本数 | Pixel Dice | Tolerant F1@3 | Presence F1 | 负图平均误检像素 |
| --- | ---: | ---: | ---: | ---: | ---: |
| test | 136 | 0.694979 | 0.927502 | 0.976562 | 33.500 |

固定验证阈值 `0.9204` 后，覆盖式骨架中心线在 val/test 上的 centerline F1@8 分别为
`0.739734` 和 `0.767794`；test precision/recall 为 `0.784026/0.752220`。

训练阶段的 checkpoint 与阈值选择统一使用
`pooled_centerline_f1_at_8_source_px`：由 mask 骨架化后映射到源图像坐标，采用与连续集相同的
中心线采样和最近距离计算，并以 Centerline F1@8 为首要排序指标；两者差值不超过 `0.005`
时以 Presence F1 决胜。上表的 Pixel Dice 与
Tolerant F1@3 继续作为静态语义诊断指标。

在最新 `1Ayoyo_consecutive`（manifest SHA-256
`2065E8C684DF3594CA1010AD4B95D3245F6B592A7E5A2670038DE07D66B56AF7`）的 927 帧、10 个 group（固定 reviewed yoyo 框、阈值
`0.9204`、颜色/亮脊增强、语义预筛、颜色候选概率均值门槛 `0.70` 和时序协议）上，pooled centerline F1@8 为
`0.766228`（precision `0.880844`、recall `0.678006`），按 pair frame 加权的
Chamfer 为 `14.4312 px`，HD95 为 `60.9098 px`。相对单主路径抽取的 F1@8
`0.564291` 和 recall `0.414447`，覆盖式骨架抽取保留同一语义连通域中的主要分支，
同时维持接近的 precision。pooled Presence F1 为 `0.991772`，零预测帧为 `18`；
最长缺失段和最大恢复延迟均为 `4` 帧。最弱 group 为 `池高宇-fef6c7bcb0`
（F1@8 `0.612640`）。评估器默认门槛与追踪器配置统一为 `0.70`。

晋升判定以连续集 pooled centerline F1@8 为主指标，同时报告最弱来源组并设置回退护栏。最长缺失段/恢复延迟和FPS 作为安全与部署门槛，Chamfer/HD95 作为几何诊断。
Pixel Dice 会随标注线宽和缓冲规则变化，不作为主排名指标；单一阈值必须先在 val 校准后
冻结，不能用 test 重新选阈值。负图平均误检像素、平均组件数、长度比仅作诊断，不能单独
决定晋升。`max_components=8`、`min_component_pixels=8`、`max_polyline_points=64`、
传播上限 `12` 和前后向误差 `4.0 px` 是固定运行参数，候选比较时保持不变。
标注中的 `string_visibility` 仅用于连续帧
评估时区分有绳、无绳和未知帧；应保留它以避免把无绳负样本混入几何指标，但不把该字段
本身作为模型质量或晋升排名指标。

同机邬聪聪视频 300 帧端到端复核（检测、语义、方向和异步写盘配置一致）吞吐为
`15.7206 -> 15.2890 FPS`，下降约 `2.7%`，无额外推理组件。

复现证据：

- `runs/experiments/semantic_ablation_nomorph_foundation_r1/run_manifest.json`
- `tmp/semantic_skeleton_cover_test_t092/test_semantic_metrics_threshold_0p9204.json`
- `tmp/semantic_skeleton_cover_val_t092/val_semantic_metrics_threshold_0p9204.json`
- `tmp/semantic_skeleton_cover_optimized/summary.json`
- `tmp/fps_skeleton_cover_optimized/邬聪聪_20260831T174144Z_2f6fbd4e/run.json`

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

模型只读取悠悠球 ROI，输出 `horizontal`、`normal`、`not_applicable` 三类。
当前默认权重为：
权重 SHA-256：`f00e3766c05d9ae7dc3fe13a9cd45faf3507aab4c9a9acfa6df73b155ff7cd91`。

在 65 张独立 test 上，Top-1 为 `0.9231`，Macro Recall 为 `0.8732`；三类 recall
分别为 `0.8462`、`0.9552`、`0.8182`。连续集使用概率 EMA `alpha=0.5`、切换 margin
`0.05`、连续 4 次确认和强切换置信度 `0.9`。856 帧离线回放 Accuracy 为 `0.903037`，
Macro Recall 为 `0.864272`。

## 推理性能与验证

当前语义模型使用单次 MobileNetV3-FPN 前向。
关闭可选 RTMPose 后，主追踪输出不变，审核 pipeline 速度更高。

统一验证命令：

```powershell
.\.venv\Scripts\python.exe -m cli.training.evaluate_semantic `
  --weights runs\experiments\semantic_ablation_nomorph_foundation_r1\weights\best.pt `
  --dataset-dir datasets\1Ayoyo_dataset\string_segmentation `
  --split test --threshold 0.9204 `
  --min-component-pixels 8 --device cuda `
  --output-dir tmp\semantic_production_latest

.\.venv\Scripts\python.exe -m string_segmentation.evaluate_consecutive `
  --weights runs\experiments\semantic_ablation_nomorph_foundation_r1\weights\best.pt `
  --dataset-dir datasets\1Ayoyo_consecutive `
  --output-dir tmp\semantic_production_latest_consecutive `
  --threshold 0.9204 --color-augment --color-semantic-prefilter `
  --color-probability-min-mean 0.70 `
  --bright-line-augment --bright-line-min-mean 0.70 `
  --temporal --max-propagation-frames 12 --max-forward-backward-error 4.0 `
  --max-polyline-points 64 --max-components 8 --min-component-pixels 8 --device cuda

.\.venv\Scripts\python.exe -m pytest -q
```

最近一次完整验证为 `164 passed`；`compileall` 和 `git diff --check` 均通过。
