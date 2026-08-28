# 悠悠球识别与追踪模型报告

## 当前默认方案

| 模块 | 当前模型与权重 | 默认输入/参数 |
| --- | --- | --- |
| 悠悠球检测 | YOLO11s，约 9.4M 参数；`runs/experiments/det_soup_new116_a0p75/weights/best.pt` | `imgsz=1024`，置信度 `0.15`，IoU `0.7` |
| 绳线分割 | MobileNetV3-FPN，约 3.0M 参数；`runs/experiments/semantic_cldice_w010_r1/weights/best.pt` | `960x544`，配置阈值 `0.40` |
| 绳线追踪 | 单语义模型 + 颜色/亮脊候选 + 按需光流 | 每帧语义推理，最多传播 12 帧 |
| 方向识别 | 悠悠球 ROI 三分类模型；`runs/candidates/yoyo_unified_2b0cfca8743a_orientation_roi_9cd9d9361ab5_best_yoyo-only-final-warm-freeze10-lr1e4-v1/weights/best.pt` | 5 FPS 稳态、25 FPS 突发，EMA 与滞回 |
| 姿态审核 | RTMPose-m WholeBody（可选） | 默认关闭，不参与主输出 |

Workbench 和 CLI 都从 `config.yaml`、`config.py` 读取以上默认值。训练数据身份由
`datasets/1Ayoyo_dataset/string_segmentation/manifest.json` 固定，SHA-256 为
`7f661d3a5cfd3a3f8ae1cf2576192097c0a44643d75493cd3847b1397d3a5a7c`。

## 悠悠球检测模型

当前权重 SHA-256：
`13b66a5c6eda03deb7f5bc4b1efc60a273df5aae8d314424a79bf2fb5b7029b8`。

在 722 张统一数据集的新 test split 上，当前模型相对上一生产模型的结果为：

| 指标 | 上一模型 | 当前模型 |
| --- | ---: | ---: |
| Precision | 0.987401 | 0.976794 |
| Recall | 0.898990 | 0.898990 |
| mAP50 | 0.959212 | 0.968342 |
| mAP50-95 | 0.574515 | 0.589861 |

在未参与微调的 `1Ayoyo_consecutive` 856 帧上，Presence F1 为 `0.976280`，Mean IoU
为 `0.792355`，中心误差为 `20.7396 px`；上一生产模型分别为 `0.958678`、
`0.771622` 和 `33.0983 px`。1024 输入单帧 GPU 推理约 `7.5 ms`。

权重 lineage、test 指标和连续集复核分别保留在：

- `runs/experiments/det_soup_new116_a0p75/run_manifest.json`
- `runs/experiments/det_soup_new116_a0p75/test_metrics.json`
- `tmp/det_soup_a75_consecutive`

## 绳线分割模型

当前模型使用 MobileNetV3-Large 编码器和轻量 FPN，训练期加入正样本 soft-clDice
拓扑损失（权重 `0.10`、迭代 `5`），同时使用 hard-negative loss 权重 `0.20`、
`lr=5e-6`，训练 3 epoch。clDice 只参与训练，不增加推理参数或前向次数。

当前权重 SHA-256：
`8ab93a877b15ece5f5e4ebdd9cfed5ef405f56004d0fae9c77d4e6b6d22614e1`。

在固定阈值 `0.40` 的独立 test（108 张）上，相对上一生产绳线模型：

| 指标 | 上一模型 | 当前模型 |
| --- | ---: | ---: |
| Pixel Dice | 0.739187 | 0.741140 |
| Tolerant F1@3 | 0.945287 | 0.947523 |
| Presence F1 | 0.980198 | 0.985075 |
| 负图平均误检像素 | 40.111 | 37.000 |

在 `1Ayoyo_consecutive` 的 856 帧、相同颜色/亮脊/时序协议和阈值 `0.40` 下，帧加权
F1@8 从 `0.654365` 提升至 `0.660675`，Chamfer 从 `34.7267` 降至 `28.4521 px`。
9 个来源组中 6 组 F1 提升；邬聪聪组 Chamfer 从 `62.0578` 降至 `15.4579 px`。
局部 F1 最大回退为 `0.0151`，其余关键弱组保持或提升。

邬聪聪 99 帧真实 pipeline 的同阈值复核：绳线 Presence F1 `0.940541 -> 0.944444`，
F1@8 `0.824923 -> 0.823792`，Chamfer `100.0474 -> 60.3611 px`；悠悠球和方向结果
逐帧一致。完整 pipeline loop 为 `7.7217 -> 8.4066 FPS`，模型结构和推理头不变。

复现证据：

- `runs/experiments/semantic_cldice_w010_r1/run_manifest.json`
- `runs/experiments/semantic_cldice_w010_r1_test_t0p40/test_semantic_metrics_threshold_0p4.json`
- `runs/experiments/semantic_cldice_w010_r1/consecutive856_t0p40/summary.json`
- `runs/experiments/semantic_cldice_w010_r1_pipeline_wu99_t0p40/metrics.json`

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

模型只读取悠悠球 ROI，类别为 `horizontal`、`normal`、`not_applicable`，不训练具名招式。
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

最近一次完整验证为 `153 passed`；`compileall` 和 `git diff --check` 均通过。
