# 悠悠球识别与追踪模型报告

## 当前生产方案

| 模块 | 模型与权重 | 运行参数 |
| --- | --- | --- |
| 悠悠球检测 | YOLO11s；`runs/experiments/det_replay_soup_a25/weights/best.pt` | `imgsz=1024`，`conf=0.15`，`IoU=0.7` |
| 绳线分割 | MobileNetV3-FPN；`runs/experiments/semantic_ablation_nomorph_foundation_r1/weights/best.pt` | `960x544` checkpoint，推理 `1088x608`（`1.125x`），验证阈值 `0.9204` |
| 绳线追踪 | 语义概率图、颜色/亮脊候选、Lucas-Kanade 光流 | 组件上限 `32`，最多传播 `12` 帧 |
| 方向识别 | 悠悠球 ROI 三分类；`runs/experiments/yoyo_unified_5673a7faf873_orientation_roi_afbae9c0cd2a_yolo11n-cls_current5673-foundation-e30-b32/weights/best.pt` | 稳态 `5 FPS`，突发 `25 FPS`，EMA 与切换滞回 |
| 姿态审核 | RTMPose-m WholeBody | 按需启用 |

Workbench 和 CLI 从 `config.yaml`、`config.py` 读取默认权重。当前绳线模型的训练
manifest SHA-256 为
`f79c9805dae3c91df2ad49eb61f96db31a3236291e505c0925e3aad31f307964`；检测权重 SHA-256
为 `2d5a0e45b9da1aa88609c79015ce7b651e86fb8206d9ae6463f0fa72cf4a0e00`，绳线权重
SHA-256 为 `5bd3b22175317cc09ff0e160888643b856213944fb008f05a7da0e9ec2de7dc4`，方向
权重 SHA-256 为 `56767a96d3d2687b991f161c1318896f9543ca2044eb7f1688e6fd5447bbaf99`。

## 性能对比

### 悠悠球检测

| 指标 | 当前生产 |
| --- | ---: |
| Precision | 0.926192 |
| Recall | 0.804196 |
| mAP50 | 0.906085 |
| mAP50-95 | 0.597246 |
| 连续集 Presence P / R / F1 | 0.987310 / 0.966460 / 0.976773 |
| 连续集 Mean / Median IoU | 0.805805 / 0.847140 |
| 连续集 IoU@0.50 命中率 | 0.955013 |
| 连续集中心误差（px） | 16.5646 |
| 最弱有效来源组 F1 | 0.928571 |
| 最长缺失段（帧） | 6 |
| 检测回放 FPS | 8.0608 |

静态结果为当前 `60b34d7d7db3...` manifest 的 152 张 test 显式跨 manifest 复评；checkpoint
未使用该 test 来源。连续集使用 `1Ayoyo_consecutive` 10 组、927 帧，其中排除 107 帧未知
yoyo 标注；按 `(source_group, frame_index)` 对齐预测，`conf=0.15`、`IoU=0.7`，关闭姿态、
绳模型和方向模型。最弱有效来源组为 `namdongxun-72f4a04fb5`；FPS 为 10 段共 927 帧
除以累计追踪墙钟时间。

### 绳线分割与追踪

静态 test 结果（manifest SHA-256
`f79c9805dae3c91df2ad49eb61f96db31a3236291e505c0925e3aad31f307964`）：

| split | 样本数 | Centerline P / R / F1@8 | Presence F1 | Pixel Dice |
| --- | ---: | ---: | ---: | ---: |
| val | 136 | 0.850034 / 0.779066 / 0.813005 | 0.987952 | 0.651010 |
| test | 136 | 0.860528 / 0.800567 / 0.829465 | 0.976562 | 0.694979 |

连续集当前生产配置与上一版组件上限的对比：

| 配置 | Centerline F1@8 | Presence F1 | Chamfer / HD95（px） | 最长缺失 / 最大恢复（帧） |
| --- | ---: | ---: | ---: | ---: |
| 上一版 `max_components=8` | 0.766228 | 0.991772 | 14.4312 / 60.9098 | 4 / 4 |
| 当前生产 `max_components=32`，`1.125x` 推理 | 0.818297 | 0.994530 | 15.6212 / 57.4703 | 2 / 2 |

当前连续集最弱来源组为 `池高宇-fef6c7bcb0`，Centerline F1@8 为 `0.638996`；300 帧
端到端吞吐为约 `11.14 FPS`（两次 `1.125x` 配对均值），相对 `1.0x` 配对均值约下降 `6.7%`。

### 方向识别

当前模型在 `yoyo_unified_5673a7faf873` manifest 的 152 张 native test 上 Top-1 为
`0.940789`，Macro Recall 为 `0.956780`，三类召回分别为
`horizontal=0.933333`、`normal=0.937008`、`not_applicable=1.000000`。在
`1Ayoyo_consecutive` 927 帧回放中，稳态/突发时序 Accuracy 为 `0.960086`、Macro
Recall 为 `0.863079`，预测切换数为 `10`；同协议旧生产权重为 `0.908306 / 0.859252 / 13`。
同一 RTX 4070 上 152 张 ROI 配对推理约 `431 FPS`，旧权重约 `456 FPS`，吞吐下降约 `5.5%`。

## 复现入口

- 检测运行：`runs/experiments/yoyo_detection_replay_20260830_detection_best_replay48x2/run_manifest.json`
- 检测 test：`runs/experiments/det_replay_soup_a25/test_metrics_external_60b34d7d7db3.json`
- 检测连续集评估：`tmp/det_production_consecutive_grouped_metrics.json`
- 连续集评估入口：`cli/tracking/evaluate_sequence.py`
- 绳线训练：`runs/experiments/semantic_ablation_nomorph_foundation_r1/run_manifest.json`
- 绳线静态评估：`tmp/semantic_production_aligned/test_semantic_metrics.json`
- 绳线连续集评估：`tmp/production_comp32_full/summary.json`
- 方向训练：`runs/experiments/yoyo_unified_5673a7faf873_orientation_roi_afbae9c0cd2a_yolo11n-cls_current5673-foundation-e30-b32/run_manifest.json`
- 方向 test：`runs/experiments/yoyo_unified_5673a7faf873_orientation_roi_afbae9c0cd2a_yolo11n-cls_current5673-foundation-e30-b32/test_metrics.json`
- 方向连续集评估：`runs/experiments/orientation_current5673_foundation_consecutive/metrics.json`
- 方向评估入口：`cli/tracking/evaluate_orientation.py`

统一测试命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
