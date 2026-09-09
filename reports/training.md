# 悠悠球识别与追踪模型报告

## 当前生产方案

| 模块 | 模型与权重 | 运行参数 |
| --- | --- | --- |
| 悠悠球检测 | YOLO11s；`runs/experiments/det_replay_soup_a25/weights/best.pt` | `imgsz=1024`，主检测 `conf=0.15`、`IoU=0.7`；可信轨迹掉检时启用 `conf=0.03`、距离门控 `1.5×` 框对角线的低置信度救援 |
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
| Precision | 0.919206 |
| Recall | 0.767857 |
| mAP50 | 0.891418 |
| mAP50-95 | 0.586875 |
| 连续集 Presence P / R / F1 | 0.984962 / 0.976398 / 0.980661 |
| 连续集 Mean / Median IoU | 0.802578 / 0.846517 |
| 连续集 IoU@0.50 命中率 | 0.955013 |
| 连续集中心误差（px） | 16.5795 |
| 最弱有效来源组 F1 | 0.923077（邬聪聪） |
| 最长缺失段（帧） | 6 |
| 最大恢复延迟（帧） | 6 |
| 检测回放 FPS | 10.63 |

静态结果为扩张后 `90f330d7516d...` manifest 的 179 张 test 显式跨 manifest 复评；checkpoint
未使用该 test 来源。连续集使用 `1Ayoyo_consecutive` 10 组、927 帧，其中排除 107 帧未知
yoyo 标注；按 `(source_group, frame_index)` 对齐预测，`conf=0.15`、`IoU=0.7`，关闭姿态、
绳模型和方向模型。完整回放覆盖 10 个来源组、927 帧，其中 820 帧标签状态可用于悠悠球
presence 评估，107 帧未知状态排除；按 `(source_group, frame_index)` 对齐预测，主检测
`conf=0.15`、`IoU=0.7`，可信轨迹掉检时启用 `conf=0.03` 的 `1.5×` 对角线空间救援。
pooled 统计由 `tmp/full_rescue_consecutive/summary.json` 生成；Jakub 来源组当前 71 帧
均为 `needs_review`，不计入已知指标。最弱有效来源组为 `邬聪聪-0d26cf65b6`（F1 `0.923077`）。

追踪器新增的低置信度救援在 `namdongxun-72f4a04fb5` 的 `3121–3225` 片段（105 帧、
60 FPS）上做了真实视频回放复核：该片段 presence P/R/F1 从 `1.0000/0.8667/0.9286`
提升到 `1.0000/1.0000/1.0000`，FN 从 14 降为 0，FP 保持 0，最长缺失段从 6 帧降为
0；mean IoU 为 `0.7733`，IoU@0.50 命中率 `0.9451`。检测专项回放（关闭绳线、方向和
姿态模型）循环吞吐约 `23.47 FPS`。该结果是局部片段安全复核，完整 10 组 pooled 指标
仍沿用上表生产基线，待下一次全量回放后再更新。

### 绳线分割与追踪

静态 test/val 结果（扩张后 string manifest SHA-256
`2689b44e3ddf27d0c97fe24cc592287820d240757c43959ad956785a9176d514`）：

| split | 样本数 | Centerline P / R / F1@8 | Presence F1 | Pixel Dice |
| --- | ---: | ---: | ---: | ---: |
| val | 178 | 0.779350 / 0.796954 / 0.788054 | 0.990937 | 0.584041 |
| test | 179 | 0.796149 / 0.831904 / 0.813634 | 0.979472 | 0.613023 |

连续集当前生产配置与上一版组件上限的对比：

| 配置 | Centerline F1@8 | Presence F1 | Chamfer / HD95（px） | 最长缺失 / 最大恢复（帧） |
| --- | ---: | ---: | ---: | ---: |
| 上一版 `max_components=8` | 0.766228 | 0.991772 | 14.4312 / 60.9098 | 4 / 4 |
| 当前生产 `max_components=32`，`1.125x` 推理 | 0.818297 | 0.994530 | 15.6212 / 57.4703 | 2 / 2 |

当前连续集最弱来源组为 `池高宇-fef6c7bcb0`，Centerline F1@8 为 `0.638996`；300 帧
端到端吞吐为约 `11.14 FPS`（两次 `1.125x` 配对均值），相对 `1.0x` 配对均值约下降 `6.7%`。

### 方向识别

扩张后统一 manifest 的三分类 ROI 派生视图（manifest SHA-256
`951edb508420fb3e7d54a9b0ee4a1ad400867a28dba3f4d02155c4b2421f0c62`）179 张 test 上，
当前模型 Top-1 为 `0.927374`，Macro Recall 为 `0.884172`，三类召回分别为
`horizontal=0.894737`、`normal=0.939597`、`not_applicable=0.818182`。在
`1Ayoyo_consecutive` 927 帧回放中，稳态/突发时序 Accuracy 为 `0.960086`、Macro
Recall 为 `0.863079`，预测切换数为 `10`；同协议旧生产权重为 `0.908306 / 0.859252 / 13`。
同一 RTX 4070 上 152 张 ROI 配对推理约 `431 FPS`，旧权重约 `456 FPS`，吞吐下降约 `5.5%`。

## 复现入口

- 检测运行：`runs/experiments/yoyo_detection_replay_20260830_detection_best_replay48x2/run_manifest.json`
- 检测 test：`runs/experiments/det_replay_soup_a25/test_metrics_external_90f330d7516d.json`
- 检测连续集评估：`tmp/det_production_consecutive_grouped_metrics.json`
- 检测时序救援完整连续集评估：`tmp/full_rescue_consecutive/summary.json`
- 连续集评估入口：`cli/tracking/evaluate_sequence.py`
- 绳线训练：`runs/experiments/semantic_ablation_nomorph_foundation_r1/run_manifest.json`
- 绳线静态评估：`tmp/production_test_compare/test_semantic_metrics_external_2689b44e3ddf.json`
- 绳线连续集评估：`tmp/production_comp32_full/summary.json`
- 方向训练：`runs/experiments/yoyo_unified_5673a7faf873_orientation_roi_afbae9c0cd2a_yolo11n-cls_current5673-foundation-e30-b32/run_manifest.json`
- 方向三分类 ROI 视图：`datasets/1Ayoyo_dataset/orientation_roi_three_new/manifest.json`
- 方向 test：`runs/experiments/yoyo_unified_5673a7faf873_orientation_roi_afbae9c0cd2a_yolo11n-cls_current5673-foundation-e30-b32/test_metrics_external_951edb508420.json`
- 方向连续集评估：`runs/experiments/orientation_current5673_foundation_consecutive/metrics.json`
- 方向评估入口：`cli/tracking/evaluate_orientation.py`

统一测试命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
