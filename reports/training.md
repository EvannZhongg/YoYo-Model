# 悠悠球识别与追踪模型报告

## 当前生产方案

| 模块 | 模型与权重 | 运行参数 |
| --- | --- | --- |
| 悠悠球检测 | YOLO11s；`runs/experiments/det_replay_soup_a25/weights/best.pt` | `imgsz=1024`，主检测 `conf=0.15`、`IoU=0.7`；可信轨迹掉检时启用 `conf=0.03`、距离门控 `1.5×` 框对角线的低置信度救援 |
| 绳线分割 | MobileNetV3-FPN；`runs/experiments/semantic_retrain_manifest2689_e12_s20260830/weights/best.pt` | `960x544` checkpoint，推理 `1088x608`（`1.125x`），验证阈值 `0.9453` |
| 绳线追踪 | 语义概率图、颜色/亮脊候选、Lucas-Kanade 光流 | 组件上限 `32`，最多传播 `12` 帧 |
| 方向识别 | 悠悠球 ROI 三分类；`runs/experiments/yoyo_unified_579b879a66ce_orientation_roi_b7fb7417b09a_yolo11n-cls_current579-three-dropout1-e20-b32/weights/best.pt` | 稳态 `5 FPS`，突发 `25 FPS`，EMA 与切换滞回 |
| 姿态审核 | RTMPose-m WholeBody | 按需启用 |

Workbench 和 CLI 从 `config.yaml`、`config.py` 读取默认权重。当前绳线模型的训练
manifest SHA-256 为
`2689b44e3ddf27d0c97fe24cc592287820d240757c43959ad956785a9176d514`；检测权重 SHA-256
为 `2d5a0e45b9da1aa88609c79015ce7b651e86fb8206d9ae6463f0fa72cf4a0e00`，绳线权重
SHA-256 为 `ff9d4c1dd5a50df5ae92ba341c3ef380b85ecfdc76c4c300f24adef300d406f6`，方向
权重 SHA-256 为 `e6d5835bfbfa629d712cd763bb7f4ad5b9bea9711bcd0d1f6c5cd26e06d055ab`。

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
| val | 178 | 0.858789 / 0.812687 / 0.835102 | 0.987952 | 0.690193 |
| test | 179 | 0.894728 / 0.857238 / 0.875582 | 0.982353 | 0.689983 |

连续集使用 10 段、8 个来源的 927 帧，以标注球框作为共同锚点。回放复用生产颜色/亮脊
补线、语义调度和 LK 传播条件。Centerline F1 对已审核的有绳帧按 2 px 间隔累加采样点。
全部已知帧的中心线 F1@8 为 `0.836130`，包括不可见帧上的预测几何。Presence 的
TP/FP/FN/TN 为 `906/2/13/6`。

| 连续集 | Centerline F1@2 / @4 / @8 | F1@8 P / R | Presence F1 | Chamfer / HD95（px） | 最长缺失 / 最大恢复（帧） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全量 927 帧 | 0.418442 / 0.671871 / 0.836205 | 0.907864 / 0.775030 | 0.991790 | 12.3643 / 50.8095 | 4 / 4 |
| 训练来源外的 243 帧 | 0.475965 / 0.745106 / 0.887242 | 0.940430 / 0.839747 | 0.997938 | 10.9422 / 50.1029 | 1 / 1 |

按来源汇总后，最弱组为 `DSCF7145-e237bbfb5d`，F1@8 为 `0.788475`。连续集含 8 帧
已审核不可见绳线，训练来源外的 243 帧均有绳线。静态 test 的 12 张负图中有 6 张误检，
平均每张负图预测 `106.17` 个前景像素。几何距离均值只统计同时有目标和预测的帧，缺失
由 Presence 与时序指标计入。

当前观测线段数小于上一帧轨迹时才运行 LK。与逐帧 LK 配对回放中，927 帧的输出几何完全
一致，有输出记录的光流尝试从 857 帧减至 349 帧，21 帧次级绳段补全全部保留。
在 RTX 4070 Laptop 上对唐浩翔同一 100 帧片段按 AB/BA 顺序复测，包含解码、检测、
方向模型和视频输出的追踪循环吞吐由 `2.67` 提至 `3.17 FPS`。两轮分别为
`2.36 → 3.15` 和 `3.08 → 3.19 FPS`，测试期间有同机训练负载。

### 方向识别

当前三分类 ROI 派生视图（manifest SHA-256
`a2c287b4a5061918659b8afb72d6e5f9358222f96a257252352840ede95dc237`）179 张 test 上，
当前模型 Top-1 为 `0.932961`，Macro Recall 为 `0.896621`，三类召回分别为
`horizontal=0.736842`、`normal=0.953020`、`not_applicable=1.000000`。在
`1Ayoyo_consecutive` 927 帧回放中，稳态/突发时序 Accuracy 为 `0.979504`、Macro
Recall 为 `0.897801`，预测切换数为 `6`；最弱来源“邬聪聪” Accuracy 为 `0.909091`，
各来源组不低于上一生产方向模型。同集外部 test 相比上一生产权重的 Top-1 / Macro Recall
为 `0.932961 / 0.896621` 对 `0.927374 / 0.884172`；其中 `horizontal` 单类召回由
`0.894737`（`17/19`）降至 `0.736842`（`14/19`），回退量为 3 张，换得 `normal`
`0.939597`→`0.953020`、`not_applicable` `0.818182`→`1.000000` 和连续集时序指标提升。
连续集推理共 `158` 次、原始批量推理耗时 `3.2883 s`（同一协议旧权重 `175` 次、
`8.7957 s`）。

## 复现入口

- 检测运行：`runs/experiments/yoyo_detection_replay_20260830_detection_best_replay48x2/run_manifest.json`
- 检测 test：`runs/experiments/det_replay_soup_a25/test_metrics_external_90f330d7516d.json`
- 检测连续集评估：`tmp/det_production_consecutive_grouped_metrics.json`
- 检测时序救援完整连续集评估：`tmp/full_rescue_consecutive/summary.json`
- 连续集评估入口：`cli/tracking/evaluate_sequence.py`
- 绳线训练：`runs/experiments/semantic_retrain_manifest2689_e12_s20260830/run_manifest.json`
- 绳线静态评估：`runs/experiments/semantic_retrain_manifest2689_e12_s20260830/test_semantic_metrics.json`
- 绳线连续集评估：`tmp/string_resolution_20261114/continuous_optimized_flow/pooled.json`
- 绳线光流等价检查：`tmp/string_resolution_20261114/flow_geometry_equivalence.json`
- 绳线完整追踪吞吐：`tmp/string_resolution_20261114/full_tracking_fps.json`
- 绳线吞吐配对：`tmp/semantic_retrain_manifest2689_benchmark.json`、`tmp/semantic_retrain_manifest2689_benchmark_r2.json`
- 方向训练：`runs/experiments/yoyo_unified_579b879a66ce_orientation_roi_b7fb7417b09a_yolo11n-cls_current579-three-dropout1-e20-b32/run_manifest.json`
- 方向三分类 ROI 视图：`datasets/1Ayoyo_dataset/orientation_roi_three_current/manifest.json`
- 方向 test：`runs/experiments/yoyo_unified_579b879a66ce_orientation_roi_b7fb7417b09a_yolo11n-cls_current579-three-dropout1-e20-b32/test_metrics.json`
- 方向连续集评估：`runs/experiments/orientation_current579_three_dropout1_consecutive/metrics.json`
- 方向评估入口：`cli/tracking/evaluate_orientation.py`

统一测试命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
