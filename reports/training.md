# 悠悠球识别与追踪模型报告

## 当前生产方案

| 模块 | 模型与权重 | 运行参数 |
| --- | --- | --- |
| 悠悠球检测 | YOLO11s；`runs/experiments/det_replay_soup_a25/weights/best.pt` | `imgsz=1024`，`conf=0.15`，`IoU=0.7` |
| 绳线分割 | MobileNetV3-FPN；`runs/experiments/semantic_ablation_nomorph_foundation_r1/weights/best.pt` | `960x544`，验证阈值 `0.9204` |
| 绳线追踪 | 语义概率图、颜色/亮脊候选、Lucas-Kanade 光流 | 组件上限 `32`，最多传播 `12` 帧 |
| 方向识别 | 悠悠球 ROI 三分类；`runs/candidates/yoyo_unified_2b0cfca8743a_orientation_roi_9cd9d9361ab5_best_yoyo-only-final-warm-freeze10-lr1e4-v1/weights/best.pt` | 稳态 `5 FPS`，突发 `25 FPS`，EMA 与切换滞回 |
| 姿态审核 | RTMPose-m WholeBody | 按需启用 |

Workbench 和 CLI 从 `config.yaml`、`config.py` 读取默认权重。当前绳线模型的训练
manifest SHA-256 为
`f79c9805dae3c91df2ad49eb61f96db31a3236291e505c0925e3aad31f307964`；检测权重 SHA-256
为 `2d5a0e45b9da1aa88609c79015ce7b651e86fb8206d9ae6463f0fa72cf4a0e00`，绳线权重
SHA-256 为 `5bd3b22175317cc09ff0e160888643b856213944fb008f05a7da0e9ec2de7dc4`，方向
权重 SHA-256 为 `f00e3766c05d9ae7dc3fe13a9cd45faf3507aab4c9a9acfa6df73b155ff7cd91`。

## 性能对比

### 悠悠球检测

| 指标 | 上一版 | 当前生产 |
| --- | ---: | ---: |
| Precision | 0.976794 | 0.957737 |
| Recall | 0.898990 | 0.924984 |
| mAP50 | 0.968342 | 0.981065 |
| mAP50-95 | 0.589861 | 0.606387 |
| 连续集 Presence F1 | 0.978056 | 0.977387 |
| 连续集 Mean IoU | 0.799332 | 0.800496 |
| 连续集中心误差（px） | 16.2794 | 17.2435 |
| 邬聪聪来源组 F1 | 0.933333 | 0.944444 |
| 完整 pipeline FPS | 9.292 | 8.652 |

连续集评估使用 `1Ayoyo_consecutive`、固定 reviewed yoyo 框、`conf=0.15` 和 `IoU=0.7`。

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
| 当前生产 `max_components=32` | 0.807238 | 0.991772 | 12.7498 / 54.2854 | 4 / 4 |

当前连续集最弱来源组为 `池高宇-fef6c7bcb0`，Centerline F1@8 为 `0.615901`；300 帧
端到端吞吐为 `14.6545 FPS`。

### 方向识别

当前模型在原训练 manifest 的 65 张 native test 上 Top-1 为 `0.9231`，Macro Recall
为 `0.8818`；在 91 张扩展 test 上 Top-1 为 `0.9231`，Macro Recall 为 `0.8732`。
三类模型输出为 `horizontal`、`normal`、`not_applicable`。连续集离线回放 Accuracy
为 `0.903037`，Macro Recall 为 `0.864272`。

## 复现入口

- 检测运行：`runs/experiments/yoyo_detection_replay_20260830_detection_best_replay48x2/run_manifest.json`
- 检测 test：`runs/experiments/det_replay_soup_a25/test_metrics.json`
- 绳线训练：`runs/experiments/semantic_ablation_nomorph_foundation_r1/run_manifest.json`
- 绳线静态评估：`tmp/semantic_production_aligned/test_semantic_metrics.json`
- 绳线连续集评估：`tmp/production_comp32_full/summary.json`
- 方向评估：`cli/tracking/evaluate_orientation.py`

统一测试命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
