# 悠悠球模型训练与迁移报告：2026-08-06

## 当前结论

- 训练代码统一迁移到 `training_v3`；`training_v2` 已移除。
- RTMPose-m WholeBody 仅用于运行时姿态推理，不向任何数据集写入手部或 pose 数据。
- 方向模型改为只使用悠悠球图像区域，不依赖手、手腕或绳线几何。
- 正式数据集名称保持为 `datasets/1Ayoyo_dataset`；连续集保持为 `datasets/1Ayoyo_consecutive`，没有创建或保留 `_v3` 数据集。

## 数据集清理

清理仅删除以下 pose 相关键：`hands`、`hands_pixel`、`hands_2d`、`hands_normalized`、`hand_landmarks_pixel`、`hand_pose`、`pose`、`pose_person`。

| 数据集 | 标签数 | 删除 `hands_pixel` | 删除 `hands_2d` | 其他字段变化 |
| --- | ---: | ---: | ---: | ---: |
| `1Ayoyo_dataset` | 460 | 460 | 460 | 0 |
| `1Ayoyo_consecutive` | 454 | 454 | 454 | 0 |

机器可读记录位于 `reports/pose_annotation_cleanup.json`。清理脚本为 `training_v3.strip_pose_annotations`，其余悠悠球、绳线、方向和序列字段保持不变。

## 检测与绳线任务核验

清理前后的任务视图按路径和 SHA-256 比较：

| 任务 | 比较文件数 | 路径差异 | 内容哈希差异 |
| --- | ---: | ---: | ---: |
| 悠悠球检测 | 920 | 0 | 0 |
| 绳线分割 | 920 | 0 | 0 |

因此检测与绳线训练从未消费手部字段，无需因本次清理重新训练。生产检测与绳线权重保持不变。

## 方向模型 A/B

最终视图位于 `datasets/1Ayoyo_dataset/orientation_roi`，view id 为 `orientation_roi_9cd9d9361ab5`。训练/验证/测试分别为 810/65/65 张分类图像，且 manifest 明确声明不依赖手部和绳线输入。

| 方案 | Top-1 | Macro recall | horizontal | normal | not_applicable |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧上下文 ROI + 旧模型 | 0.8000 | 0.6562 | 0.5000 | 0.9130 | 0.5556 |
| 仅悠悠球 ROI + 旧模型零样本 | 0.7846 | 0.7048 | 0.6000 | 0.8478 | 0.6667 |
| 仅悠悠球 ROI + 最终模型 | **0.9231** | **0.8818** | **0.8000** | **0.9565** | **0.8889** |

最终权重：

`runs/candidates/yoyo_unified_2b0cfca8743a_orientation_roi_9cd9d9361ab5_best_yoyo-only-final-warm-freeze10-lr1e4-v1/weights/best.pt`

权重 SHA-256：`f00e3766c05d9ae7dc3fe13a9cd45faf3507aab4c9a9acfa6df73b155ff7cd91`。正式 test 结果位于同一 run 的 `test_metrics.json`，数据集 manifest 与 checkpoint 完全匹配。

## RTMPose 运行时评估

30 帧连续视频 smoke test 使用项目内 RTMPose-m WholeBody 与 YOLOX-m ONNX 模型：

- pose 成功：30/30 帧。
- 双手完整 21 点：60/60 组。
- 完整 pipeline：1.7081 FPS，17.5632 秒。
- 旧 YOLO Pose 对照：2.0642 FPS，14.5332 秒。

RTMPose 约慢 17%，但提供 133 点 WholeBody 输出；可视审核确认选中了前景表演者。所有下载模型均位于 `models/rtmpose`，C 盘 RTMLib checkpoint 缓存为空。

切换最终方向权重并统一运行时裁剪后，又执行了 30 帧完整 smoke：pose 30/30 帧成功、双手 30/30 帧完整、方向模型按 cadence 推理 3 次，所有方向记录均使用 `yoyo_bbox_square_3p0_min_12pct; no_yoyo_center_square_28pct`。本次完整 pipeline 为 1.5526 FPS，产物位于 `runs/experiments/rtmpose_yoyo_only_orientation_final_smoke`。

## 验证状态

- `compileall` 通过。
- `pytest -q`：140 项全部通过；`pytest.ini` 将项目测试范围固定为 `tests/`。
- 结构化扫描：两个数据集共 914 个 canonical JSON，pose/手部键残留数为 0。

## 复现命令

```powershell
.\.venv\Scripts\python.exe -m training_v3.download_rtmpose_models
.\.venv\Scripts\python.exe -m training_v3.strip_pose_annotations
.\.venv\Scripts\python.exe -m training_v3.orientation_view --dataset-dir datasets\1Ayoyo_dataset --clear
.\.venv\Scripts\python.exe -m training_v3.evaluate runs\candidates\yoyo_unified_2b0cfca8743a_orientation_roi_9cd9d9361ab5_best_yoyo-only-final-warm-freeze10-lr1e4-v1 --device 0
.\.venv\Scripts\python.exe -m pytest -q
```
