# 训练经验

## 方向时序滤波的时间确认

**结论**：将方向切换确认改为累计时间并配合连续时间 EMA，可以显著减少抖动，但在当前连续集上会牺牲宏召回；暂不作为默认生产策略。

**证据**：同一批 856 帧 raw predictions、5 FPS 稳态/25 FPS 突发回放下，现有按 observation 方案 accuracy/macro recall/预测切换数为 `0.903037/0.864272/13`；时间方案（确认 `0.16 s`、EMA 时间常数 `0.15 s`）为 `0.910047/0.836840/12`。时间方案减少 1 次切换并提高 pooled accuracy，但 macro recall 下降 `0.027432`，且邬聪聪来源组 accuracy 为 `0.727273`，低于现有 `0.747475`。

**适用范围**：当前三分类 ROI 模型、`1Ayoyo_consecutive` 856 帧、5 FPS 稳态与 25 FPS 突发调度；结论仅针对该数据规模和滤波参数。

**后续建议**：保留时间参数作为可复现实验开关；收集真实方向切换和 hard negative 后，再在独立连续集上重新校准确认时长与 EMA 时间常数。

## 检测 replay-only 与参数 soup

**结论**：在当前 replay 数据与评估协议下，replay-only 不能替代 replay+soup；soup 对连续帧召回和弱来源组有明显收益。

**证据**：同一 `1Ayoyo_consecutive` 回放中，replay-only pooled Presence F1 为 `0.927382`（TP/FN `696/109`，FP `0`），最弱邬聪聪组 F1 为 `0.634921`；replay+soup pooled F1 为 `0.977387`（TP/FN `778/27`，FP `9`），该组 F1 为 `0.944444`。同一独立 test split 复评中，replay-only mAP50-95 为 `0.546335`，replay+soup 为 `0.574198`。

**适用范围**：`detection_replay_20260830_r2` manifest、YOLO11s、`imgsz=1024`、`conf=0.15`、`IoU=0.7`，以及当前 856 帧连续集回放。

**后续建议**：保留 soup 作为默认检测权重；下一步优先收集真实 hard negatives，再评估是否能在不依赖 soup 的情况下恢复弱场景召回。
