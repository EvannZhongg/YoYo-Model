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

## 真实 backup yoyo 补充训练

**结论**：在 replay 数据中加入每个来源组第二个时间分离的 reviewed backup-yoyo 帧，只带来很小的连续集收益，尚不足以替换当前生产权重。

**证据**：9 个连续组、相同 `conf=0.15` 回放下，`k2_soup_a005` pooled Presence F1 为 `0.978084`（TP/FN `781/24`，FP `11`），当前 replay+soup 为 `0.977387`（`778/27`，FP `9`）；候选最弱组 F1 `0.934010`，当前最弱组 `0.923077`，但候选平均 IoU `0.799267` 低于 `0.800496`。将候选阈值调到 `0.25` 后 FP 降至 `9`，F1 降至 `0.967660`，最弱组降至 `0.900524`。候选训练所引用的 k2 数据 manifest 当前不在工作区，native test lineage 也无法复现；仅有跨 manifest test 复评 mAP50-95 `0.575302`，不能作为晋升证据。

**适用范围**：YOLO11s、reviewed backup-yoyo 两帧/来源组、`imgsz=1024`，当前 `1Ayoyo_consecutive` 9 组回放。

**后续建议**：保留当前 replay+soup；优先补齐真实视频 hard-negative 标注和可复现 manifest，再重新训练单模型并以 FP 护栏评估。

## FPN 解码器容量消融

**结论**：在当前 reviewed 绳线数据上，将 MobileNetV3-FPN 解码器由 32 通道增至 48 通道没有改善连续帧中心线质量；验证集提升不能直接代表连续集收益。

**证据**：相同 `f79c9805dae3c91df2ad49eb61f96db31a3236291e505c0925e3aad31f307964` manifest、输入尺寸、损失和 12 epoch 训练下，48 通道候选验证集 centerline F1@8 为 `0.7541`，但 `1Ayoyo_consecutive` 10 组/927 帧 pooled F1@8 为 `0.7519`，低于当前生产 `0.7662`；Presence F1 为 `0.9902`，FP/FN 各 `9`，不足以抵消几何指标回退。

**适用范围**：当前 632 张训练图、MobileNetV3-FPN、`960x544` 输入和现有颜色/亮脊/时序评估协议；不推断更大数据规模或不同解码器结构的结果。

**后续建议**：保持当前解码器容量，优先投入真实视频 hard-negative 的人工确认与来源隔离训练；只有在新数据扩大后才重新验证容量变化。

## Hard-negative 重加权

**结论**：在 replay 检测训练中将 5 个 reviewed not-visible hard negatives 各重复 5 次，会消除连续集误检但显著损害召回，不能作为当前生产权重。

**证据**：固定 `imgsz=1024`、`conf=0.15`、`IoU=0.7` 的 9 个连续序列评估中，当前 replay+soup 为 `TP/FN/FP=778/27/9`、Presence F1 `0.9774`；重加权候选为 `718/87/0`、Presence F1 `0.9429`，邬聪聪组 F1 从 `0.9444` 降至 `0.7445`。候选 native test mAP50-95 为 `0.5583`（召回 `0.8024`），与连续集回退一致。

**适用范围**：YOLO11s、`detection_replay_20260830_r2_hn_reweight` manifest、5 个训练来源 hard negatives、12 epoch 微调及当前 `1Ayoyo_consecutive` 评估协议。

**后续建议**：保留 replay+soup 作为默认；hard negative 应扩大来源和数量，并采用较低采样权重后在独立连续集重新验证。
