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

## 检测 hard-negative 重加权

**结论**：在 replay 检测训练中将 5 个 reviewed not-visible hard negatives 各重复 5 次，会消除连续集误检但显著损害召回，不能作为当前生产权重。

**证据**：固定 `imgsz=1024`、`conf=0.15`、`IoU=0.7` 的 9 个连续序列评估中，当前 replay+soup 为 `TP/FN/FP=778/27/9`、Presence F1 `0.9774`；重加权候选为 `718/87/0`、Presence F1 `0.9429`，邬聪聪组 F1 从 `0.9444` 降至 `0.7445`。将采样降为每个 hard negative 1 次并缩短至 6 epoch 后仍为 `698/107/1`、Presence F1 `0.9282`。候选 native test mAP50-95 为 `0.5583`（召回 `0.8024`），与连续集回退一致。

将 5 倍候选与生产权重做参数插值可恢复大部分召回：`alpha=0.05/0.10` 均为 `TP/FN/FP=777/28/8`、Presence F1 `0.97736`、平均 IoU `0.80355/0.80806`；但主 F1 仍低于生产，邬聪聪组 F1 为 `0.94382`，不具备晋升资格。

**适用范围**：YOLO11s、`detection_replay_20260830_r2_hn_reweight` manifest、5 个训练来源 hard negatives、12 epoch 微调及当前 `1Ayoyo_consecutive` 评估协议。

**后续建议**：保留 replay+soup 作为默认；hard negative 应扩大来源和数量，并采用较低采样权重后在独立连续集重新验证。

## 语义 hard-negative 与负样本采样消融

当前语义生产训练并非单纯的“Focal + Dice”，还包含 `hard-negative_weight=0.2` 和空 mask `negative sampling ×4`。本轮固定 MobileNetV3-FPN、`960x544` 输入、manifest `f79c9805dae3c91df2ad49eb61f96db31a3236291e505c0925e3aad31f307964` 以及现有颜色/亮脊/时序协议，集中比较 hard-negative 权重、负样本采样和训练日程；参数与 lineage 均记录在各 run 的 `run_manifest.json`。

在相同 foundation、batch 8、冻结 backbone 3 epoch、12 epoch 的四格消融中，固定 test 阈值 `0.92` 的 centerline F1@8 / Presence F1 / 负图平均误检像素为：`(0,1)=0.7474/0.9725/64.0`、`(0,4)=0.7461/0.9764/45.4`、`(0.2,1)=0.7488/0.9804/36.3`、`(0.2,4)=0.7575/0.9881/20.9`。因此 `(0.2,4)` 在静态 test 上同时取得较低误检和较高 Presence，但连续集复核仍不足：`(0.2,4)` pooled F1@8 `0.7378`、最弱组 `0.5538`、Presence `0.9907`、缺失/恢复 `6/6`；`(0,4)` 为 `0.7429/0.5513/0.9806/7/7`，均低于生产 pooled `0.7662`。快速冻结筛选中，`(0,1)`/`(0,4)` 的 val F1@8 为 `0.6229/0.6246`，启用 `0.2` 的两格在工作阈值下无正预测；该现象只说明短训练阶段响应受抑，不能外推到完整训练。4 epoch warm-up 或降至 `0.1` 也没有形成更好的折中：warm-up 的 val/test F1@8 为 `0.7116/0.7565`，中间权重为 `0.7161/0.7565`，均未进入晋升。

固定 seed `20260902` 的 2 epoch warm-start 去 hard-negative 结果在两个 seed 上分别为连续集 F1@8 `0.780601` 和 `0.783599`，方向不稳定。为排除分段 resume 的 scheduler 差异，又从同一生产 checkpoint 直接训练 12 epoch：`hard-negative=0.0` 的连续集 pooled F1@8 为 `0.782017`，`0.2` 为 `0.781829`，差值仅 `0.000188`；Presence 均为 `0.991209`，最长缺失/恢复均为 `4/4`，最弱组为 `0.625142/0.625423`，Chamfer 为 `14.0352/14.0369 px`，HD95 为 `60.3637/60.3906 px`。独立 test F1@8 为 `0.764793/0.764385`，Presence 均为 `0.980392`。在当前 warm-start、低学习率和 12 epoch 条件下，移除 hard-negative 没有产生可辨识的不同最终 basin，差异低于当前数据波动。

综合来看，`0.2 × hard-negative` 与 `negative sampling ×4` 仍是当前生产配置；本轮没有证据支持直接删除 hard-negative，也没有必要继续堆叠新的 Loss。该结论只适用于当前 632/136/136 reviewed split、输入尺寸和训练日程，不外推到从头训练、不同学习率或更大真实 hard-negative 数据。后续若要重新判断，应在多个 seed、从头训练和扩充真实 hard-negative 来源后，继续以连续集 pooled centerline F1@8、最弱来源组、Presence 及缺失段护栏共同评估。

可复现实验位于 `runs/experiments/semantic_ablation_hn*_neg*_fullscreen_r1`、`runs/experiments/semantic_warmprod_hn0_seed20260902_full12_r1` 和 `runs/experiments/semantic_warmprod_hn02_seed20260902_full12_r1`。
