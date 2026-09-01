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

**证据**：固定 `imgsz=1024`、`conf=0.15`、`IoU=0.7` 的 9 个连续序列评估中，当前 replay+soup 为 `TP/FN/FP=778/27/9`、Presence F1 `0.9774`；重加权候选为 `718/87/0`、Presence F1 `0.9429`，邬聪聪组 F1 从 `0.9444` 降至 `0.7445`。将采样降为每个 hard negative 1 次并缩短至 6 epoch 后仍为 `698/107/1`、Presence F1 `0.9282`。候选 native test mAP50-95 为 `0.5583`（召回 `0.8024`），与连续集回退一致。

将 5 倍候选与生产权重做参数插值可恢复大部分召回：`alpha=0.05/0.10` 均为 `TP/FN/FP=777/28/8`、Presence F1 `0.97736`，平均 IoU 分别为 `0.80355/0.80806`；但主 F1 仍低于生产，邬聪聪组 F1 为 `0.94382`，不具备晋升资格。

**适用范围**：YOLO11s、`detection_replay_20260830_r2_hn_reweight` manifest、5 个训练来源 hard negatives、12 epoch 微调及当前 `1Ayoyo_consecutive` 评估协议。

**后续建议**：保留 replay+soup 作为默认；hard negative 应扩大来源和数量，并采用较低采样权重后在独立连续集重新验证。

## 语义训练 hard-negative 与负样本采样消融优先级

**结论**：当前语义生产训练口径不是单纯的“Focal + Dice”，而是额外包含
`0.2 × hard-negative` 与空 mask `negative sampling ×4`。下一阶段应先固定模型结构、
输入、增强、seed 和数据 manifest，对这两个训练约束做独立及组合消融，再考虑继续增加
新的 Loss 项。

**证据**：生产 run `semantic_ablation_nomorph_foundation_r1` 的 manifest 记录
`hard_negative_weight=0.2`、`negative_sample_weight=4.0`；同一代码路径中的历史
`semantic_nomorph_hn005_ft_r1` 已显示将 hard-negative 权重降至 `0.05` 会改变最佳
阈值和验证行为，但该 run 为 warm-start，不能作为独立晋升证据。现有训练入口已将两
参数写入 `run_manifest.json`，可直接复用以保证消融可追溯。

**适用范围**：当前 632/136/136 reviewed semantic split、MobileNetV3-FPN、
`960x544` 输入和现有阈值/连续集评估协议；历史 warm-start 结果仅作方向提示。

**后续建议**：优先运行四格配置 `(hard-negative weight ∈ {0, 0.2}) ×
(negative sample weight ∈ {1, 4})`，每格使用独立 foundation 初始化或明确标注 lineage，
并以连续集 pooled centerline F1@8 为主指标，同时检查最弱来源组、Presence F1、最长缺失
段/恢复延迟和 FPS。只有在该消融显示稳定收益且通过独立 test/连续集护栏后，才考虑新 Loss。

**筛选记录（非晋升证据）**：在同一 foundation、`960x544`、batch 8、冻结 backbone
两 epoch 的快速筛选中，关闭 hard-negative 的 `(0,1)` 与 `(0,4)` 两格 val
centerline F1@8 分别为 `0.6229` 和 `0.6246`；启用 `0.2` 的 `(0.2,1)` 与
`(0.2,4)` 在工作阈值 `0.85/0.92/0.97` 下均无正预测（F1=0）。该结果只说明
hard-negative 在短训练和冻结阶段会强烈压低响应，不能外推到完整 12 epoch 或生产阈值。
对应 run manifest 保存在 `runs/experiments/semantic_ablation_*_screen_r*`。

**完整四格结果**：在相同 foundation、`960x544`、batch 8、冻结 backbone 3 epoch、
12 epoch 训练下，固定 test 阈值 `0.92` 的 centerline F1@8 / Presence F1 / 负图
平均误检像素分别为：`(0,1)=0.7474/0.9725/64.0`、
`(0,4)=0.7461/0.9764/45.4`、`(0.2,1)=0.7488/0.9804/36.3`、
`(0.2,4)=0.7575/0.9881/20.9`。在连续集上仅复核了 test 最优的 `(0.2,4)` 与
去掉 hard-negative 的 `(0,4)`（固定阈值 `0.9204`）：前者 pooled centerline F1@8
`0.7378`、Presence F1 `0.9907`、最弱来源组 `0.5538`、最长缺失/恢复 `6/6`；后者为
`0.7429/0.9806`、最弱 `0.5513`、最长缺失/恢复 `7/7`。两者均低于生产 pooled
`0.7662` 且未通过缺失段护栏，故不晋升；由于几何护栏已失败，未将其作为部署候选测量
FPS。完整 run 位于
`runs/experiments/semantic_ablation_hn*_neg*_fullscreen_r1`，连续集 summary
位于各 run 的 `consecutive_full/summary.json`。

**解释与后续**：在当前数据规模和训练日程下，`0.2 × hard-negative` 与
`negative sampling ×4` 的组合在独立 test 上同时带来最低误检和最高 Presence F1，
但连续集几何质量仍不足；去掉 hard-negative 未带来可靠的中心线收益。该结论不支持
继续增加新 Loss，也不支持直接删除现有 FP 约束。下一轮若要继续，应固定这两个约束，
优先尝试更长 warm-up/解冻日程或补充真实 hard-negative，并重新跑同一连续集护栏。

## Hard-negative 权重 warm-up

**结论**：在当前训练日程中，将 `0.2` hard-negative 权重在前 4 个 epoch 从 0
线性升高，未改善独立验证或 test 的中心线质量，不替换固定权重方案。

**证据**：相同 foundation、MobileNetV3-FPN、`960x544`、batch 8、12 epoch 和
`negative sampling ×4` 下，warm-up run `semantic_hn02_neg4_warmup4_fullscreen_r1`
的 val centerline F1@8 为 `0.7116`，固定权重 `(0.2,4)` 为 `0.7219`；固定 test
阈值 `0.92` 时 warm-up 的 centerline F1@8 / Presence F1 / 负图平均误检为
`0.7565/0.9841/17.4 px`，固定权重为 `0.7575/0.9881/20.9 px`。warm-up 虽略降
误检，但中心线和 Presence 均未提升，未进入连续集晋升评估。

**适用范围**：当前 632/136/136 reviewed split、batch 8、冻结 backbone 3 epoch、
12 epoch 训练；该结果不代表其他 warm-up 长度或更大真实 hard-negative 数据。

**后续建议**：保留固定 `0.2` 权重作为默认，后续优先扩大真实 hard-negative 来源，
只有训练日程或数据规模变化时才重新筛选 warm-up。

## Hard-negative 中间权重筛选

**结论**：将 hard-negative 权重从 `0.2` 降至 `0.1`，在当前数据和训练日程下没有
形成更好的 FP/召回折中，不值得进入连续集评估。

**证据**：相同 foundation、MobileNetV3-FPN、`960x544`、batch 8、冻结 backbone
3 epoch、12 epoch 训练和 `negative sampling ×4` 下，`semantic_ablation_hn01_neg4_fullscreen_r1`
的 val centerline F1@8 为 `0.7161`；固定 test 阈值 `0.92` 时 centerline F1@8 /
Presence F1 / 负图平均误检为 `0.7565/0.9843/24.5 px`，而固定 `0.2/4` 为
`0.7575/0.9881/20.9 px`。中间权重在三个指标上均未改善，未进入连续集复核。

**适用范围**：当前 632/136/136 reviewed split、batch 8、冻结 backbone 3 epoch、
12 epoch 训练；不外推到更大 hard-negative 数据或不同优化日程。

**后续建议**：保留固定 `0.2` 权重，停止在 `0.1` 附近继续微调；下一步优先收集
真实 hard-negative 或改进来源平衡，再进行大范围权重搜索。

## Warm-start 去除 hard-negative 的 seed 稳定性

**结论**：从当前生产权重 warm-start、移除 hard-negative（`0.0`）并仅训练 2 个
epoch 的方向，在当前连续集上对随机 seed 敏感，尚无稳定晋升证据。

**证据**：相同 foundation、`negative sampling ×4`、batch 2、学习率 `1e-5` 和
完整 `1Ayoyo_consecutive` 协议下，seed `20260901` 的 pooled centerline F1@8 为
`0.780601`（生产重跑 `0.781939`），seed `20260902` 为 `0.783599`；两者最弱来源组
分别为 `0.618013` 和 `0.620159`，Presence F1 均为 `0.990654`（生产 `0.991772`），
最长缺失/恢复均为 `4/4`。独立 test centerline F1@8 分别为 `0.768493` 和
`0.769260`，但连续集方向不一致。

**适用范围**：当前 632/136/136 reviewed split、MobileNetV3-FPN、`960x544` 输入、
生产权重 warm-start、2 epoch 低学习率微调及现有连续集评估协议；不外推到更长训练
或不同初始化。

**后续建议**：保留固定 `0.2` hard-negative 生产配置；若继续探索 warm-start，应先
增加独立 seed 和训练步数，再以连续集 pooled centerline F1@8 及 Presence/缺失段护栏
共同判断。
