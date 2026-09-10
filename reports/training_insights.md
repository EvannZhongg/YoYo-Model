# 训练经验

## Hessian ridge prior 融合筛选

**结论**：在当前语义 MobileNetV3-FPN 与纯模型连续集协议下，多尺度 Hessian/Frangi ridge prior 的残差门控融合没有形成可复现的中心线收益，并增加推理开销，因此不保留为默认结构。

**证据**：同一生产权重初始化、seed `20260907` 的 4 epoch 全量微调 A/B 中，ridge 相对配对基线 pooled centerline F1@8 仅 `+0.000896`，独立来源 F1 `-0.001451`，source-cluster bootstrap 95% 区间为 `[-0.00619, 0.00428]`；6 epoch 仅训练 ridge 模块时，pooled F1@8 为 `0.814955`，低于生产 `0.816521`，独立来源为 `0.875916` 对 `0.876800`，静态独立 test 为 `0.867818` 对生产 `0.877249`。冻结候选的端到端模型与后处理 FPS 低于生产，纯模型前向约增加 15%。

**适用范围**：当前 `1Ayoyo_dataset` reviewed manifest、`1Ayoyo_consecutive` 927 帧/8 来源组、`1088x608` 输入、无颜色/亮线/时序增强的纯模型评估，以及当前 MobileNetV3-FPN 容量和训练日程。

**后续建议**：暂不继续投入该 ridge 融合；若未来连续来源标注显著扩充或输入分辨率改变，应以相同来源隔离和吞吐护栏重新验证。

## 连续集低语义阈值的召回-精度折中

**结论**：在当前生产 MobileNetV3-FPN、`max_components=32` 和颜色/亮脊/光流协议下，将连续集语义阈值从 `0.9204` 降至 `0.70` 能稳定改善弱组召回、最长缺失段和几何尾部，但 pooled centerline F1@8 的全局收益很小，不足以替换生产阈值。

**证据**：同一 `semantic_ablation_nomorph_foundation_r1` 权重与 `1Ayoyo_consecutive` 927 帧回放，阈值 `0.9204/0.80/0.70` 的 pooled F1@8 为 `0.807238/0.807276/0.807949`，最弱组为 `0.615901/0.621213/0.623976`；`0.70` 将 Presence F1 从 `0.991772` 提到 `0.993428`，最长缺失/恢复由 `4/4` 降至 `2/2`，Chamfer/HD95 由 `12.7498/54.2854` 改善至 `12.5738/53.0208`，但 precision 由 `0.879165` 降至 `0.869287`。当前 manifest 的显式跨 manifest 独立 test 中，centerline F1@8 为 `0.765575/0.772613`，负图平均误检像素为 `40.182/45.273`（`0.9204/0.70`）。

**适用范围**：当前 MobileNetV3-FPN checkpoint、输入尺寸、`1Ayoyo_consecutive` 10 组/927 帧和 reviewed test 当前 manifest；阈值只作用于语义 mask，未改变训练权重。

**后续建议**：若后续候选模型带来明显更大的全局 F1 增益，可将 `0.70` 作为偏召回运行档重新评估；在当前权重上保留 `0.9204` 默认值，并把最长缺失段改善视为可接受的候选辅助收益而非单独晋升依据。

## 双阈值连通生长在静态与连续集上的分歧

**结论**：在固定高阈值 `0.9204` 下，Canny 式低阈值连通生长可改善独立静态 test 的中心线召回，但在包含时序融合与颜色候选的连续集上未提升 pooled centerline F1@8，因此不能仅凭静态指标晋升。

**证据**：同一语义 checkpoint、输入尺寸、组件上限和后处理协议下，当前静态 view（manifest 与 checkpoint 记录不同，使用显式跨 manifest 评估）test 基线 centerline F1@8 为 `0.765575`；H1 (`low=0.60`) 为 `0.773443`，H2/H3/H4/H5 分别为 `0.768612/0.769459/0.771055/0.767086`。连续集 10 组、927 帧的 pooled F1@8 基线为 `0.766256`，H1-H5 为 `0.761115/0.762203/0.759872/0.758454/0.756842`；Presence F1 均约 `0.992`，最长缺失段由基线 `4` 帧降至 H1-H3 的 `2` 帧、H4-H5 的 `3` 帧。

**适用范围**：当前 MobileNetV3-FPN checkpoint、`1Ayoyo_dataset` reviewed test 与 `1Ayoyo_consecutive` 927 帧、`max_components=8`、颜色/亮脊增强及光流时序协议；低阈值仅作用于语义 mask 连通生长。

**后续建议**：默认仍使用单阈值；若未来引入视频训练或改变时序融合策略，应在连续集主协议下重新验证低阈值，优先关注 H1 的缺失段收益是否能在不牺牲 pooled F1 的情况下保留。

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

## 更新 manifest 的语义 checkpoint 与缺失段权衡

**结论**：在扩展后的 reviewed manifest 上重新训练的语义 checkpoint 能小幅改善连续集中心线质量和最弱来源组，但会增加最长缺失段；当前收益不足以抵消安全回退，因此不替换生产权重。放宽光流前后向门限不能修复该缺失。

**证据**：相同 `1Ayoyo_consecutive`、颜色/亮脊增强、`max_components=32` 和时序协议下，生产 checkpoint 的 pooled centerline F1@8 / Presence F1 / 最长缺失段 / Chamfer / HD95 为 `0.807238 / 0.991772 / 4 / 12.7498 / 54.2854`。新 manifest checkpoint 在阈值 `0.8956` 与 `0.92` 下分别为 `0.808331 / 0.991238 / 6 / 12.3335 / 53.4835` 和 `0.808770 / 0.991238 / 6 / 12.4089 / 53.2909`；最弱来源组 F1@8 由 `0.615901` 提升至 `0.620894`。将前后向门限放宽至 `8 px` 后 pooled F1@8 为 `0.808795`，最长缺失段仍为 `6`。独立 test centerline F1@8 为 `0.763222`，低于生产 test 的 `0.767794`。

**适用范围**：当前 `0a2dbe...` reviewed manifest、MobileNetV3-FPN、`960x544` 输入和 927 帧连续集；结论不外推到新增真实视频或不同训练日程。

**后续建议**：保留生产 checkpoint 与 4 帧缺失护栏；只有在独立 test 不回退且连续集获得显著（而非千分级）F1 提升时，才重新考虑允许缺失段增加。

## 组件最小像素门限与碎片召回

**结论**：将后处理组件门限从 8 降到 4 可以缩短缺失段，但主要增加短碎片，不能稳定提升中心线几何质量；不作为默认门限。

**证据**：生产 checkpoint、相同 927 帧协议下，`min_component_pixels=8` 的 pooled F1@8 / Presence F1 / 最长缺失段 / Chamfer / HD95 为 `0.807238 / 0.991772 / 4 / 12.7498 / 54.2854`；降至 4 后为 `0.807938 / 0.992885 / 2 / 13.6885 / 58.3372`。扩展 manifest checkpoint 在门限 4 下为 `0.808569 / 0.990695 / 6 / 12.2831 / 52.7888`，未修复其 6 帧缺失。

**适用范围**：当前 MobileNetV3-FPN、`960x544` 输入、颜色/亮脊增强和 927 帧连续集；更高分辨率或不同组件融合策略需重新校准。

**后续建议**：生产默认保留 8；若部署优先关注短缺口恢复，可在 review-only 流程单独评估 4，并同时监控 Chamfer/HD95 与误检组件数。

## 连续帧密集监督的来源泛化

**结论**：将训练来源组的 684 个连续标注帧直接加入静态语义训练没有改善未见来源，且会使完整连续集不再具备独立评估资格；当前不采用该数据混合方式。

**证据**：此前 684 个连续标注帧的 warm-start 候选独立 test centerline F1@8 为 `0.756825`，低于生产 `0.767794`，负图平均误检像素由 `33.5` 增至 `38.455`；只比较未进入训练的 3 个连续来源组（243 帧），生产/候选 pooled F1@8 为 `0.869090/0.864678`。本轮将 856 个连续帧加入当前 train 来源、保持 val/test 来源不变，候选独立 test centerline F1@8 提升至 `0.891883`，Presence F1 `0.982456`，但负图平均误检升至 `58.364 px`；唯一未参与稠密训练的 Jakub 连续组由生产 `0.799098` 降至 `0.790673`，alpha `0.25/0.50/0.75` 插值也仅为 `0.791526/0.793347/0.790731`。

**适用范围**：当前 708 张静态 train 加 856 张同来源连续帧、4 px 折线栅格化、低学习率 warm-start 和 MobileNetV3-FPN；不否定来源更丰富且单独划分连续验证集后的训练收益。

**后续建议**：连续帧训练必须在来源组层面预留独立连续验证/test；后续优先增加新来源而非继续加密已有来源帧，并控制相邻帧采样权重以减少冗余。

## 轻量运动模糊增强的来源偏移

**结论**：小核方向运动模糊增强的收益不稳定且具有来源偏移，不能替换生产模型；同一扩展 manifest 的 0% 对照表明主要收益来自数据覆盖变化，而非该增强本身。

**证据**：在相同 `0a2dbe850a4dd2ce0a8e21602046a83a92c33378ff0bd5f86dddcb49a0f46eea` manifest、MobileNetV3-FPN、`960x544`、12 epoch 和 ImageNet 初始化协议下，仅对训练图以 `25%` 概率施加 `3/5 px` 四方向运动模糊。候选在同一独立 test 的 centerline F1@8 为 `0.774673`，高于生产模型的 `0.765575`；在 `1Ayoyo_consecutive` 927 帧相同后处理协议下，pooled F1@8 由 `0.807238` 提升至 `0.809548`，最弱组由 `0.615901` 提升至 `0.628453`，最长缺失/恢复保持 `4/4`。但 Jakub 来源组由 `0.770222` 降至 `0.727715`，池高宇 `4663-4812` 由 `0.712719` 降至 `0.694250`；Presence F1 由 `0.991772` 降至 `0.989543`，Chamfer/HD95 由 `12.7498/54.2854 px` 恶化至 `13.4852/56.6535 px`。在更新后的 `d24d1a0d3b56c77c60e91b66ea7d18eb1a9b05724cb45612133e865fd5f7a98a5` manifest 上，10% 模糊对照的连续集 pooled F1 为 `0.817185`，但最弱组仅 `0.577464`、HD95 `110.82 px`；同协议 0% 对照 pooled F1 为 `0.814346`，最弱组 `0.595667`、Presence F1 `0.989578`，仍低于生产弱组和 Presence 护栏。对 0% 对照测试 `0.15/0.25/0.30/0.50/0.70` 阈值后，池高宇最高 `0.601694`、Jakub 最高 `0.760966`，无法同时恢复弱组。上述结果说明扩展 reviewed manifest 带来部分 pooled/static 增益，但模糊增强概率未形成可部署的稳健收益。

**适用范围**：当前旧/新 manifest 的 661/708 张训练图、方向均匀的合成线性模糊、MobileNetV3-FPN 与现有颜色/亮脊/光流协议；不能外推到由真实相机曝光轨迹估计的模糊核或来源更丰富的数据集。

**后续建议**：若新增真实模糊样本或能从失败帧估计曝光轨迹，可降低增强概率并只使用与实际退化匹配的核重新验证；仍需同时检查 pooled、最弱组、Jakub、池高宇 `4663-4812` 和 Chamfer/HD95，不能只按平均召回晋升。

## 软拓扑损失与跨来源校准

**结论**：在当前细绳 reviewed 数据和 MobileNetV3-FPN 训练规模下，直接加入 soft-clDice 会把局部连通性收益转化为跨来源的 Presence 回退；将候选权重与同 manifest 对照做线性插值也不能同时保留弱组收益和安全指标，因此不采用该训练损失。

**证据**：同一 `d24d1a0d3b56c77c60e91b66ea7d18eb1a9b05724cb4561213e865fd5f7a98a5` manifest、seed `20260903`、`960x544`、12 epoch 和 ImageNet 初始化下，0% 对照独立 test 的 centerline F1@8 / Presence F1 / 负图误检像素为 `0.844142 / 0.992958 / 22.727`。clDice `0.2`（10 次迭代）为 `0.840217 / 0.979167 / 36.091`，clDice `0.05`（3 次迭代）为 `0.841587 / 0.968858 / 72.273`；将后者与 0% 对照按候选权重 `0.1` 插值后为 `0.84`、`0.992958`、`30.27`。关键连续来源中，clDice `0.2` 将池高宇 F1@8 提至 `0.6295`，但邬聪聪最长缺失段增至 `7`；插值模型在阈值 `0.9204` 下池高宇为 `0.6126`、Jakub 为 `0.7447`，仍低于生产 `0.6159` 和约 `0.7702`。提高空样本 hard-negative 权重至 `0.4` 将 test 负图误检降至 `11.455`，但 F1@8 降至 `0.834579`、Presence 为 `0.982332`。

**适用范围**：当前 708/151/152 reviewed split、MobileNetV3-FPN、现有 Focal+Dice+hard-negative 损失、组件级中心线评估和连续后处理；不外推到更大真实视频数据或带时序监督的拓扑模型。

**后续建议**：只有在新增真实断裂/遮挡标注并能独立校准空帧概率时，才重新验证拓扑损失；下一轮优先分析失败来源的输入退化与标注覆盖，避免继续叠加损失项。

## 骨架图分解的噪声放大

**结论**：将逐次最长路径 cover 直接替换为 endpoint/junction graph edge decomposition 会保留交叉点拓扑，但当前语义骨架中的局部毛刺会被扩成大量独立短链，降低整体召回和几何质量，因此不替换生产后处理。

**证据**：同一生产 checkpoint、阈值 `0.9204`、颜色/亮脊/光流协议和 `1Ayoyo_consecutive` 927 帧下，生产 cover 的 pooled centerline F1@8 / 最弱组 / Presence F1 / Chamfer / HD95 为 `0.807238 / 0.615901 / 0.991772 / 12.7498 / 54.2854 px`。按像素度数直接分解为 graph edge chain 时 pooled F1 降至 `0.748504`；将相邻 endpoint/junction 像素压缩成节点簇后恢复到 `0.774048`，但最弱组仍为 `0.612874`、Presence 为 `0.991218`、Chamfer/HD95 为 `13.2860/57.1648 px`，平均预测链数由 `15.82` 增至 `21.77`。在三个关键组把容量从 `32` 提至 `64` 后，F1@8 为 `0.917538/0.732636/0.612874`，仍低于生产的 `0.927129/0.734611/0.615901`，且前两组平均输出 `57.22/54.28` 条链。仅在最长路径 cover 中保留仍连接未覆盖支路的 junction，并继续使用 8 邻域，三个关键组为 `0.923525/0.728703/0.612118`，仍全面低于生产。剪除 `8 px` 内 endpoint-junction 叶枝又将弱组降至 `0.558442`，说明短边不能仅按长度视为噪声。

**适用范围**：当前 MobileNetV3-FPN 高阈值二值掩码、形态学骨架、8 邻域像素图和最多 64 点的 polyline 输出；不外推到直接预测矢量拓扑、显式 junction 监督或经过可靠 spur pruning 的骨架。

**后续建议**：若重新尝试图表示，应先让模型输出稳定的节点/边置信度，或用独立监督学习 spur pruning；不要在当前噪声骨架上继续增加按长度、角度或链数门控。

## 新 manifest 候选的召回-安全折中

**结论**：在扩展 reviewed manifest 上训练的 clDice 候选或与现有权重做参数插值，能提高静态 test 和连续集 pooled centerline F1@8，但会引入跨来源 Presence/缺失段回退；将 `1.125x` 下的组件过滤按面积放大、移除 mask closing 或删除亮脊增强，也不能得到更稳健的默认路径。

**证据**：同一 `1Ayoyo_consecutive` 927 帧、`1.125x` 和现有后处理下，`clDice` 候选 soup `alpha=0.25` 的 pooled F1@8 为 `0.831619`，但 Presence F1 `0.990695`、最长缺失 `3` 帧；固定阈值 `0.92` 仍为 Presence `0.989612`、最长缺失 `3`。生产权重的等效面积过滤（`min_component_pixels=10`）为 pooled `0.817964`、最弱组 `0.634876`，低于默认 `0.818297/0.638996`。移除 closing 的完整十组汇总为 `0.822660/0.992358/2`（pooled F1/Presence/最长缺失），几何 Chamfer/HD95 `16.5628/62.3552 px`；删除亮脊增强在关键组均小幅回退。低阈值 `0.70` 的关键组 Presence F1 `0.991720`，但 FP/FN `5/4` 且邬聪聪 Chamfer `51.94 px`。

**适用范围**：当前 `d24d1a0d...` reviewed manifest、生产 MobileNetV3-FPN 权重、`1Ayoyo_consecutive` 10 组/927 帧、`1.125x` 输入及颜色/亮脊/光流协议；不外推到新增真实视频或不同模型校准。

**后续建议**：保留现有权重和 `1.125x` 默认；若要继续优化，优先补充池高宇/邬聪聪等弱来源的真实连续标注或做来源条件校准，不再堆叠这些后处理变体。

## 标注邻域锐度门控的运动模糊增强

**结论**：用标注绳线邻域的 Laplacian P75 作为清晰度门控，可以避免对原生模糊样本重复施加退化；在当前 708 张 reviewed train 图上，阈值 `37` 约筛出每轮 260-286 个 eligible 样本，10% 概率实际增强约 22-33 个。该候选在部分初始化下改善连续集 pooled F1，但静态 test 误检或几何尾部回退，仍不满足生产护栏，因此暂不晋升。

**证据**：同一 `d24d1a0d3b56c77c60e91b66ea7d18eb1a9b05724cb4561213e865fd5f7a98a5` manifest、MobileNetV3-FPN、seed `20260903`、`960x544`、12 epoch 和现有后处理协议下，选择性模糊候选（`p=0.10`, `sharpness>=37`）独立 test centerline F1@8 / Presence F1 / 负图误检像素为 `0.873075 / 0.975779 / 59.818`，同 manifest 0% 对照为 `0.844142 / 0.992958 / 22.727`。连续集候选 pooled F1@8 / 最弱组 / Presence F1 / 最长缺失-恢复 / Chamfer-HD95 为 `0.824158 / 0.637303 / 0.988612 / 2-2 / 26.677-87.340`；阈值 `0.995` 仅将 pooled 调为 `0.822270`、最弱组 `0.632262`、Chamfer-HD95 `25.363-84.927`，仍低于当前生产弱组约 `0.638996` 和几何基线 `15.621-57.470`。

从当前生产权重 warm-start 的复核（同一 `d24d1a0d...` manifest、seed `20260904`、相同 `1.125x` 与后处理协议）得到独立 test `F1@8/Presence F1/负图误检` 为 `0.871269/0.979167/73.364`，严格同协议生产对照为 `0.877249/0.979167/60.364`；连续集候选 pooled F1@8 / 最弱组 / Presence F1 / 最长缺失-恢复 / Chamfer-HD95 为 `0.819801/0.670450/0.995079/1-1/13.91-51.51`，生产为 `0.818297/0.638996/0.994530/2-2/15.72-57.50`。候选端到端 `11.462 FPS`，生产配对均值约 `11.14 FPS`；速度通过但静态 test 和误检护栏失败，不能晋升。

**适用范围**：当前 reviewed 训练规模、Laplacian 邻域锐度判定、3/5 像素方向核、MobileNetV3-FPN 和 `1Ayoyo_consecutive` 10 组协议；不外推到真实曝光轨迹或更大来源数据。

**后续建议**：保留该增强开关和 run manifest 统计，默认概率为 0；只有补充真实模糊来源并证明几何尾部不恶化时，才重新评估并考虑部署。

## DSCNet and Ariadne+ architecture screening

**结论**：在当前绳线语义任务和训练预算下，DSCNet 的动态偏移采样没有带来可测收益，Ariadne+ 的 ResNet-101 虽优于 MobileNetV2，但仍明显低于生产验证指标；两者都不适合作为替换方向。

**证据**：DSCNet 4 epoch 对照中，启用动态 offset 与关闭 offset 的验证 Centerline F1@8 均为 `0`、Presence 均为 `0`，loss 分别为 `1.0019/1.0022`；动态采样推理约 `14.48 FPS`。Ariadne+ 4 epoch 中，ResNet-101 的验证 F1@8 / Presence 为 `0.7169/0.9818`、约 `47.82 FPS`，MobileNetV2 为 `0.6992/0.9817`、约 `155.83 FPS`；当前生产验证为 `0.813005/0.987952`。

**适用范围**：当前 reviewed 训练集、同一输入与评估脚本、4 epoch 筛选预算及 RTX 4070 Laptop；短训练筛选不能代表这些架构在更大数据规模上的理论上限。

**后续建议**：不保留 DSCNet offset 分支或 Ariadne+ backbone 作为生产候选；若未来数据规模显著增加，应先以独立 test 和连续集护栏重新验证，再考虑架构级投入。

## RT-DLO semantic backbone and graph solver comparison

**结论**：RT-DLO 的 ResNet-101 语义骨干在当前连续集上有小幅 pooled F1@8/几何收益，但参数量、显存和端到端速度代价过大，且最长漏检段明显恶化；官方图求解器接入后进一步失效，不具备替换生产方案的条件。

**证据**：同一 `1Ayoyo_consecutive` 927 帧、`1.125x` 和现有颜色/亮脊/光流协议下，RT-DLO 语义骨干 pooled centerline F1@8 / Presence F1 为 `0.844558/0.990077`，生产为 `0.818297/0.994530`；Chamfer/HD95 为 `9.1097/37.4760 px` 对 `15.6212/57.4703 px`，但最长缺失/最大恢复从生产 `2/2` 恶化为 `7/7`。RT-DLO 最弱来源组为 `池高宇-fef6c7bcb0` 的 `0.683994`，生产连续子组最低为 `0.638996`（同来源另一段为 `0.735104`）。纯模型为 `46.97 FPS`、`557 MB`、45.6M 参数，生产约 `156.68 FPS`、`138 MB`；同一邬聪聪 300 帧端到端复测为 RT-DLO `10.593 FPS`，生产 `13.655/10.891 FPS`。使用官方 `CP_angle.pth`、等价几何采样/KNN 实现和仓库图路径求解器后，pooled F1@8 / Presence F1 降至 `0.224730/0.759837`，最长缺失/最大恢复为 `33/32`。

**适用范围**：当前 RT-DLO-abhay ResNet-101 适配、同一训练 manifest 与 checkpoint、RTX 4070 Laptop、`1Ayoyo_consecutive` 10 组/927 帧及邬聪聪 300 帧端到端协议；图求解器结果还受其原始工业缆线掩码假设影响，不外推到其他分辨率或数据域。

**后续建议**：保留 RT-DLO 语义结果作为架构上限和失败证据，不晋升权重或图后处理；后续优先补充弱来源连续标注与模型校准，避免在当前细绳任务上继续堆叠图求解流程。

## RT-DLO ASPP migration and teacher distillation screening

**结论**：将 RT-DLO 的五分支 ASPP 加到 MobileNetV3-FPN 最深层，或用已训练的 RT-DLO ResNet-101 soft logits 蒸馏同一 student，在当前短训练筛选中都没有超过原始 MobileNetV3-FPN；ASPP 还降低了纯模型吞吐，因此两条路线均不进入完整训练或独立 test/连续集评估。

**证据**：固定 manifest `d4f0cc89cdb4e6727b723e609e7b02c35cac8da4c6a2275a89d8a74ee7a62344`、seed `20260830`、ImageNet 初始化、前三轮冻结 backbone、4 epoch 和同一 35 阈值验证协议，原始 MobileNetV3-FPN 的 val centerline F1@8 / Presence F1 为 `0.793190 / 0.978723`；RT-DLO ASPP（`1x1 + rates 6/12/18 + global pooling`，空洞卷积 depthwise-separable）为 `0.783161 / 0.989247`；RT-DLO teacher soft-logit 蒸馏（temperature `2.0`、weight `0.5`）为 `0.778857 / 0.989170`。蒸馏 checkpoint 已验证只含 3.016M student 参数和标准 MobileNetV3-FPN state dict，不含 teacher 参数。RTX 4070 Laptop、`1088x608`、AMP、串行 100 次前向下，ASPP 为 `90.38 FPS`，原始 student 为 `99.04 FPS`，约下降 `8.7%`；ASPP 参数为 3.028M，对照为 3.016M。

**适用范围**：当前 798/151/152 reviewed split、MobileNetV3-FPN 32 通道 decoder、4 epoch 架构筛选预算，以及由同一 train split 训练且与 val/test 来源隔离的 RT-DLO teacher；短训练结果不代表更大数据规模或其他蒸馏权重的理论上限。

**后续建议**：当前不继续增加 ASPP 分支，也不为该 teacher 扩展蒸馏损失；若未来 teacher 在新的独立连续来源上形成更强且更稳定的优势，可先用同 seed 短训练 A/B 重新筛选较低蒸馏权重，再决定是否投入完整训练。

## 稀疏中心线的正像素加权监督

**结论**：对 skeleton distance transform 形成的稀疏中心线目标，正像素加权 BCE 配合较低 Dice 权重，比 Dice 主导的损失更能避免模型退化为全背景；该改动改善了中心线召回，是后续多任务训练的必要监督设置。

**证据**：在相同 MobileNetV3-FPN、`960x544` 输入、reviewed manifest 和训练日程下，原 Dice 主导配置在稀疏几何分支出现近全背景响应；加入正像素加权 BCE 并降低 Dice 项后，验证集中心线响应恢复，随后可稳定进行 2theta tangent 多任务训练。该结论来自连续的 loss 对照与 warm-start 复核，而非单次随机波动。

**适用范围**：当前绳线中心线的距离变换热图、稀疏正像素比例和 MobileNetV3-FPN 容量；不外推到稠密分割目标或不同标注线宽。

**后续建议**：保留正像素加权 BCE 作为稀疏中心线默认监督；若标注缓冲半径、距离变换 sigma 或目标密度变化，应重新校准正负权重与 Dice 比例。

## 双角切线场多任务候选

**结论**：共享 MobileNetV3-FPN backbone、mask/heatmap 与 `(cos 2theta, sin 2theta)` tangent head 的融合模型可以学习稳定的无向切线几何，几何指标接近生产量级，但当前弱来源与 pooled 连续集 F1 仍低于生产护栏，因此不晋升默认模型。

**证据**：候选 `runs/experiments/centerline_fusion_2theta_warm_e12` 使用 manifest `d4f0cc89cdb4e6727b723e609e7b02c35cac8da4c6a2275a89d8a74ee7a62344`、ImageNet 兼容 warm-start、等效 12 epoch；验证集 fused F1 为 `0.7020`、2theta cosine 为 `0.9341`，独立静态 test centerline F1@8 为 `0.7466`。与生产追踪器相同的 `1088x608` 连续集协议下，pooled centerline F1@8 / Presence F1 为 `0.7861 / 0.9918`，最长缺失/恢复为 `2/2`，FPS 约 `9.85`；生产对照为 `0.8072 / 0.9945`，弱来源组候选约 `0.5757` 对生产 `0.6159`。因此候选的缺失段护栏达到生产水平，但 pooled、弱组和 Presence 均未达到晋升要求。

**适用范围**：当前 `798/151/152` reviewed split、`960x544` checkpoint、`1.125x` 连续集推理、颜色/亮脊/光流协议和 MobileNetV3-FPN 共享 backbone；不外推到更大连续标注规模或不同来源分布。

**后续建议**：保留 2theta 表示和最小几何融合路径作为后续实验基线，优先补充池高宇等弱来源的连续标注并重新校准阈值；只有在独立 test、连续集 pooled/弱组和吞吐同时满足护栏时才考虑晋升。

## 融合输出直接监督的短训退化

**结论**：对最终 `mask × heatmap` 融合概率直接加入加权 BCE，会在当前 4 epoch 训练预算下显著抑制中心线响应，不能作为现有多任务损失的附加项。

**证据**：同一 `d4f0cc89...` manifest、MobileNetV3-FPN、ImageNet 初始化、`960x544` 和 seed `20260906` 下，融合 BCE 权重 `0.25` 的验证 fused F1 为 `0.3780`，而同协议无该项的短训模型约为 `0.69`；训练期间第 1 epoch fused F1 仅 `0.0364`，第 4 epoch 仍未恢复。

**适用范围**：当前 mask/heatmap 乘积解码、稀疏中心线目标、4 epoch 筛选日程和现有正像素加权 BCE；不外推到不同融合函数或长周期重新平衡后的训练。

**后续建议**：保留分支独立监督，若未来改变融合定义，应先重新设计归一化目标并进行独立 A/B，不直接叠加融合损失。

## 高斯中心线热图继续训练的连续集回退

**结论**：将高斯中心线热图筛选模型从 8 epoch warm-start 到等效 12 epoch，可恢复较高的静态验证与 test 质量，但连续集存在性和弱来源明显回退，不能替代当前生产模型。

**证据**：同一 `d4f0cc89...` manifest、MobileNetV3-FPN、`960x544` 和 seed `20260906` 下，`centerline_gaussianheat_screen_e8` warm-start 候选的独立 test pooled centerline F1@8 为 `0.7691`；`1Ayoyo_consecutive` pooled F1@8 / Presence F1 / 最长缺失-恢复 / Chamfer-HD95 / FPS 为 `0.7930 / 0.9891 / 6-6 / 10.16-39.81 px / 11.76`，最弱来源组 F1@8 为 `0.6087`。均低于当前生产连续集 `0.8183 / 0.9945 / 2-2` 及最弱组 `0.6390`。

**适用范围**：当前 reviewed manifest、8+4 epoch warm-start、Gaussian sigma `3.0`、现有颜色/亮脊/光流协议和 RTX 4070 Laptop；不外推到重新标注的连续帧或不同热图权重。

**后续建议**：保留高斯热图作为失败筛选证据，不继续投入完整训练；后续应优先补充连续弱来源标注，再评估直接中心线监督或目标重加权。

## 语义解码器迁移到几何多任务模型

**结论**：将已训练语义 MobileNetV3-FPN 的 encoder/FPN 和 `classifier` 解码器映射到几何模型的 mask head，再进行 12 epoch 几何 warm-start，能稳定提高验证和独立 test 的中心线质量；但 Presence 仍略低于生产护栏，不能仅凭几何 F1 晋升。

**证据**：同一 `d4f0cc89cdb4e6727b723e609e7b02c35cac8da4c6a2275a89d8a74ee7a62344` manifest 下，语义解码器迁移候选 `runs/experiments/centerline_semantic_init_e12` 的验证 fused proxy F1 / 2theta cosine 为 `0.7247 / 0.9140`，高于原 2theta 候选的 `0.7020 / 0.9341`；使用验证选定阈值 `0.04` 的独立 test centerline F1@8 为 `0.7667`，高于原候选 `0.7466`。同一颜色、亮脊、光流和 `1.125x` 连续集协议下，候选 pooled centerline F1@8 为 `0.8072`、最弱来源组 `0.6360`、Presence F1 `0.99345`（`TP/FP/FN=910/8/4`），几何 Chamfer/HD95 均值约 `9.44/40.13 px`；当前生产 pooled F1@8 `0.8183`、最弱组 `0.6390`、Presence F1 `0.9945`。将解码支持阈值收紧到高阈值的 `0.60` 倍可把 pooled F1 提到 `0.8191`，但 Presence 降到 `0.9913`，说明主要回退来自时序存在性而非中心线定位。

**适用范围**：当前 `798/151/152` reviewed split、MobileNetV3-FPN 32 通道、`960x544` checkpoint、`1.125x` 连续集推理和现有候选后处理；不外推到新来源、不同训练日程或更大连续标注规模。

**后续建议**：保留“语义解码器初始化 + 几何监督”的训练策略作为后续基线，但暂不把它写入默认生产路径；下一步优先针对弱来源的连续存在性补充标注或做来源隔离校准，避免继续堆叠解码阈值规则。

## 直接人工 polyline 监督的短训收益不持续

**结论**：将 canonical 标注中的人工 `string_polylines_pixel` 直接栅格化为 heatmap/tangent target，在 4 epoch 筛选中略优于从 polygon mask 骨架化的 target，但 warm-start 到等效 12 epoch 后反而落后；当前标注的 polyline 采样密度和可用性不足以替换现有 mask-skeleton target。

**证据**：同一 manifest、MobileNetV3-FPN、`960x544`、batch 4、seed `20260906` 下，4 epoch direct-polyline 的验证 fused F1 为 `0.4002`，同预算 mask-skeleton 对照为 `0.3652`；从该 checkpoint 再 warm-start 8 epoch 后，验证 fused F1 仅 `0.6796`，低于 2theta mask-skeleton 候选 `0.7020` 和语义解码器迁移候选 `0.7247`。因此短训差异不能作为直接人工中心线监督有效的证据。

**适用范围**：当前 `798/151/152` reviewed split、约 1024/1101 canonical 样本有人工 polyline、现有 polygon mask 与 target 生成流程；不外推到重新审核或更密集的中心线标注。

**后续建议**：删除该实验专用 target 分支，保留 polygon-mask 骨架化作为默认；若未来补齐连续帧中心线标注，应先建立来源隔离的 direct-polyline A/B，再投入完整训练。

## 连续弱来源的中心线指标分解

**结论**：在当前连续集的弱来源中，pooled centerline F1@8 的主要波动来自目标覆盖召回和可见性切换，不是单纯的骨架 path 碎片化；对称 F1 适合保留为定位主指标，但不能单独代表可部署追踪质量。

**证据**：基于 `semantic_mobilenetv3_fpn_e30` 的逐帧原图、标注和预测对齐诊断（几何均值仅在有目标中心线的帧上计算），池高宇 `381-500` 的 union precision/recall/F1@8 均值为 `0.972/0.823/0.888`，但最长单 path 的 target 覆盖仅 `0.245`，平均预测 path 数 `20.7`；池高宇 `4663-4812` 为 `0.916/0.650/0.748`，邬聪聪 `6200-6399` 可见帧为 `0.822/0.752/0.773`，另有 `5` 个 `not_visible` 帧。逐帧 F1 与 recall 的相关系数在四组为 `0.942-0.981`，而与碎片化指数的相关系数仅 `0.302-0.420`（Jakub 对照组 `0.343`）。

**适用范围**：当前 `1Ayoyo_consecutive` 十组连续集、`1088x608` 输入、e30 语义模型和现有 path 输出/评估实现；诊断基于人工 `string_polylines_pixel` 与逐帧 JSONL 预测，不外推到重新标注的数据。

**后续建议**：保留 pooled centerline F1@8 作为几何主排名（只在有目标中心线的帧上累计采样点），同时固定报告可见帧的 precision、recall、F1，以及 `not_visible` 帧的 presence 误检率/误检帧数；另外报告最长缺失/恢复延迟、主干覆盖率和预测碎片化指数。当前数据中 `visible` 与 `partial` 没有形成稳定的几何差异，不单独设为晋升门槛。只有当主指标、最弱来源和时序护栏同时通过时才晋升；不要用单 path 覆盖率替换对称 F1，也不要仅凭 presence F1 晋升。

## Visibility auxiliary head screening

**结论**：将 `string_visibility` 作为共享 FPN 的三分类辅助头，当前数据规模下不能稳定提供帧级阈值策略；该组件不进入默认模型。

**证据**：同一 `b0d246da...` manifest、MobileNetV3-FPN、ImageNet 初始化和现有后处理协议下，未加权辅助头（8 epoch）独立 test 的 centerline F1@8 / Presence F1 / 负图误检为 `0.878590 / 0.972414 / 111.091 px`，连续集 pooled F1@8 / Presence F1 / 最弱组 / 最长缺失段为 `0.817177 / 0.995326 / 0.661527 / 1`；类别加权复训（visible/partial/not_visible 权重 `3.0/0.5/2.0`，6 epoch）test 为 `0.877502 / 0.978873 / 53.455 px`。未加权头验证混淆矩阵中 `visible` 13 张全部被判为 `partial/not_visible`，test 25 张全部漏检，说明辅助头没有学到可部署的清晰度状态。

**适用范围**：当前 798/151/152 张 reviewed 训练/验证/测试图、`visible/partial/not_visible=92/652/54` 的长尾分布、MobileNetV3-FPN 和 8/6 epoch 筛选训练。

**后续建议**：若未来补充数量平衡且跨来源的清晰/模糊连续标注，可重新验证辅助状态头；在此之前保持单一语义概率和现有时序后处理，避免把不可靠的状态预测接入阈值控制。

## Confidence-distribution threshold policy screening

**结论**：直接用语义概率图自身的高分分位数作为帧级“保守/宽松”阈值开关，可以在验证集得到极小的几何收益，但连续集 Presence 和几何尾部明显回退，不具备部署价值。

**证据**：同一 `semantic_retrain_newmanifest_baseline_e12` checkpoint，在验证集对 `p99.9` 等输出统计做二档阈值搜索，固定低阈值对照约为 `0.8181`，最佳 `p99.9` 策略（阈值低/高 `0.10/0.92`，切分 `0.999422`）约为 `0.8200`。在 `1Ayoyo_consecutive` 十组、`1.125x`、颜色/亮脊/时序协议下，该策略 pooled F1@8 `0.821165`，高于生产 `0.818297`，但 Presence F1 `0.988399`（生产 `0.994530`）、最弱来源组 `0.637425`（生产 `0.638996`）、最长缺失段 `2`，Chamfer/HD95 均值约 `41.48/96.18 px`，未通过安全与几何护栏。

**适用范围**：当前 `1Ayoyo_dataset` 验证集 151 张、`1Ayoyo_consecutive` 十组 927 帧、MobileNetV3-FPN 语义 checkpoint 和现有组件后处理；策略阈值在同一验证集上搜索，未用于默认配置。

**后续建议**：若未来获得更多连续弱域帧，可把概率分布作为诊断特征继续研究，但必须加入来源隔离校准并优先满足 Presence、最弱来源和 HD95 护栏；当前保持单阈值默认路径。

## Partial-label background-loss screening

**结论**：把 `partial` 样本未标出的像素视为弱背景（背景 loss 权重 `0.5`），试图减少对模糊绳段的过度抑制，当前训练规模下没有带来有效收益，且静态 test 几何指标回退。

**证据**：同一 `b0d246da...` manifest、MobileNetV3-FPN、ImageNet 初始化、8 epoch 和现有阈值扫描协议下，`partial` 背景权重 `0.5` 的验证 F1@8 为 `0.796187`，独立 test Centerline F1@8 / Presence F1 / 负图误检为 `0.868486 / 0.975610 / 73.545 px`；同日新 manifest 对照为 `0.884101 / 0.978873 / 12.636 px`。因此降低 partial 背景惩罚没有改善弱绳线召回，反而增加了误检。

**适用范围**：当前 798/151/152 张 reviewed split、`partial` 占训练集 652 张、MobileNetV3-FPN 和 8 epoch 训练；未改变连续集评估协议。

**后续建议**：只有在 partial 标注明确区分“未知区域”并补充连续帧验证后，才值得重新尝试 ignore/soft-target 监督；当前保留标准背景监督。

## Hysteresis threshold screening

**结论**：在固定高阈值语义种子上使用低阈值连通扩张，能改善部分弱来源的中心线召回；但 pooled 增益很小，并伴随 Presence 或几何尾部回退，当前不替换默认单阈值路径。

**证据**：当前生产权重、`1Ayoyo_consecutive` 十组、`1.125x`、颜色/亮脊/时序和 `max_components=32` 协议下，固定高阈值 `0.9204` 的低阈值 `0.2/0.4/0.6` 分别得到 pooled F1@8 `0.820138/0.819548/0.819458`，Presence F1 `0.992916/0.994012/0.994012`，最弱组 `0.654377/0.649767/0.648590`，最长缺失段均为 `2` 帧；Chamfer/HD95 均值分别为 `18.42/65.55`、`18.77/66.26`、`18.98/65.96 px`。相较生产 `0.818297/0.994530/0.638996/2`，`0.2` 的弱组收益最大但 Presence 回退更明显，`0.4` 的安全回退较小但总体增益仍不足以抵消部署风险。

**适用范围**：当前生产 MobileNetV3-FPN 权重、RTX 4070 Laptop、十组连续集 927 帧及现有后处理；结果保存在 `tmp/hyst_0p2/`、`tmp/hyst_0p4/`、`tmp/hyst_0p6/`。

**后续建议**：若未来能由来源隔离校准器只对高置信且时序连续的帧启用 hysteresis，可重新验证；当前不把低阈值扩张写入默认 tracking 配置。

## Rasterized string-target width screening

**结论**：将栅格化细绳监督的最小宽度从 `1` 提高到 `2` 个输入像素，可在短训验证中提高中心线召回，但会把最优阈值推到很低并增加误检；它没有形成跨 split、跨来源的稳健收益，不替换默认监督。

**证据**：同一 `b0d246da...` manifest、生产 MobileNetV3-FPN 权重 warm-start、seed `20260909` 和 4 epoch 预算下，`min_mask_width_px=2` 的验证 centerline F1@8 / Presence F1 为 `0.823810 / 0.989091`，最优阈值 `0.1749`；独立 test 为 `0.880246 / 0.989474`，负图平均误检 `34.0 px`，低于同 manifest 对照 centerline F1@8 `0.884101`。按现有十组连续集、`1.125x`、颜色/亮脊/时序协议评估，候选 pooled F1@8 约 `0.833105`、Presence F1 `0.991238`、最弱组 `0.638201`、最长缺失/恢复 `4/4`、Chamfer/HD95 `13.79/55.23 px`；虽高于生产 pooled `0.818297`，但 Presence、弱组和静态 test 均未过护栏。

**适用范围**：当前 `798/151/152` reviewed split、`1Ayoyo_consecutive` 十组 927 帧、MobileNetV3-FPN 和 4 epoch warm-start 筛选；连续集 pooled 数值由逐组 centerline hit 汇总得到。

**后续建议**：保持 `min_mask_width_px=1`。若未来补齐可靠的细绳宽度/不确定区域标注，应先在来源隔离的完整训练中验证空间软目标，再考虑调整栅格宽度。

## Fixed high-frequency input channel screening

**结论**：在 RGB 输入前追加固定 Laplacian 高频通道，能提高部分弱来源的定位召回，但未同时满足 Presence 与缺失段护栏；该网络分支不进入默认模型。

**证据**：同一 `b0d246da...` manifest、MobileNetV3-FPN、生产权重 warm-start 和 seed `20260909` 下，4 epoch 高频通道候选在独立 test 的 centerline F1@8 / Presence F1 / 负图误检为 `0.892414 / 0.986014 / 28.909 px`；连续集 pooled F1@8 约 `0.833522`（阈值 `0.1749`）和 `0.836923`（阈值 `0.995`），最弱组分别约 `0.6648/0.6567`，但 Presence 仅 `0.992900/0.987397`，最长缺失/恢复为 `4/4` 和 `6/6`。延长至等效 12 epoch 后，独立 test 为 `0.895912 / 0.992958 / 21.636 px`，连续集 pooled F1@8 `0.840550`、最弱组 `0.656663`、Presence `0.987397`、最长缺失/恢复 `6/6`、Chamfer/HD95 `12.29/54.17 px`，仍未通过安全门槛。

**适用范围**：当前 `798/151/152` reviewed split、`1Ayoyo_consecutive` 十组 927 帧、MobileNetV3-FPN 和固定 Laplacian 输入通道实验；连续集按现有颜色/亮脊/时序协议评估。

**后续建议**：保持三通道 RGB 默认输入。若未来引入高频表征，应优先使用可学习且来源隔离的轻量 stem，并以 Presence/最长缺失护栏约束，不保留本轮专用分支。

## FPN nearest-upsampling screening

**结论**：将 MobileNetV3-FPN 自顶向下融合的双线性上采样替换为 nearest，可改善存在性和短缺失段，但会产生严重几何尾部回退，不适合作为默认解码器。

**证据**：同一 `b0d246da...` manifest、生产权重 warm-start、seed `20260909` 和 4 epoch 筛选下，nearest 候选验证 centerline F1@8 为 `0.8218`。独立 test centerline F1@8 / Presence F1 / 负图误检为 `0.893491 / 0.979167 / 43.364 px`。连续集按现有颜色/亮脊/时序协议的 pooled F1@8 约 `0.8294`，Presence F1 `0.997278`，最弱来源组 `0.6574`，最长缺失/恢复 `1/1`，但 Chamfer/HD95 `31.13/99.76 px`，其中单个来源组 Chamfer 约 `192 px`，超过几何护栏。

**适用范围**：当前 `798/151/152` reviewed split、十组 927 帧连续集、MobileNetV3-FPN 和 4 epoch warm-start；未改变评估类别口径。

**后续建议**：保留 bilinear 上采样默认路径。若未来重新设计解码器，应要求几何尾部与 Presence 同时通过，不能仅凭缺失段改善晋升。

## Shallow input-detail skip screening

**结论**：向最高分辨率 FPN 层加入一个轻量 RGB 浅层细节支路，短训中提高了验证中心线召回，但独立 test 几何和误检显著回退，不进入默认网络。

**证据**：同一 `b0d246da...` manifest、生产权重 warm-start、seed `20260909` 和 4 epoch 筛选下，input-skip 候选验证 centerline F1@8 为 `0.8296`、Presence F1 `0.9928`；独立 test centerline F1@8 / Presence F1 / 负图误检为 `0.873214 / 0.982578 / 56.273 px`，均低于同 manifest 生产对照的 `0.884101 / 0.978873 / 12.636 px`（对照数值来自同日评估）。

**适用范围**：当前 `798/151/152` reviewed split、MobileNetV3-FPN、4 epoch warm-start 和既有组件后处理；未进入连续集完整评估，因为独立 test 已显示明显几何与误检回退。

**后续建议**：保留原始 bilinear FPN，不增加浅层旁路；后续网络创新应先通过独立 test 的几何和负图误检筛选，再投入连续集资源。

补充：高频通道候选在等效 12 epoch 权重上将连续集阈值降至 `0.15` 时，pooled F1@8 约升至 `0.8453`、Chamfer/HD95 改善至 `11.07/50.26 px`，但 Presence F1 仅 `0.9907`、最弱来源组约 `0.67`，最长缺失/恢复仍为 `4/4`；因此该 operating point 仍不能满足部署安全门槛。

## 语义边界带加权筛选

**结论**：在 focal 项中对标注掩码的一像素形态学边界带额外加权，未改善独立 test，且增加误检；该损失改动不进入默认训练。

**证据**：同一 `b0d246da...` manifest、生产权重 warm-start、MobileNetV3-FPN、seed `20260909` 和 4 epoch 筛选下，边界权重 `0.5` 的独立 test centerline F1@8 / Presence F1 / 负图平均误检像素为 `0.886799 / 0.975610 / 66.455`；同协议生产对照为 `0.888527 / 0.979167 / 60.364`。验证集最佳 epoch 为 2，阈值 `0.3805`，短训收益未能转化为 test 几何或存在性收益。

**适用范围**：当前 `798/151/152` reviewed split、MobileNetV3-FPN 32 通道、掩码骨架目标和 4 epoch warm-start 筛选；未进入连续集评估。

**后续建议**：保持标准 focal + Dice + hard-negative 损失。若未来获得更精确的中心线/边界标注，应先在来源隔离 test 上验证边界监督，再考虑投入完整训练。

## Visibility-conditioned semantic logit screening

**结论**：让共享 FPN 的全局质量头学习 `string_visibility`（visible/partial/not_visible）并对分割 logits 做帧级自适应偏置，短训中未形成可部署的置信策略；其独立 test 的召回与存在性均低于生产。

**证据**：同一 `b0d246da...` manifest、生产权重 warm-start、MobileNetV3-FPN、seed `20260911` 和 4 epoch 筛选下，候选 test centerline F1@8 / Presence F1 / 负图平均误检像素为 `0.886232 / 0.972414 / 70.455`，生产同协议对照为 `0.888527 / 0.979167 / 60.364`。候选验证选择的阈值为 `0.6877`；在 test 上扫描 `0.15–0.995` 后 F1 最高约 `0.8868`，Presence 始终 `0.9724`，说明质量头没有提供可利用的 operating-point 分离。

**适用范围**：当前 `798/151/152` reviewed split、`visible/partial/not_visible=92/652/54` 长尾分布、MobileNetV3-FPN 和 4 epoch warm-start 筛选；未进入连续集完整评估。

**后续建议**：不要把当前 `visible/partial` 辅助头接入推理阈值；若未来补充跨来源、帧级质量标注，应先验证质量分数与弱来源召回/误检的独立相关性，再考虑自适应策略。

## Partial-frame positive-gradient weighting screening

**结论**：仅提高 `partial` 样本中正像素的 focal 梯度权重，能降低部分负图误检并提高 Presence，但中心线 F1@8 在独立 test 仍回退，不能作为默认训练策略。

**证据**：同一 `b0d246da...` manifest、生产权重 warm-start、MobileNetV3-FPN、seed `20260912` 和 4 epoch 筛选下，partial 正像素权重 `0.5` 的 test centerline F1@8 / Presence F1 / 负图平均误检像素为 `0.882565 / 0.982578 / 35.818`；此前同协议生产对照为 `0.888527 / 0.979167 / 60.364`。验证最佳阈值为 `0.2268`，说明该改动主要把 operating point 推向宽松召回，未改善几何主指标。

**适用范围**：当前 `798/151/152` reviewed split、`partial` 长尾标注、MobileNetV3-FPN 和 4 epoch warm-start 筛选；未进入连续集评估。

**后续建议**：保持统一正例监督；若未来能构造可靠的 partial 未知区域 mask，应改用显式 ignore/soft-target 监督并重新进行来源隔离评估。

## FPN decoder deep-supervision screening

**结论**：在 FPN 的三个中间解码尺度加入训练期深监督，独立 test 的中心线指标与生产几乎持平，但 Presence 和负图误检回退；当前不保留该训练分支。

**证据**：同一 `b0d246da...` manifest、生产权重 warm-start、seed `20260913` 和 4 epoch 筛选下，深监督权重 `0.15` 的 test centerline F1@8 / Presence F1 / 负图平均误检像素为 `0.888456 / 0.975779 / 63.909`，同协议生产对照为 `0.888527 / 0.979167 / 60.364`。候选最佳 epoch 为 2、阈值 `0.8414`，主指标差异仅 `-0.000071`，不足以抵消安全回退。

**适用范围**：当前 MobileNetV3-FPN、1/8–1/2 三尺度辅助头、自适应最大池化目标和 4 epoch warm-start；未进入连续集评估。

**后续建议**：保持单一最终解码监督；只有训练数据和正样本密度明显增加时，才重新验证深监督权重与低分辨率目标构造。

## FPN concatenation fusion screening

**结论**：将 FPN 的逐层相加改为 lateral/top-down 特征拼接后投影，能提高连续集 pooled centerline F1@8 和最弱来源组，但 Presence、缺失段和推理吞吐回退，当前不晋升。

**证据**：同一 `b0d246da...` manifest、生产权重 warm-start、MobileNetV3-FPN、seed `20260915` 和 4 epoch 筛选下，独立 test centerline F1@8 / Presence F1 / 负图误检为 `0.897600 / 0.982578 / 28.727 px`，生产对照为 `0.888527 / 0.979167 / 60.364 px`。完整 `1Ayoyo_consecutive` 十组、`1.125x`、颜色/亮脊/时序协议下，候选 pooled F1@8 约 `0.8316`、最弱组 `0.6844`、Presence `0.9913`、最长缺失/恢复 `4/4`、Chamfer/HD95 `12.54/51.47 px`；生产为 `0.8183/0.6390/0.9945/2/2/15.62/57.47 px`。同一 GPU 纯模型前向约 `98.84 FPS`，生产 `108.91 FPS`，下降约 `9.2%`。将组件上限降至 16 后 pooled F1@8 约 `0.8163`，未保留主指标收益；阈值 `0.995` 时 pooled 约 `0.8244`、Presence `0.9896`、最长缺失 `5`，不能恢复安全护栏。

**适用范围**：当前 reviewed split、`1Ayoyo_consecutive` 927 帧、MobileNetV3-FPN 32 通道和现有后处理；连续集结果用于同协议候选判断，不外推到不同硬件或更大数据规模。

**后续建议**：保持原始 bilinear FPN 相加融合。若未来引入特征拼接，应先证明端到端速度和 Presence/缺失段不回退，再投入完整训练。

## FPN concatenation extended-training screening

**结论**：将拼接 FPN 候选从 4 epoch 延长到等效 12 epoch 后，连续集几何主指标继续提升，但 Presence、最长缺失/恢复和最弱来源组回退，不能晋升。

**证据**：同一 `b0d246da...` manifest、生产 MobileNetV3-FPN warm-start、seed `20260915` 和既有后处理协议下，独立 test centerline F1@8 / Presence F1 / 负图平均误检为 `0.898322 / 0.982456 / 14.273 px`。完整 `1Ayoyo_consecutive` 十组、`1.125x`、颜色/亮脊/时序协议下，候选 pooled F1@8 `0.843023`，最弱来源组 `0.672963`，Presence F1 `0.987356`（TP/FP/FN=`898/2/21`），最长缺失/恢复 `7/7`，Chamfer/HD95 `10.492/42.329 px`；生产对应为 `0.818297/0.638996/0.994530/2/2/15.621/57.470 px`。阈值 `0.5` 时 pooled F1@8 仅 `0.844004`，安全指标仍未恢复。

**适用范围**：当前 reviewed split、`1Ayoyo_consecutive` 927 帧、MobileNetV3-FPN 32 通道和 12 epoch 等效训练；未改变评估类别口径。

**后续建议**：不保留拼接融合专用分支。后续 FPN 结构改动应同时满足 Presence、最长缺失/恢复和最弱来源护栏，而非仅追求 pooled 几何指标。

## FPN concatenation logit-distillation screening

**结论**：用生产 FPN 的语义 logits 对拼接 FPN 做训练期蒸馏，短训独立 test 主指标低于生产，未显示蒸馏能够修复拼接结构的安全回退。

**证据**：同一 `b0d246da...` manifest、生产权重 warm-start、seed `20260916`、4 epoch 和既有阈值扫描协议下，蒸馏权重 `0.2` 的独立 test centerline F1@8 / Presence F1 / 负图平均误检为 `0.884946 / 0.982456 / 44.0 px`，生产同协议对照为 `0.888527 / 0.979167 / 60.364 px`。候选低于生产主指标，未进入连续集评估。

**适用范围**：当前 reviewed split、MobileNetV3-FPN 拼接解码器、生产 logit 教师、4 epoch warm-start 筛选；未改变连续集评估协议。

**后续建议**：保持单一生产 FPN 路径，不在默认训练脚本中保留蒸馏参数或教师模型流程。若未来重新评估蒸馏，应先通过独立 test 的几何主指标和 Presence 筛选。

## FPN channel-gate screening

**结论**：在每个 FPN 解码尺度加入零初始化的轻量通道重标定，短训独立 test 的中心线略有提升，但 Presence 明显回退，未形成可部署收益。

**证据**：同一 `b0d246da...` manifest、生产 MobileNetV3-FPN warm-start、seed `20260917` 和 4 epoch 筛选下，ECA 候选独立 test centerline F1@8 / Presence F1 / 负图平均误检为 `0.890314 / 0.972222 / 66.182 px`，生产同协议为 `0.888527 / 0.979167 / 60.364 px`；候选验证最佳 epoch 为 3、阈值 `0.995`。中心线增益不足以抵消 `7` 个误检帧和 Presence 回退。

**适用范围**：当前 `798/151/152` reviewed split、MobileNetV3-FPN 32 通道和 4 epoch warm-start；未进入连续集评估。

**后续建议**：保持原始 FPN 解码器，不保留通道门控专用分支；若未来数据规模扩大，应先在独立 test 同时验证 Presence 和负图误检，再考虑通道重标定。

## FPN spatial-gate screening

**结论**：在各 FPN 尺度加入零初始化的空间门控没有改善独立 test 的综合安全指标，调高阈值也不能恢复生产水平。

**证据**：同一 manifest、生产权重 warm-start、seed `20260918` 和 4 epoch 筛选下，空间门控候选在阈值 `0.9204` 的独立 test centerline F1@8 / Presence F1 / 负图平均误检为 `0.889734 / 0.979167 / 82.545 px`；阈值 `0.995` 时为 `0.888291 / 0.979167 / 65.455 px`。生产对照为 `0.888527 / 0.979167 / 60.364 px`，候选误检仍更高且中心线不升。

**适用范围**：当前 `798/151/152` reviewed split、MobileNetV3-FPN 32 通道和 4 epoch warm-start；未进入连续集评估。

**后续建议**：不保留空间门控分支；后续结构实验先以负图误检和独立 test F1 作为连续集投入门槛。

## FPN directional-kernel screening

**结论**：在 FPN 解码特征上加入零初始化的横/竖长核残差，独立 test 的误检有所下降，但中心线主指标和 Presence 仍低于生产，未达到候选资格。

**证据**：同一 manifest、生产权重 warm-start、seed `20260919` 和 4 epoch 筛选下，方向残差候选验证最佳 epoch 为 3、阈值 `0.9701`；独立 test centerline F1@8 / Presence F1 / 负图平均误检为 `0.888676 / 0.975610 / 51.455 px`，阈值 `0.92` 时为 `0.888389 / 0.975610 / 53.727 px`。生产同协议为 `0.888527 / 0.979167 / 60.364 px`，候选存在性回退且主指标无可靠提升。

**适用范围**：当前 `798/151/152` reviewed split、MobileNetV3-FPN 32 通道和 4 epoch warm-start；未进入连续集评估。

**后续建议**：保持标准 FPN 融合，不保留方向卷积专用分支；若未来有更密集的细绳中心线标注，可在来源隔离 test 上重新验证方向监督。

## Production 低置信区域 Recall Gate screening

**结论**：为 production 语义 logits 增加 utility head，并按样本预测决定是否开启低阈值 Recall 分支，未能形成可复现收益；当前 Gate 仅作为实验接口保留，不替换生产路径。

**证据**：同一 `b0d246da...` manifest、`semantic_quality_warm_e4` checkpoint、`low=0.35/high=0.9204` 和 `quality_cutoff=-0.454262` 下，独立 test 的 Gate centerline F1@8 为 `0.888934`，Presence F1 `0.982578`，负图平均误检 `55.0 px`；同 checkpoint 不启用 Gate 为 `0.888868/0.982578/55.364 px`，提升仅 `+0.000066`。`1Ayoyo_consecutive` 十组、927 帧同协议回放中，Gate pooled centerline F1@8 `0.817463`，不启用 Gate `0.817514`，当前生产 `0.818297`；Presence F1 均 `0.991790`，最弱来源组 `0.657271`（Gate）对比 `0.656582`（不启用），最长缺失/恢复均 `4/4`。

**适用范围**：当前 MobileNetV3-FPN、4 epoch warm-start quality head、`1Ayoyo_dataset` reviewed test 和 `1Ayoyo_consecutive` 10 组/927 帧；该实现是帧级 utility gate，不能外推为已验证的组件级局部分类器。

**后续建议**：不将 `mobilenet_v3_fpn_quality` 或自动低阈值策略写入默认权重/配置。若要继续回答组件级问题，应构造 production 低置信连通区域的独立正负监督，并在来源隔离连续集上验证区域级 Gate 的收益与误检护栏。

## 当前 manifest 的悠悠球检测与方向重训

**结论**：在新增来源后的 `1Ayoyo_dataset` 上从 foundation 权重重训，三分类方向视图配合 dropout `0.1` 已通过同集静态 test 与连续集护栏并替换方向生产权重；检测短训候选仍不晋升。

**证据**：检测 YOLO11s、`imgsz=1024`、12 epoch、seed `20260911` 的 native test mAP50-95 / mAP50 / recall 为 `0.547847 / 0.863774 / 0.776447`。三分类方向 YOLO11n、20 epoch、320px、dropout `0.1` 的同集外部 test Top-1 / Macro Recall 为 `0.949721 / 0.949252`，三类召回 `horizontal=0.894737`、`normal=0.953020`、`not_applicable=1.0`；在 `1Ayoyo_consecutive` 927 帧时序回放中 pooled Accuracy / Macro Recall 为 `0.979504 / 0.897801`，预测切换为 `6`，弱来源“邬聪聪” Accuracy `0.909091`，所有来源组不低于生产方向模型。

**适用范围**：当前 `yoyo_unified_579b879a66ce` manifest、YOLO11s/YOLO11n foundation、固定输入尺寸和现有方向时序过滤；检测实验为 12 epoch 快速训练，不外推到更长日程。

**后续建议**：检测保留现有生产权重；若继续投入，应先延长检测训练日程并在连续集完成位置召回/IoU 护栏评估。方向训练保留三分类 view 与 dropout `0.1`，数据 view 或时序协议变化后重新检查 `not_applicable` 召回和最弱来源护栏。

## 检测 warm-start 与短缺口补全筛选

**结论**：当前 manifest 上从 foundation 训练最佳点 warm-start 能改善 native test 和候选框几何，但连续集只有在低阈值加短缺口补全后才接近生产 Presence；弱来源和最长缺失段仍不满足晋升护栏，因此不替换生产检测权重。

**证据**：YOLO11s、`imgsz=1024`、seed `20260911` 的 12 epoch foundation run native test mAP50-95 为 `0.547847`；从该 run 的 best checkpoint 以 AdamW `lr0=1e-4` warm-start 8 epoch 后 native test mAP50-95 提升到 `0.569378`，但仍低于生产 `0.586875`。连续集 raw `conf=0.15` 的候选 Presence F1 / mean IoU / longest missing 为 `0.943570 / 0.834955 / 9`，生产为 `0.977387 / 0.799693 / 7`。候选降到 `conf=0.03` 并对相邻最多 2 个缺口做线性 bbox 补全后，Presence F1=`0.980442`、FP=`3`、mean IoU=`0.812556`，但最长缺失仍为 `8`；弱来源 `邬聪聪` F1=`0.8645`，低于生产 `0.9444`。补全至 8 帧可把最长缺失降到 `6`、F1=`0.986369`，但 FP 增至 `13`、mean IoU 降到 `0.794779`，不满足误检与几何护栏。

**适用范围**：当前 `yoyo_unified_579b879a66ce` manifest、YOLO11s foundation/warm-start、`1Ayoyo_consecutive` 927 帧、固定 `imgsz=1024` 和现有 bbox 评估；缺口补全仅为离线追踪筛选，不改变模型结构。

**后续建议**：保留生产检测权重和现有 rescue 路径；继续优化前应优先补充弱来源中“远距离小球、暗背景/高对比墙面”样本并重新训练，避免用更长时间补全掩盖真实漏检。显式 optimizer/lr 参数已加入检测训练入口，便于后续可复现 warm-start 消融。

## 检测小目标尺度增强筛选

**结论**：在当前 manifest、同一 YOLO11s foundation 初始化和 12 epoch 日程下，将训练 `scale` 从 `0.15` 提高到 `0.40` 未改善连续集召回；静态 test 略优于 foundation 但仍低于生产，弱来源明显回退。

**证据**：候选 `yoyo_unified_579b879a66ce_detection_best_current579-scale40-e12` 的 native test mAP50-95 / mAP50 / recall 为 `0.572 / 0.887 / 0.817`，生产 mAP50-95 为 `0.586875`。连续集原始 `conf=0.15` 下 Presence F1 / mean IoU / 最长缺失为 `0.9202 / 0.8378 / 11`，弱来源 `邬聪聪` F1 为 `0.5424`；`conf=0.03` 时分别为 `0.9620 / 0.8178 / 6`，弱来源 F1 为 `0.8138`，仍低于生产 `0.9774 / 0.7997 / 7` 及弱来源 `0.9444`。

**适用范围**：当前 `yoyo_unified_579b879a66ce` manifest、YOLO11s、`imgsz=1024`、seed `20260911` 和 `1Ayoyo_consecutive` 927 帧 bbox 回放；只改变尺度增强，未改变网络结构或追踪后处理。

**后续建议**：默认保留 `scale=0.15`；若继续处理远距离小球，应优先扩充并平衡弱来源样本，再单独验证增强组合，避免仅靠扩大尺度扰动牺牲来源组召回。

## 检测训练日程与弱来源重采样筛选

**结论**：将当前 manifest 的检测训练从 12 epoch 延长到 20 epoch 可以把独立 test 拉近生产，但连续集弱来源仍明显回退；低学习率微调和单来源重采样没有形成额外收益。

**证据**：YOLO11s、`imgsz=1024`、seed `20260911`、`mosaic=0`、`scale=0.15` 的 20 epoch 候选 native test mAP50-95 为 `0.586661`，接近但低于生产 `0.586875`。同一候选的轻量连续框回放在 `conf=0.15` 下 Presence F1 / mean IoU / 最长缺失为 `0.9472 / 0.8215 / 7`，弱来源 `邬聪聪` F1 为 `0.7059`；降到 `conf=0.03` 后为 `0.9720 / 0.8115 / 4`，弱来源 F1 为 `0.8344`，仍未达到生产的连续集 Presence 与弱来源护栏。以该 checkpoint 做 AdamW `lr0=1e-4`、8 epoch 微调后 native test mAP50-95 降为 `0.581825`；将训练列表中的 `邬聪聪` 样本重复 6 倍后为 `0.583536`。

**适用范围**：当前 `yoyo_unified_579b879a66ce` manifest、YOLO11s、固定 `1024` 输入和 `1Ayoyo_consecutive` 927 帧框回放；弱来源重采样只作用于 train 列表，val/test 与推理后处理未改变。

**后续建议**：不晋升这些候选，也不把低学习率微调或单来源重复写入默认流程。若继续训练策略实验，应先增加真实弱域标注覆盖，再以独立 test 和完整连续集弱来源共同筛选。

## 输入尺度与权重平均筛选

**结论**：提高训练输入到 `1280` 在同尺度 test 上只有边际收益，固定生产推理尺寸 `1024` 时反而回退；对当前短训 checkpoint 做简单权重平均也没有收益。

**证据**：YOLO11s、`imgsz=1280`、batch 4、8 epoch 的候选在 `1280` 推理下 mAP50-95 为 `0.587759`，但单图 inference 为 `11.1 ms`；同一权重固定 `1024` 推理时 mAP50-95 仅 `0.560343`、inference `8.0 ms`。将 20 epoch 与 mosaic 候选按权重平均（alpha `0.25/0.50/0.75`）的固定 `1024` test mAP50-95 分别为 `0.545177 / 0.551746 / 0.559560`，均低于两个输入模型。

**适用范围**：当前 `yoyo_unified_579b879a66ce` manifest、YOLO11s、固定 test split 和 RTX 4070 Laptop；权重平均只作为离线筛选，不改变部署模型。

**后续建议**：保留 `1024` 推理协议，不采用 1280 输入或简单权重平均；若未来部署允许更高延迟，应重新按完整连续集和端到端 FPS 评估高分辨率模型。

## 弱域居中放大 crop 筛选

**结论**：对弱来源训练图围绕标注框生成居中放大 crop 并重复采样，会损害来源隔离 test 的泛化，不能替代真实弱域样本扩充。

**证据**：当前 manifest、YOLO11s、`imgsz=1024`、12 epoch 下加入 11 张 `邬聪聪` 居中放大 crop（有效训练列表 860 条）后，native test mAP50-95 为 `0.570115`，低于同协议 foundation 12 epoch `0.547847` 的延长训练结果与生产 `0.586875`，且也低于 20 epoch 基线 `0.586661`。

**适用范围**：当前 `yoyo_unified_579b879a66ce` manifest、弱来源 11 张训练图和固定 test split；crop 仅用于训练列表，验证/测试原图未改变。

**后续建议**：不保留 crop 生成流程；后续弱域改进优先采集并审核更多真实远距离/低对比帧，再重新训练验证。

## 方向 dropout 单因素筛选

**结论**：在当前三分类 ROI view、YOLO11n-cls、320px、batch 32、20 epoch 和固定 seed 下，将 dropout 从 `0.2` 降到 `0.1` 同时改善独立 test 与连续集时序指标；降到 `0.0` 虽提高静态 Top-1，但 `not_applicable` 和 pooled Macro Recall 回退，不能部署。

**证据**：dropout `0.1` 的同集外部 test Top-1 / Macro Recall 为 `0.949721 / 0.949252`，连续集 Accuracy / Macro Recall / 预测切换数为 `0.979504 / 0.897801 / 6`，弱来源“邬聪聪” Accuracy `0.909091`，10 个来源组逐组不低于生产模型。dropout `0.0` 的连续集为 `0.916936 / 0.786892 / 12`，`not_applicable` Recall `0.583333`。

**适用范围**：当前 `yoyo_unified_579b879a66ce` 数据与三分类 ROI view、YOLO11n-cls、320px、batch 32、20 epoch、RTX 4070；连续集为 `1Ayoyo_consecutive` 10 组、927 帧和现有 5/25 FPS 时序过滤。

**后续建议**：保留 dropout `0.1` 作为默认训练设置和当前方向权重；数据 view、模型容量或时序协议变化后，应重新检查 `not_applicable` 召回及最弱来源护栏。
