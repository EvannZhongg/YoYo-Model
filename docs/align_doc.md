# 未来架构与当前实现的对齐说明

本文只回答两个问题：未来方案在当前代码和数据上是否可行，以及训练前必须补齐哪些数据结构。模型架构本身见 [`future_architecture.md`](future_architecture.md)，本文不重复展开。

## 1. 结论

方案方向可行，但不能直接用当前数据训练完整 Tokenizer。当前仓库已经有悠悠球检测、细绳分割、方向分类、短时跟踪和评分工作台；缺口在于这些结果还没有以统一视频时间轴组成可追溯的连续序列。

当前评分监督只采用你的单人 `yoyo_score_annotation_v2` 标注。多裁判、多标注者和跨规则校准属于未来扩展，不是当前训练前提。视频按单视角处理，不要求多视角同步。

## 2. 当前能力核验

| 组件 | 当前事实 | 对未来方案的含义 |
| --- | --- | --- |
| 感知训练 | `training_v3/prepare_dataset.py` 生成 `yoyo_multitask_dataset_v6`，包含悠悠球、细绳和方向任务；按 `source_group` 隔离并记录 manifest/hash | 可作为上游感知基线，不能直接产生元素序列 |
| 稀疏图像集 | `datasets/1Ayoyo_dataset` 约 722 个样本、147 个来源组 | 适合检测/分割，不足以监督连续边界 |
| 连续集 | `datasets/1Ayoyo_consecutive` 约 856 帧、7 个来源组 | 可验证跟踪和时序指标，不等于完整表演语料 |
| Tracker | `video_tracking/tracker.py` 可输出悠悠球框、字符串几何、方向及质量信息 | 需要封装为版本化序列，并区分模型预测与人工真值 |
| 评分标注 | `workbench/score_annotation.py` 校验 `yoyo_score_annotation_v2`；每条事件含 evidence 起止、anchor、帧索引和 `score_delta` | 每条 evidence 区间都是真实 Element/Event；可直接作为第一版边界和评分监督 |
| 阶段标记 | `serve_receive_events`、场景区间和排除区间独立于元素事件 | 只能作为阶段上下文或忽略区间，不能自动当作技术元素 |
| 分值语义 | `score_delta` 是单独评价该 Element 时的离散观测值；0 不是独立类别 | Score Head 学习 raw continuous score；重复、Combo 和整场规则后置到关系/规则层 |

## 3. 需要补齐的数据结构

### P0：连接与审计

- 为评分视频补齐 `source_video_sha256`、`source_group`、fps、分辨率和 division。
- 统一秒值与帧索引，保留原始 event ID、anchor、FPS 假设和修订版本。
- 建立 manifest，把视频、帧、感知运行和评分文档通过 hash/relative path 连接。
- 为每条感知记录标注 `source`（`human`、`model`、`derived`、`unknown`）和质量状态；缺失资料不能作为负样本。

### P1：连续单视角感知

- 为完整表演提供连续或关键窗口的悠悠球 bbox、可见字符串 polyline、方向和遮挡/模糊状态。
- 同时保留模型预测和人工修订，记录传播、插值、低置信补救等 provenance。
- 训练、验证、测试按视频和 `source_group` 切分，连续帧不得跨 split。

### P2：元素对齐

- 从 score v2 生成不覆盖原始文件的 `element_annotation_v1`，增加内部 `element_id`、阶段窗口、边界不确定度和感知关联。
- 对每个 Element/Event 补充完成、失败、中断、跨遮挡和 0/1/多对多关联；动作名称不是前置条件。
- 保留 `score_delta` 原值；它是连续元素分值估计的离散观测。只有在已完整审阅且元素标注穷举的窗口内，evidence 区间外帧才是 Background/Non-element；其他未覆盖区域必须忽略，不能作为边界负例。
- 相邻 evidence 允许短暂重叠；训练时将其转换为共享 boundary uncertainty window，原始人工 start/end 不修改。
- Score Head 只学习单个 Element 的 raw continuous score estimate；不得把重复、Combo 或“之前是否出现过”混入该监督。最终 credited score 由关系识别和 `RuleAggregator(rule_version)` 决定。
- serve/receive anchor 可按时间顺序构造候选 `Combo_k=[serve_k,receive_k]`；数量不一致时保留孤立 anchor，不强行配对。

数据量不是当前阻塞因素；关键是每条记录都能回溯到同一单视角视频、同一帧时间轴和同一版本的感知运行。

## 4. 可行性与阶段门槛

1. **对齐基线**：视频 hash 闭环，能复现现有连续跟踪指标。
2. **Frame State**：缺失和遮挡显式编码，输入字段按 `future_architecture.md` 的评分模型契约冻结。
3. **边界模型**：只在完整审阅窗口使用 Background 负例；对相邻重叠区间使用 soft boundary target，并在 held-out 完整视频上报告 ±2/±5/±10 帧 boundary F1、事件召回、背景误报、过切/欠切和跨遮挡片段召回。
4. **Element Token**：每个 evidence 区间都生成 Token，Background 不生成 Element Token；验证跨视频稳定性、边界扰动鲁棒性和连续分值相关性。
5. **元素评分**：候选区间已被视为 Element，Score Head 使用连续回归（优先测试 Huber，可加 ranking loss）预测 raw continuous score estimate；不确定性留作未来扩展，输出必须引用原始 event ID。
6. **Element Relation/Transformer**：待元素区间、感知对齐和 Token 检索达标后再训练，第一版输出 `sequence_context` embedding、单元素 `element_repeat_relation` 和 `combo_relation`；不新增没有 GT 的 `transition_quality` 头，由 `RuleAggregator` 按规则版本产生最终 credited score。

所有 run 需保存数据 manifest hash、权重 hash、输入契约版本、参数和初始化 lineage。至少保留一个不参与调参的外部单视角视频集合；模型传播、插值和增强结果不得回写人工真值。

## 5. MVP

第一版交付应包括：带 provenance 的逐帧 Frame State；在 held-out、完整审阅的单视角视频窗口上自动预测 Element 区间并与人工 evidence 比较，同时提供边界修正工具；相似元素检索；映射回 score v2 的可审计分项结果；以及对未审阅、未知、遮挡、证据不足片段的人工复核提示。自动整场评分和多裁判一致性属于后续阶段。
