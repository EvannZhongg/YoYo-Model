# 悠悠球比赛自动评分：Element Tokenizer 未来架构

本文保留“运动元素 Token”这一长期方向，并按当前仓库的数据和代码能力重新组织为可落地的技术路线。目标是从单视角视频中的悠悠球、绳线和上下文状态学习可复用的元素表示，再把元素证据映射到可审计的评分事件。

## 1. 目标与边界

目标数据流：

```text
视频（单视角）
  -> 逐帧感知（悠悠球、绳线、方向、姿态）
  -> Frame State 序列
  -> 元素区间（人工事件或边界模型）
  -> Element Segment / Element Token
  -> 元素质量与评分事件
  -> Combo 与整场评分
```

Element Token 是模型内部的连续向量，不等同于动作名称。未知或自创动作可以作为未命名元素参与表示学习。`score_delta` 是特定规则和当前标注下的记分事件，不能直接解释为跨规则的客观技术价值；整场分数也不能默认等于元素分数简单求和。

当前系统已经具备悠悠球检测、细绳分割、方向分类、短时跟踪和 `yoyo_score_annotation_v2` 评分事件。当前数据仍以稀疏图像和连续评估片段为主，不能直接视为完整的元素语料。实现 Tokenizer 前，必须先完成视频 hash、时间轴和逐帧感知运行的对齐。

## 2. Element/Event 定义

当前标注已经定义了元素事件，不需要再由 Score Head 判断候选是否“存在”。对每一条 score v2 事件，定义：

\[
E_i=(start_i,end_i,score\_delta_i)
\]

其中 `[start_i, end_i]` 是该事件的 evidence 区间，`score_delta_i` 是当前标注下的离散观测分值：

```text
score_delta > 0     正向分值
score_delta < 0     负向分值
score_delta = 0     数值为零的已标注事件
```

只有在已完整审阅、确认元素标注穷举的时间窗口内，不属于任何 `evidence_start/end` 区间的帧，才可标为 `Background / Non-element`。未完整审阅区域的未覆盖帧必须标为 `unlabeled/ignore`，不能作为 Background 负例。因此：

\[
\boxed{\text{Zero-score Element} \neq \text{Background}}
\]

`score_delta=0` 的事件同样生成 `z_i` 并参与事件序列；它不构成独立运动类别。只有审阅穷举窗口内的区间外帧才用于边界检测的负例或背景建模。

相邻事件允许存在短暂人工重叠。若重叠来自 Combo 中无法确定唯一切点，训练预处理将其解释为共享的 boundary uncertainty window，而不是两个 Element 同时发生；原始人工 `start/end` 不修改。

## 3. Frame State

每一帧的表示应包含可见证据和不确定性，而不是强行恢复完整绳线：

```text
YoYo: bbox、中心、track_id、置信度、短窗口运动特征
String: 可见 fragment 的 polyline、长度、方向、曲率、几何置信度
Relations: fragment 距离、端点邻近、crossing/连接候选（软关系）
Pose: 人体框、关键点/手腕、置信度、选人策略
Orientation: normal / horizontal / not_applicable / unknown 及概率
Quality: occluded、blur、edge_clipped、out_of_frame、unknown 等状态
```

建议使用独立的 `motion_sequence_v1` 序列文件，不改造现有 `agent_yoyo_string_annotation_v5` 图像标签。每个视频或片段至少保存：

```json
{
  "schema_version": "motion_sequence_v1",
  "video": {
    "source_video_sha256": "...",
    "source_group": "...",
    "fps": 30.0,
    "width": 3840,
    "height": 2160,
    "division": "1A"
  },
  "frames": [
    {
      "frame_index": 158,
      "timestamp_s": 5.267,
      "yoyo": {"bbox": [120.0, 80.0, 220.0, 180.0], "track_id": 0, "confidence": 0.91, "source": "model"},
      "string_fragments": [
        {"fragment_id": "f0", "polyline_pixel": [[120.0, 180.0], [220.0, 260.0]], "visibility": "partial", "confidence": 0.74}
      ],
      "pose": {"people": [{"bbox": [100.0, 50.0, 400.0, 900.0], "keypoints": [{"index": 9, "x": 180.0, "y": 300.0, "confidence": 0.88}]}], "selected_person": 0, "source": "model", "model_version": "pose_v1"},
      "orientation": {"label": "normal", "prob": {"normal": 0.8, "horizontal": 0.1, "unknown": 0.1}},
      "quality": {"occluded": false, "blur": false, "bad_case": []}
    }
  ]
}
```

约束：`frame_index`、`timestamp_s` 和 `source_video_sha256` 必须共同用于连接数据；bbox 和 polyline 使用原始像素坐标；`source` 区分 `human`、`model`、`derived`、`unknown`；未检测到要区分不可见、遮挡、出画和未知；速度/加速度记录窗口、平滑方法和单位，不能把像素差解释为物理速度。姿态数据应记录模型版本、置信度和缺失原因，供评分模型输入审计。

上述序列是 `Observed Frame State`，不是无噪声的真实状态。Frame Encoder 与 Temporal Encoder 从观测窗口得到 `Latent Motion State`：

\[
O_t \rightarrow h_t=F(O_{t-k:t+k})
\]

`h_t` 用于边界检测和元素编码，使模型可以吸收 fragment、姿态和检测噪声，而不要求先恢复完整绳线拓扑。

## 4. 模型架构

建议采用分阶段、可替换的模块化结构：

```text
Frame State
  -> modality encoders（YoYo / String / Orientation / Pose）
  -> Frame Encoder（Set 或 Graph + Temporal Encoder）
  -> h_1 ... h_T
       |                         |
       +-> Boundary Head         +-> Element Segment Encoder
              |                         |
              +-> [start, end]          +-> z_i
                                              |
                               +--------------+--------------+
                               |                             |
                         Element Score Head            Element Transformer
                               |                   input: z_1...z_n + time/context
                           q_hat_i                          |
                               |                       sequence_context g_i
                               +--------------+--------------+
                                              |
                                      分解式整场评分
```

Frame Encoder 将每帧的多段 fragment 作为集合编码，再用短时 Temporal Encoder 建模运动变化；不要求先恢复完整绳线。Boundary Head 学习事件区间相对于 Background 的起止概率和边界置信度，解码器负责时长、相邻关系和候选区间约束。Element Segment Encoder 对每个 `[start_i,end_i]` 区间池化 `h_t` 形成 `z_i`，所有已标注事件使用同一表示空间。Element Score Head 不再预测元素存在性或分值类别，只回归当前评分体系下的连续元素分值估计 `q_hat_i`。Element Transformer 的主输入是元素 Token 序列 `[z_1, ..., z_n]` 与时间/持续时间/阶段上下文，不把 `q_hat_i` 作为核心输入；它输出 `sequence_context=g_i` 和可解释的元素关系，再由规则聚合器结合 `q_hat_i` 处理最终账本。

姿态是未来评分模型的输入分支，提供手腕、人体区域和悠悠球/身体关系。所有姿态特征都带置信度和缺失掩码，Pose 是辅助 Context 分支。

## 5. Element Tokenizer

Tokenizer 拆成三个可独立评估的模块。

### 5.1 Boundary Detector

`yoyo_score_annotation_v2` 已提供真实事件的 `evidence_start/end`、anchor、秒值和帧索引，可作为第一版边界监督。仅在完整审阅且标注穷举的窗口内，区间外帧才构成 Background/Non-element 负例；其他未覆盖帧标为 `unlabeled/ignore`。`serve_receive_events`、场景区间和排除区间是上下文或不可训练区间，不是自动生成的技术元素。

模型输入 Frame Embedding 序列，输出事件起止概率 `P(start)`、`P(end)` 和 `P(element)`。其中 `P(element)` 表示当前帧位于某个已标注 Element 区间内的概率，`P(background)=1-P(element)`；未穷举审阅区域的帧不参与这两个概率的监督。解码时使用 `P(element)` 过滤起止不一致的 proposal，并加入最小/最大长度、相邻元素约束；允许生成“候选修正”供人工确认，但不得静默覆盖原始事件。边界 target 对明确切点使用尖峰或三角/Gaussian 分布，对人工重叠使用 uncertainty window；评估使用 ±2、±5、±10 帧 boundary F1、held-out 事件召回、背景误报、过切/欠切率和跨遮挡片段召回。

### 5.2 Element Encoder

对边界区间内的 Frame State 进行 Temporal Transformer、Temporal CNN 或 Attention Pooling，输出 `z_i`。训练时从 GT `[s_i,e_i]` 随机采样边界扰动 `[s_i+Δ_s,e_i+Δ_e]`，模拟 Boundary Head 的真实误差，并对同一完整 Element 的多种观测保持表示稳定，以减少 Boundary→Encoder 的级联暴露差异。编码器应对低置信度 fragment、姿态观测缺失和边界扰动保持稳定，并保留输入运行的 manifest/hash。

### 5.3 Element Representation

先采用 masked Frame State reconstruction、时间对比学习和相似元素检索等自监督任务，再验证 embedding 是否对同一运动结构的增强视图保持稳定、保留绳线—悠悠球运动结构，并与 Background/Non-element 分离。评分分值通过独立 Score Head 和 ranking objective 施加，不要求整个 embedding 空间仅按分值聚类；分值相近但结构不同的元素仍应保持可区分。`score_delta=0` 的事件不是背景负例，必须保留其 `z_i` 和序列位置。正样本应来自同一完整 Element 的不同观测视图，例如 boundary jitter、fragment dropout、遮挡模拟、检测噪声、采样率变化、坐标抖动或置信度扰动；不要把同一元素内部不完整的不同时间裁剪强制作为正样本。不建立闭集动作分类器，也不要把 embedding 相似度单独当作技术价值证明。

## 6. 评分模型与序列模型

元素评分头只处理已确定的 Element Segment，学习当前评分体系下的 Continuous Element Score Estimate：

\[
\hat q_i=f(z_i),\qquad \hat q_i\in\mathbb{R}
\]

当前整数 `score_delta` 是连续分值估计的离散观测，不要求模型输出整数。第一阶段可使用 Huber 回归损失，并在有可靠相对关系时加入 ranking loss：

\[
L=L_{Huber}(\hat q_i,score\_delta_i)+\lambda L_{rank}
\]

规则允许的取整、截断、重复惩罚和展示格式放在 Rule Layer，不放入 Element Representation 或 Score Head。模型输出范围第一版保持连续；若训练稳定性或规则边界需要，再采用显式的连续范围约束。

不确定性头 `(\mu_i,\sigma_i)` 暂列为未来扩展，不作为第一版实现。未来有多裁判数据时，可用评分均值监督期望分数，并用裁判分歧监督不确定性，而不是增加 `zero` 类别。

当前监督以你的单人标注为准。需要补充完成/失败/中断、遮挡跨越和事件关联置信度后，才适合训练更稳定的质量或总分模型。裁判 ID、规则版本、多裁判事件和多标注者一致性属于未来扩展，不是当前训练前提。输出应保留原始 event ID、连续分值估计和不可评分原因。

Element Transformer 的主输入是元素 Token 序列 `[z_1, ..., z_n]`，以及时间顺序、持续时间和阶段上下文；不把 `q_hat_i` 作为核心输入，避免 Score Head 输出造成 shortcut。第一版只输出三类结果：逐元素 `sequence_context=g_i` embedding、`element_repeat_relation` 和 `combo_relation`。其中关系输出描述元素之间的结构关系，不要求额外的 transition quality GT。训练和推理使用相同的 `z_i` 与上下文输入。RuleAggregator 接收 `q_hat_i`、`g_i`、关系结果和显式 `rule_version`，将已定义的重复/Combo 规则转换为 adjustment，再生成最终账本贡献，而不是直接输出不可解释的总分。

Score Head 的输出是元素本身的 raw score，不考虑其之前是否出现、是否重复或属于何种 Combo：

\[
\hat q_i=f(z_i),\qquad c_i=RuleAggregator(\hat q_i,g_i,rule\_version)
\]

例如，元素单独评价得到 `raw_score=2.13`，关系模型识别其 `repeat_of=#2`，规则层才可根据 `rule_version` 将最终 `credited_score` 置为 0。`2.13` 仍是正确的单元素评分估计。重复检测属于 Element-to-Element Relation，应综合 `Sim(z_i,z_j)`、时间顺序、String-YoYo 结构、Pose/Orientation、持续时间和周围元素，不并入 Score Head 回归；这里判断的是单个元素是否重复，不定义“重复 Combo”类别。

若存在 serve/receive anchor，可将相邻的开始/结束标记按时间顺序构造成候选流程窗口：

\[
Combo_k=[serve_k,receive_k]
\]

serve 与 receive 数量不一致时，不强行一一配对；应依据时间顺序、视频边界和标注证据生成可审计的候选窗口，未能配对的 anchor 保留为孤立阶段标记。

## 7. 数据补充与实施阶段

数据 schema、字段映射和当前数据缺口以 [`align_doc.md`](align_doc.md) 为准。架构实施只规定顺序：

1. 先完成视频 hash、时间轴、感知运行和 score v2 的连接审计；
2. 再建立连续单视角 Frame State，并冻结感知输入契约；
3. 以人工元素区间训练 Boundary Head，在 held-out 完整视频上评估自动预测，再提供边界修正工具；
4. 在稳定区间上训练 Element Token 和检索任务；
5. 最后加入元素评分、整场分解和 Element Transformer。

当前使用你的单人标注；多裁判、多标注者和跨规则校准只在未来需要扩展评分一致性时加入。

## 8. 验收与风险控制

各阶段分别报告感知、边界、元素、事件和整场指标，不用单一总分掩盖上游失败。所有 run 保存数据 manifest hash、权重 hash、输入契约版本、参数和初始化 lineage；连续帧不得跨 split，近重复图像按 `image_sha256` 去重；模型传播、插值和增强几何不能回写人工真值。

MVP 只要求：输出带 provenance 的 Frame State；在 held-out 完整单视角视频上自动预测 Element 区间并与人工 evidence 比较，同时提供人工边界修正工具；检索相似历史元素；映射回 score v2 生成可审计分项结果；对未知、遮挡和证据不足片段明确要求人工复核。

长期目标仍是：

```text
连续绳线-悠悠球状态变化 -> 运动元素 Token -> 技术价值
```

但 Element Tokenizer 和 Element Transformer 的训练前提是上述数据契约、对齐流程和外部验证全部成立。