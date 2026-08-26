# 悠悠球比赛自动评分：未来架构可行性与落地路线

本文是对“基于 Element Tokenizer 的运动语言建模方案”的工程化修订版。它描述长期目标，但所有结论都以当前仓库的代码、数据契约和已有标注为准。文中“当前”指本仓库在 2026-08-26 的状态；本文不要求立即修改代码。

## 1. 结论先行

长期方案**方向可行，直接实施仍需数据整合**。当前评分工作台已经在标注元素：每个元素有顺序、证据起止时间和 anchor，可带正分、负分或无分；文档还记录发球/收球时间范围。下一步不是重新发明元素边界，而是把这些元素标注与逐帧感知结果绑定成同一套、可追溯的连续序列，再训练 Token/Transformer。

当前可复用的基础已经足够好：

* `training_v3/prepare_dataset.py` 已形成 `yoyo_multitask_dataset_v6`，包含悠悠球检测、细绳分割、粗粒度方向分类三个任务；数据按 `source_group` 隔离切分，并记录 manifest/hash。
* `datasets/1Ayoyo_dataset` 当前约有 722 个样本、147 个来源组；这适合感知模型，不足以监督连续元素分词。
* `datasets/1Ayoyo_consecutive` 有 856 个连续帧、7 个来源组，主要用于跟踪/时序评估，不是元素级训练集。
* `video_tracking/tracker.py` 已输出每帧的悠悠球框、字符串几何、方向预测和质量/异常信息；`video_tracking/sequence_metrics.py` 已支持连续序列评估。
* `workbench/score_annotation.py` 已保存 `yoyo_score_annotation_v2`：元素顺序、证据区间、anchor、正负分值并可表达无分元素、场景区间、排除区间；`serve_receive_events` 记录发球/收球边界及其证据窗口。

主要缺口：

1. 元素区间已有，但尚未与逐帧感知记录建立统一的 `sequence_id/element_id` 绑定。当前 v2 用 `positive` + `score_delta=0` 表达无分元素（现有 4 个评分文件中可见 37 条），规范化视图应把它提升为显式 `score_status=unscored`。
2. 元素区间、发球/收球区间、场景/排除区间需要统一到同一时间轴，并处理相邻、重叠和边界不确定度。
3. `score_delta` 是裁判记分结果，不等于可跨规则版本比较的客观技术价值；`action_name` 大多数为空。
4. 字符串标注是稀疏 anchor 或小段连续帧，不能直接构成全视频的 Frame State 序列。
5. 当前方向标签是 `normal/horizontal/not_applicable`，不是稳定的运动语义；姿态模型默认关闭且仅为运行时 review 元数据。

当前感知训练视图统一使用 `agent_yoyo_string_annotation_v5`；现有训练任务不读取 pose/hand 字段，但会保留规范标签中的其他信息。姿态信息由独立人体姿态模型在运行时产出，再与悠悠球/绳线状态按帧汇合。

因此，文档后续将 Element Token 定义为**模型内部的可复用表示**，而不是现在就要求一个固定 Trick 词典；同时保留可选的人工语义标签，用于评估和解释。

## 2. 当前系统基线与未来方案的对应关系

| 未来模块 | 当前实现/数据 | 可行性 | 必须补的工作 |
| --- | --- | --- | --- |
| String/YoYo Perception | YOLO 悠悠球检测、细绳语义分割、字符串中心线增强与短时光流传播 | 高 | 将预测与人工真值统一为逐帧记录；区分预测、人工确认和未知 |
| String Fragment Set | 标注中的 `string_polylines_pixel`，推理输出的多段中心线 | 中 | 每段增加稳定 ID、可见性、几何置信度；crossing/连接只能先存软关系 |
| YoYo State | bbox、track_id、方向、时序跟踪 | 中高 | 从 bbox/中心轨迹派生速度、加速度、转向；明确像素坐标而非物理速度 |
| Pose Context | 独立 RTMPose 后端已接入 `tracker.py`，当前 `enable_pose=false`；v3 方向训练不消费 pose | 中高（最终必需，当前可选） | 将 pose 作为独立模型输出和版本化输入分支；记录人体/手腕置信度与缺失状态，并做消融和漂移评估 |
| Orientation Context | `training_v3/orientation_view.py` 的三分类与时序滤波 | 中高 | 将 `not_applicable/unknown` 与 trick 场景分开；作为上下文，不当作动作类别 |
| Frame Encoder | 尚无 Frame Embedding 模型 | 中 | 先定义版本化输入张量和缺失值协议，再比较 Set/Graph/Temporal Encoder |
| Boundary Detector | score v2 已有元素证据起止时间和 anchor，带边界不确定性 | 中高 | 做时间轴归一化、边界审计和解码；不要把 serve/receive 或场景区间误当技术元素 |
| Element Encoder/Token | 尚无元素切片、embedding 或对比学习任务 | 中 | 依赖可靠边界和足够多的连续序列；先做自监督/检索基线 |
| Element Semantic/Score | 评分事件 v2 本身就是元素级标注，4 个文档共约 214 条记录，其中 37 条为 `positive + 0` 无分元素；动作名大多为空 | 中（元素级建模），低（直接跨规则回归） | 显式区分 scored/unscored；补充元素语义、完成/失败和多裁判共识 |
| Element Transformer | 尚无元素序列数据 | 低 | 至少先有数百段完整表演、稳定边界和事件关联 |

## 3. 目标数据流（修订版）

目标仍然是：

```text
视频
  -> 逐帧感知（预测 + 置信度 + 可见性）
  -> Frame State 序列
  -> 元素区间（来自 score v2 或边界模型）
  -> Element Segment / Element Token
  -> 元素级质量与评分事件关联
  -> Combo/整场评分
```

重要调整：感知预测不能直接当标签；每条数据必须带 `source`（`human`、`model`、`derived`、`unknown`）和质量状态。任何缺失或遮挡都应显式记录，不能用插值结果伪造字符串拓扑。

### 3.1 Frame State 建议契约

建议新增独立的序列文件（例如 `motion_sequence_v1.jsonl` 或 Parquet），不要把序列字段塞回现有 v5 图像标签。每个视频/片段至少包含：

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
      "yoyo": {"bbox": [x1, y1, x2, y2], "track_id": 0, "confidence": 0.91, "source": "model"},
      "string_fragments": [
        {"fragment_id": "f0", "polyline_pixel": [[x, y], [x, y]], "visibility": "partial", "confidence": 0.74}
      ],
      "pose": {"people": [], "selected_person": null, "source": "pose_model", "confidence": 0.0},
      "orientation": {"label": "normal", "prob": {"normal": 0.8, "horizontal": 0.1, "unknown": 0.1}},
      "quality": {"occluded": false, "blur": false, "bad_case": []}
    }
  ]
}
```

约束：

* `frame_index`、`timestamp_s`、`source_video_sha256` 是主键的一部分；禁止只用图片文件名连接不同数据集。
* bbox、中心线均使用原始像素坐标；派生的归一化值必须注明坐标系。
* `string_fragments` 是当前帧可见证据，不是完整绳线重建。连接概率、crossing、隐藏路径放在单独的软关系字段。
* `pose` 来自独立人体姿态模型，不回写 `agent_yoyo_string_annotation_v5` 图像标签；至少保留人体框、关键点/手腕、每点置信度、选人策略、模型版本和缺失原因。pose 缺失时 Frame Encoder 必须仍可工作。
* 速度和加速度只在轨迹质量足够时计算，并记录窗口长度、平滑方法和单位；不能把像素差直接解释为物理速度。
* 没检测到不等于不存在：至少区分 `not_visible`、`occluded`、`out_of_frame`、`unknown`。

### 3.2 Element/Score 建议契约

现有 `yoyo_score_annotation_v2` 已是元素监督来源。建议先写一个不破坏原始标注的规范化视图（例如 `element_annotation_v1`），把 score v2 元素、发球/收球边界和帧感知运行通过 hash 连接起来：

```json
{
  "schema_version": "element_annotation_v1",
  "source_video_sha256": "...",
  "source_group": "...",
  "elements": [
    {
      "element_id": "e0001",
      "start_frame": 120,
      "end_frame": 158,
      "anchor_frame": 158,
      "boundary_status": "human_confirmed",
      "boundary_uncertainty_frames": 2,
      "visibility": "usable",
      "score_status": "scored",
      "score_delta": 1,
      "score_family": "positive",
      "score_links": [
        {"source_event_id": "...", "association": "direct", "confidence": 0.9}
      ],
      "semantic_labels": {"family": null, "action_name": null, "open_set": true},
      "quality": {"execution": null, "completion": null, "notes": ""}
    }
  ],
  "unscored_intervals": [],
  "annotation_provenance": {"annotator": "...", "revision": 0}
}
```

规范化视图要保留 `source_event_id` 和原始时间值。按当前标注语义，一条 score v2 event 就是一条元素记录；如后续发现复合元素，再显式拆分而不是静默重解释。当前 v2 的正向分值允许 0，现有评分文件中已有 37 条 `positive + score_delta=0`，因此转换时可将其映射为 `score_status=unscored`；但“没有该元素标注”仍必须使用独立的 `unlabeled`，不能与无分元素混淆。原始 v2 通过 `serve_receive_events` 保存发球/收球时间范围；规范化时应为每个元素引用所属阶段窗口，而不是复制或丢弃这些边界。发球/收球是阶段上下文，不要自动当作技术元素。若元素与感知片段无法可靠对齐，使用 `association=unknown`，不要强行分摊分值。

建议的字段映射如下：

| score v2 字段 | 元素视图含义 |
| --- | --- |
| `event_id`、`sequence_index` | 稳定的 `source_event_id` 和元素顺序；新生成的 `element_id` 只作为内部主键 |
| `timing.evidence_start_*` / `evidence_end_*` | 元素起止时间/帧；`anchor_*` 保留为元素关键帧 |
| `label.family`、`label.score_delta` | 正分、负分和分值；`positive + 0` 映射为无分元素；缺失事件/空白文档仍是 `unlabeled` |
| `serve_receive_events` | 发球/收球阶段窗口，供元素添加 `phase` 或窗口引用 |

## 4. 对原方案各模块的可行性判断

### 4.1 Frame Representation：可行，但必须接受稀疏与不确定性

细绳中心线和悠悠球 bbox 已经存在，且当前标注契约明确禁止用隐藏路径生成 segmentation truth。因此，Fragment Set + YoYo State 的表示是兼容现有系统的。第一版不应追求完整拓扑恢复，建议使用：

* 多段 polyline 的几何编码（采样点、长度、方向、曲率）；
* fragment 间的距离、最近端点、交叉候选和时间一致性；
* bbox 中心/尺寸、track_id、短窗口速度与加速度；
* 可见性、遮挡、模糊、edge-clipped 等质量 token。

Pose 是最终目标中的独立帧数据分支；当前 v3 方向任务不消费 pose 字段。仓库已有 `rtmpose_backend.py` 和 tracker 集成，但配置默认关闭。建议先以模型输出的手腕/人体框和置信度作为上下文，再逐步加入手-身体、悠悠球-身体、绳线-身体区域关系；所有关系都必须允许缺失，不能把低置信 pose 当作几何真值。

### 4.2 Element Boundary Detector：已有监督，但要先做规范化

score v2 的元素记录包含 `evidence_start/end`、anchor、帧索引和 `boundary_uncertainty_allowed`；按当前标注约定，它们就是元素的时间范围。`serve_receive_events` 则是发球/收球阶段标记，不能与元素事件混为一类。训练前需要审计相邻/重叠区间、秒到帧的 FPS 舍入、无分元素和排除区间，而不是重新假设没有边界标签。

可行的过渡方案：

1. 从 score v2 导出元素区间，保留原始秒值、FPS 假设、anchor source 和不确定度；不要覆盖原始标注。
2. 单独标记 `serve`、`freestyle`、`receive`、`non_element`、排除和未标注区间。
3. 用 start/end 概率 + 最小/最大元素时长 + 相邻元素约束进行解码；模型可以修正边界，但不能默默改变人工区间。
4. 评估采用 boundary F1（容忍 ±2、±5、±10 帧）、过切/欠切率，并与 score v2 区间基线比较；对无分元素单独报告召回。

### 4.3 Element Token：可以先自监督，不能直接宣称有技术价值

不要求固定 Trick 名称是正确方向，但 embedding 的“相似”必须有验证任务。建议顺序：

* masked frame-state reconstruction / temporal contrastive learning；
* 同一元素的邻帧、不同视角或不同表演者作为正样本；
* 以人工粗粒度 family、方向、可见性和评分区间做检索/聚类评估；
* 只有在 embedding 能稳定区分“元素/过渡/非元素”后，才接 score head。

未知动作应标为 `open_set=true`，不要为了训练分类器把它强行归入最近的 Trick 名称。

### 4.4 Score Model：以元素分值状态与排序为先

当前 `score_delta` 的正向范围为 0–10，负向范围为 -10–-1，重点扣分有固定值；它更接近裁判记分账本，而不是动作的客观技术价值。现有 4 个评分文件共有 37 条 `positive + 0` 的无分元素；另有一个文件包含 0 条 event，这只能说明该文档当前为空，不能推断整场没有元素。

第一阶段只建议训练：

* 元素区间是否存在、边界是否正确，以及模型是否漏检元素；
* 元素的 scored/unscored 状态和正/负分方向；
* 元素间的相对难度或质量排序，以及同一视频、同一裁判内的分值增量预测。

暂不建议直接回归整场绝对分。需要同时采集：裁判 ID、规则版本、视频版本、多个裁判的事件和总分、事件是否为技术项/失误/重大扣分、以及事件与元素的关联置信度。整场分数应由元素分数、重复惩罚、流程/完成度和扣分项组成，并保留可解释分解。

## 5. 必须补充的训练数据（按优先级）

### P0：让数据可以连接和审计

* 为每个评分视频补齐 `source_video_sha256`，与 v5 图像标注使用同一 `source_group`。
* 将 score v2 的元素记录和排除/场景区间导出为帧索引，并保留秒值、FPS 假设和原始 event ID。
* 建立统一 manifest：视频、帧、感知运行、人工标注、评分文档均通过 hash/relative path 连接。
* 明确 `scored`、`unscored`（当前由 `positive + 0` 导出）、`unlabeled`、`no_event`、`not_applicable`、`unknown`，禁止把缺失标注当负样本。

### P1：连续感知序列

* 至少为 20–30 个完整表演提供全程或高密度（建议 10–30 FPS 的关键窗口）悠悠球 bbox、字符串可见中心线、可见性/遮挡状态。
* 在现有 7 组、856 帧连续集之外，加入不同摄像机、分辨率、灯光、肤色/服装和方向的完整视频；按视频而非帧划分 train/val/test。
* 每个连续片段保留模型预测和人工修订两份，记录 propagation、temporal_fusion、low_confidence_rescue 等来源。

### P2：元素语义、无分元素与感知对齐

* 将现有 score v2 事件规范化为元素记录，补齐 `element_id`、`score_status=scored|unscored`、`phase` 和边界不确定度；将 `positive + 0` 作为无分元素，空白文档保持 `unlabeled`。
* 对每个元素标注 `non_element/transition`、完成/失败/中断、是否跨遮挡，以及与感知片段的 0/1/多对多关联。
* 元素区间和分值至少由两名标注者独立复核；保留裁判间分歧，不要只存平均值。
* `action_name` 可选；若为空，必须有粗粒度 family 或 open-set 标记，不能把空字符串当类别。

### P3：评分与泛化

* 覆盖不同名次和表现质量，避免数据只来自高分选手。
* 为至少一个完整赛制采集规则版本、裁判总分和扣分明细；记录场景、入场/退场、不可评分区间。
* 预留 2–3 个完全未见赛事/摄像机作为外部测试，不参与调参。

## 6. 分阶段实施与验收门槛

### 阶段 A：数据对齐与基线（先做）

产物：`motion_sequence_v1`、统一视频 manifest、score v2 元素视图、数据审计报告。

门槛：视频 hash 可闭环；train/val/test 无 `source_group` 或 image hash 泄漏；每个 frame 的未知/排除状态可解释；感知指标复现 `sequence_metrics.py` 的结果。

### 阶段 B：连续 Frame Encoder

先不训练 Tokenizer。比较 MLP/Temporal CNN/Transformer/Set Encoder，输入包括 bbox、polyline、方向、质量字段和独立 pose 分支；同时保留无 pose 的消融基线，以量化姿态对元素边界和评分的真实增益。

门槛：在外部视频上不降低悠悠球检测、字符串中心线和方向的时序指标；缺失帧不会导致 embedding 崩溃；推理输出保留模型版本和输入 manifest hash。

### 阶段 C：元素区间规范化与边界模型

以 score v2 的人工元素区间作为第一版验证真值，先实现时间轴规范化、无分元素表示、发球/收球阶段和 `non_element`。随后再训练 start/end 模型，用于发现边界修正候选，而不是替换人工记录。

门槛：报告 ±2/±5/±10 帧 boundary F1、过切/欠切、平均元素时长、跨遮挡片段召回和无分元素召回；与“现有 score 区间”和“固定时长切片”基线比较。

### 阶段 D：Element Token 与检索

在稳定边界上训练自监督 embedding 和轻量语义头，先做相似元素检索、未知动作检测和跨选手/赛事泛化。

门槛：检索 Recall@K、聚类稳定性、跨 source_group 性能；必须证明 embedding 对边界扰动和感知缺失具有鲁棒性。

### 阶段 E：元素级评分模型

先做元素检测/排序，再做分解式总分。模型输出应包含元素分数、原始 score event 引用、置信区间和不可评分原因。

门槛：按视频隔离的事件级 AP/F1、排序相关性（Spearman/Kendall）、校准误差、与裁判间一致性；外部赛事测试合格后才考虑整场分数。

### 阶段 F：Element Transformer

只有当元素区间、Token 检索和元素-感知对齐均达到门槛，且拥有足够多完整表演时，才训练序列模型建模 Combo、重复和流程。它应作为增量模型，不替代可解释的元素分数与扣分账本。

## 7. 评估、版本和泄漏控制

* 所有任务按 `source_group`/视频切分；连续帧不能跨 split，近重复图片按 `image_sha256` 去重。
* 每个模型 run 保存数据 manifest hash、权重 hash、输入契约版本、训练参数和初始化 lineage；沿用 `training_v3/train.py` 的可晋级规则。
* 分别报告感知、边界、元素、事件和整场指标；不要用一个总分掩盖上游失败。
* 评分标签要区分“没有事件”和“没有标注”；对排除区间不计算训练和评估损失。
* 任何由模型传播、插值、颜色/Hough 增强得到的几何都必须保留 provenance，不能回写为人工真值。
* 需要至少一个完全冻结的外部测试集；调阈值、选择 model soup 或选择元素长度都不能查看该集合标签。

## 8. 不建议现在做的事情

* 不要丢弃现有 `evidence_start/end`、anchor、正/负分、无分状态和发球/收球区间；应先规范化并保留来源。
* 不要把 722 个稀疏图像样本或 856 个连续评估帧直接拼成“大规模动作语料”。
* 当前 v3 图像任务不读取 pose 字段；最终 Frame State 可以消费独立 pose 分支，但必须支持缺失和低置信状态。
* 不要用空 `action_name` 训练闭集动作分类器，也不要把 embedding 相似度当作技术价值证明。
* 不要以整场总分的简单求和替代事件去重、重复动作、失败和重大扣分逻辑。
* 不要把 `not_visible`、`occluded`、`out_of_frame` 和 `unknown` 合并成同一个负类。

## 9. 推荐的最小可行产品（MVP）

第一版可交付目标不是“自动裁判”，而是：

1. 对新视频输出带 provenance 的逐帧 Frame State 和质量标记；
2. 使用现有元素区间，在人工确认的连续片段上补出感知对齐和边界修正候选及 ±帧置信区间；
3. 对每个候选元素检索相似历史元素，并显示字符串/悠悠球证据；
4. 将模型元素区间映射回 score v2 元素记录，由人工确认 scored/unscored 状态并生成可审计的分项分数；
5. 对未知、遮挡、证据不足的片段明确输出“需人工复核”，而不是强行给分。

达到上述目标后，Element Tokenizer 才有真实的工程输入，Element Transformer 也才有可验证的训练目标。长期目标“连续绳线-悠悠球状态变化 -> 运动元素 Token -> 技术价值”可以保留，但必须以这些数据契约和验收门槛为前置条件。
