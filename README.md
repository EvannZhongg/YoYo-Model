# YoYo Model

悠悠球检测、绳线分割、三分类方向识别和完整视频追踪项目。训练数据统一由一个 canonical 数据集管理，不再按来源版本拆分训练流程。

## 命令行入口

所有命令行入口集中在 `cli/`，从项目根目录使用 `python -m` 调用，查看任一命令的参数说明：

```powershell
.\.venv\Scripts\python.exe -m cli.training.train --help
```

## 当前数据集

统一数据集目录：

```text
datasets/1Ayoyo_dataset/
  manifest.json
  canonical/
  detection/
  string_segmentation/
  orientation/
  orientation_roi/
```

`manifest.json` 是训练数据身份和来源隔离策略的唯一依据。训练和评估都会校验 manifest；不要在模型训练完成后重建或修改对应 manifest。

数据集由 `annotations/` 下所有包含 `labels/` 的直接子目录统一构建，自动排除 `score_annotations/`。构建过程只纳入质量审核通过的标注，按图像 SHA-256 全局去重，并按来源视频组隔离、按标签分布优化 `train/val/test` 拆分：

```powershell
.\.venv\Scripts\python.exe -m cli.dataset.prepare_training --clear
.\.venv\Scripts\python.exe -m cli.dataset.prepare_orientation_view --clear
```

对已有输出执行 `--clear` 时会默认冻结当前 manifest 的来源组拆分：旧来源保留原 `train/val/test`，新增来源只进入 `train`。只有在有意创建全新评估协议时才使用 `--resplit`；也可用 `--freeze-splits-from <manifest.json>` 明确指定拆分血缘。

当前标签包含：

- 悠悠球 bbox
- 绳线可见性和几何标注
- `trick_orientation` 三分类：`horizontal`、`normal`、`not_applicable`
- `presentation_orientation` 画面朝向：`frontal`、`edge_horizontal`、`edge_vertical`、`unknown`
- 人工审核状态和来源信息

RTMPose-m WholeBody 仅在视频追踪运行时使用，其 ONNX 模型只存放在项目内的 `models/rtmpose/`，不会写入 C 盘用户缓存。方向模型只读取悠悠球 ROI。

不包含具名招式类别。项目不再训练或推理具体招式名称，也不生成招式时间段或候选短视频。

## 主训练流程

统一训练入口位于 `training_v3/`，命令行入口集中在 `cli/`：

```powershell
.\.venv\Scripts\python.exe -m cli.training.train `
  --dataset-dir datasets/1Ayoyo_dataset `
  --project-dir runs/v2v3 `
  --task detection `
  --epochs 100 `
  --device 0 `
  --auto-download
```

可用任务：

- `detection`
- `string_segmentation`
- `orientation`
- `all`

语义绳模型继续使用统一数据集的 `string_segmentation` view：

```powershell
.\.venv\Scripts\python.exe -m cli.training.train_semantic `
  --dataset-dir datasets/1Ayoyo_dataset/string_segmentation `
  --project runs/v2v3 `
  --name yoyo_v2v3_semantic_string `
  --architecture lraspp_mobilenet_v3 `
  --pretrained-backbone `
  --negative-sample-weight 4.0 `
  --device cuda
```

ROI 方向模型使用：

```powershell
.\.venv\Scripts\python.exe -m cli.training.train_orientation `
  --view-manifest datasets/1Ayoyo_dataset/orientation_roi/manifest.json `
  --project-dir runs/v2v3 `
  --device 0
```

Gradio 的 `Unified Training` 页签调用同一套入口：训练使用 `workbench_train_v2v3`，评估使用 `workbench_evaluate_v2v3`。

## 模型评估

只评估带 `run_manifest.json` 的统一训练运行：

```powershell
.\.venv\Scripts\python.exe -m cli.training.evaluate runs/v2v3/<run-name> --device 0
```

评估器会校验数据 manifest 和 best weights 的 SHA-256，然后在来源隔离的 test split 上运行。

当前已固化的训练版本、独立测试指标和权重哈希见 [`reports/training.md`](reports/training.md)。

## 完整视频追踪

```powershell
.\.venv\Scripts\python.exe -m cli.tracking.track_video path\to\input.mp4
```

默认追踪配置在 `config.yaml` 的 `tracking` 区块。RTMPose 不参与当前悠悠球检测、
绳线追踪或方向识别，因此 `tracking.enable_pose=false`，Workbench 中需要人体/手部
审核信息时可勾选 RTMPose，CLI 可传入 `--pose`。姿态推理只使用 RTMPose WholeBody
及其 YOLOX 人体检测器。每次运行输出：

- 完整追踪视频
- 逐帧 JSONL
- 固定宽度逐帧特征
- 视觉审核图和审核索引
- 输入路径、权重路径和参数的 `run.json`（追踪默认不再读取整段视频或权重来计算 SHA-256）

逐帧结果保留球体、绳线、姿态、bad case 和 `trick_orientation`。追踪器不会基于运动阈值生成时间窗口，不会导出 segment 或候选 clip。

追踪审核和模型索引只保留运行所需的路径及参数元数据；训练/评估阶段用于数据集隔离的
manifest 校验仍然启用。需要生成发布归档哈希时，可运行模型索引的
`--include-hashes` 选项显式开启，避免日常刷新重复读取大型 checkpoint。

## 操作台

```powershell
.\.venv\Scripts\python.exe app.py
```

默认地址：<http://127.0.0.1:7866>

操作台包含：

- 统一数据集训练与评估
- 悠悠球计分事件标注（五轨剪辑式时间轴，含三条计分轨、场景轨、不可标记轨，以及 Anchor、Evidence interval、自动续标与 JSON 元数据）
- 完整视频追踪和逐帧审核

数据集人工确认状态独立保存在 `workbench_state/dataset_review_status.json`。
人工确认只表示核验者看过并认可当前标注；`uncertain` 也可以核验完成，但仍不进入
悠悠球检测和方向训练监督。
当前 `yoyo_dataset_review_v3` 使用标签文件大小和纳秒修改时间使编辑后的确认失效，
不再保存标签 SHA-256。该映射不属于 `agent_yoyo_string_annotation_v5`
标签 schema，因此无需升级或重建已有训练标签。

计分标注每次修订都会原子写入 `annotations/score_annotations/`。每个 JSON 的 `video.source_path` 指向 `videos/` 下对应的受控源视频；`Score Annotation` 页签中的会话管理抽屉可直接加载视频并继续标注，也可修改组别/裁判、导出或删除会话。该目录属于独立的计分模型 pipeline，目前仅用于数据标注，不接入当前训练。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
