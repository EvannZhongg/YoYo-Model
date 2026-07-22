---
domain:
- cv
- multi-modal
tags:
- yoyo
- object-detection
- auto-labeling
models:
- Qwen/Qwen3.6-35B-A3B
- Qwen/Qwen3.5-35B-A3B
license: apache-2.0
---

# YoYo Auto Annotation

这是一个用于悠悠球图像识别/追踪模型前期数据准备的自动标注工具。当前流程会调用多模态大模型识别图片中的悠悠球，并保存：

- 原图副本
- 坐标 JSON
- 带框可视化图片

后续可以把这些标注结果转换为 YOLO、COCO 等训练格式，用于训练悠悠球检测与视频追踪模型。

## 项目结构

```text
app.py                    Gradio 界面入口
config.py                 读取 .env 和 config.yaml
config.yaml               模型、数据集路径、输出路径、OSS/base64、YOLO 配置
.env.example              密钥配置示例

annotation/               大模型自动标注模块
  annotator.py            大模型标注、bbox 解析、图片/JSON 保存逻辑
  prompts.py              悠悠球标注 prompt
  batch_annotate.py       批量自动标注入口
  sync_annotations.py     人工审核后同步清理标注三件套

yolo_training/            YOLO11 训练模块
  prepare_dataset.py      将 annotations 转为 YOLO 数据集
  download_model.py       下载 YOLO11 预训练权重
  train.py                准备数据并训练 YOLO11

string_segmentation/      审核门控的绳子分割模型
  prepare_dataset.py      多笔中心线转 YOLO segmentation 标签
  train.py                训练并记录数据/权重哈希

video_tracking/           视频识别与跟踪模块
  tracker.py              YOLO 检测 + ByteTrack + bad case/片段清单

video_dataset/            视频优先的数据集构建与绳线标注协议
  build.py                源视频清单、按 source_group 切分、抽帧

annotation/video_frame_annotator.py
                          球体框、绳子折线、手点和 bad case 自动预标注

batch_annotate.py         annotation.batch_annotate 的兼容 wrapper
sync_annotations.py       annotation.sync_annotations 的兼容 wrapper
prepare_yolo_dataset.py   yolo_training.prepare_dataset 的兼容 wrapper
download_yolo_model.py    yolo_training.download_model 的兼容 wrapper
train_yolo.py             yolo_training.train 的兼容 wrapper
track_video.py            video_tracking.tracker 的兼容 wrapper

dataset/                  原始数据集图片
annotations/              自动标注输出目录，默认生成
yolo_dataset/             YOLO 训练数据集，默认生成
models/                   YOLO11 权重默认下载目录
runs/                     YOLO 训练输出目录
tracked_videos/           视频跟踪可视化输出目录
datasets/video_v1/        视频源清单、抽帧和审核门控标注
archive/                  旧静态数据/旧模型的可恢复归档
tmp/                      临时图片、临时 JSON、单图检测输出
```

## 安装依赖

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 配置密钥

复制 `.env.example` 为 `.env`，然后填写自己的 API Key：

```env
API_KEY=your_dashscope_api_key

# 只有 config.yaml 里的 model.image_transport 为 "oss" 时才需要：
OSS_ACCESS_KEY_ID=your_oss_access_key_id
OSS_ACCESS_KEY_SECRET=your_oss_access_key_secret
```

不要把 `.env` 提交到仓库。

## 配置模型和数据集

主要配置都在 `config.yaml`：

```yaml
model:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  default_model: "qwen3.6-35b-a3b"
  image_transport: "oss"

dataset:
  image_input_dir: "dataset/Positive_Sample/1A"
  annotation_output_dir: "annotations"
```

`image_transport` 有两种模式：

- `oss`：把本地图片上传到阿里云 OSS，生成签名 URL 后发给模型。适合模型接口只稳定支持公网图片 URL 的情况。
- `base64`：把本地图片转为 data URL 直接发给模型。适合不想配置 OSS、且接口支持 base64 图片输入的情况。

如果使用 `oss`，还需要在 `config.yaml` 里配置：

```yaml
oss:
  endpoint: "your_oss_endpoint"
  region: "your_oss_region"
  bucket_name: "your_bucket_name"
```

## 启动界面

```bash
.\.venv\Scripts\python.exe app.py
```

默认访问：

```text
http://127.0.0.1:7866
```

界面包含以下页签：

- `Single Image`：单张图片测试 prompt 和模型效果。
- `Dataset Auto Label`：批量读取 `dataset.image_input_dir` 下的图片并自动标注。
- `Video Workbench`：视频数据集的统一操作台，依次完成抽帧、候选筛选、VLM 预标注、QA、YOLO 导出和可视化审核。
- `Annotation Review`：兼容旧流程的独立审核页。

### 视频操作台审核规则

`Video Workbench` 中的审核是组件级的：`bbox` 和 `string` 必须分别选择并审核。VLM 结果只作为预标注，带有 `VLM REVIEW ONLY` 标记的可视化图不能直接视为训练真值。只有 bbox 状态为 `approved` 或 `reviewed` 的样本才会进入检测数据集；只有 string 状态为 `approved` 或 `reviewed` 的样本才会进入绳子分割数据集。

通过动作带有强制真值门控：球体为 `visible/partially_visible` 时必须有有效框，球体为 `absent/out_of_frame` 时不得保留框；绳子为 `visible/partial` 时必须有人工折线或已审核 mask，`not_visible` 会清除旧几何，`uncertain` 不允许通过。工作台的 `Scene Label` 单独记录 `trick`、`transition`、`non_trick` 和 `unknown`，进场、离场、颁奖等场景可保留为显式负场景，而不与目标不可见混为一类。

旧标注升级到该门控 schema 时先 dry-run，再应用并重新严格审计：

```bash
.\.venv\Scripts\python.exe -m video_dataset.migrate_review_schema --dataset-dir datasets/video_v1
.\.venv\Scripts\python.exe -m video_dataset.migrate_review_schema --dataset-dir datasets/video_v1 --apply
.\.venv\Scripts\python.exe -m video_dataset.audit --dataset-dir datasets/video_v1 --strict
```

复杂招式中的绳子可能有交叉、环形结构或被手/球遮挡后的多个可见段。编辑器支持多笔 string stroke：完成一段后点击 `Start New String Stroke` 再标下一段。`not_visible` 的已审核帧会作为分割负样本；`uncertain`、待审核和拒绝标注不会进入训练。

string stroke 只描述画面中实际可见的绳段，不强制起点或终点靠近手/球。当前 `videos/` 全部是 1A，
工作台只显示 `hand_and_yoyo_attached` 和 `unknown`；未来组别仍保留在底层 schema，但不进入当前训练。
该标签会进入跟踪帧特征，但不改变 segmentation 数据的正负样本门控。

上传视频追踪时，绳线会把当前帧的模型/颜色观测与 Lucas-Kanade 光流做前后向一致性校验；一致时写为
`temporal_fusion`，不一致时保留冲突标记并要求复核。光流传播最多持续
`tracking.string_max_propagation_frames` 帧，且 `string_flow_fb_max_error` 会拒绝漂移。悠悠球暂时出画时，
若绳线仍可跟踪会保留记录并标记 `string_without_yoyo`，不会假设端点连接关系。`unknown` 模式下，远离球体且靠近
画面边缘的颜色候选会标记 `string_spatially_ambiguous`，不会作为下一帧的传播锚点；这些结果都不是训练真值。

绳子复核支持主动学习排序。工作台的 `Build String Review Queue` 会结合 QA、颜色提案失败、bad case、
视频来源覆盖和可选的 semantic v3 不确定性生成独立队列与 4x4 联系表；它不会修改标注或自动批准模型输出。
命令行等价操作：

```bash
.\.venv\Scripts\python.exe -m video_dataset.string_review_queue --limit 16 --with-model --device cuda
```

推荐操作顺序：

1. `Build / Refresh Frame Manifest` 建立视频隔离的抽帧清单。
2. `Audit Dataset Integrity` 检查源视频 SHA、帧/标注路径、source_group split 隔离和审核计数；严格模式下 warning 也会阻止继续训练。
3. `Select Candidate Frames` 用当前检测器生成候选帧。
4. `Run VLM Pre-annotation` 生成带框、绳线和手点的预标注。
5. 点击 `Build String Review Queue` 生成一批高价值样本和联系表；刷新 String 队列后会按 rank 排序。
6. 在同一页逐帧查看 `Visual Verification` 和 `Raw / Annotation / Semantic Detail`，先审核 bbox，再单独审核 string。三联细节视图会围绕球体、手点和绳线自动裁剪，并明确把 semantic 原始 mask 标为 `REVIEW ONLY`。可点击 `Load Semantic Prediction` 将经过 1A 球体空间锚定的候选折线载入未保存编辑器，再删除误报、补点或拆分 stroke；该操作本身不会写入标注。
7. `Run QA + Export Reviewed YOLO Dataset` 重新生成 QA 报告和审核门控后的 YOLO 数据集。
8. `Export Reviewed String Dataset` 导出已审核的绳子 segmentation 数据集。
9. train/val 均有已审核的可见绳线后，运行 `Train String Segmentation`。

命令行审计等价于：

```bash
.\.venv\Scripts\python.exe -m video_dataset.audit --dataset-dir datasets/video_v1 --strict
```

报告会写入 `datasets/video_v1/dataset_audit.json`。它不会修改数据；发现源视频被替换、帧路径失效、标注 split 与源视频不一致、source_group 泄漏或 SHA 不一致时会报告错误。VLM 只作为预标注，只有显式 `approved`/`reviewed` 的组件才计入训练样本。

模型版本索引可在操作台点击 `Refresh Model Registry`，或运行：

```bash
.\.venv\Scripts\python.exe model_registry.py
```

结果写入 `runs/model_registry.json`，会记录每个 run 的权重 SHA-256、训练数据 manifest 当前/记录哈希、独立测试指标、当前默认角色和缺失/漂移警告。没有 `run_manifest.json` 或独立测试指标的权重会被列为不完整版本，不会因为目录名看起来像 `best.pt` 就自动成为默认模型。

## 绳子分割训练

已下载的初始化权重默认是 `models/yolo11n-seg.pt`。先在操作台逐帧审核绳线；也可使用命令行导出：

```bash
.\.venv\Scripts\python.exe prepare_string_dataset.py --annotations-dir datasets/video_v1/annotations --output-dir datasets/video_v1/string_seg_v6 --clear
```

导出 manifest 会分别统计 positive、negative 和 segmentation instances，并验证 `source_group` 没有跨 train/val/test 泄漏。开始训练：

  ```bash
.\.venv\Scripts\python.exe train_string_model.py --clear-dataset --auto-download --device 0
  ```

细绳训练默认使用 `imgsz=960`、`mask_ratio=1`，并降低平移、缩放和 mosaic 增强。可用
`--imgsz`、`--mask-ratio`、`--translate`、`--scale` 和 `--mosaic` 显式覆盖。独立测试集评估：

```bash
.\.venv\Scripts\python.exe evaluate_string_model.py --weights runs/yolo/yoyo_string_v1/weights/best.pt --data datasets/video_v1/string_seg_v6/data.yaml --split test --imgsz 960
```

评估 JSON 同时记录 bbox 与 mask 指标，以及 positive/negative 图像数；小型测试集指标只能作为版本间比较依据。

若 train 或 val 没有人工通过的可见绳线，训练会直接终止并报告样本数。训练结果默认写入 `runs/yolo/yoyo_string_v1/`，其中 `run_manifest.json` 记录数据集、初始化权重、参数和环境版本。

### 细绳语义分割实验

YOLO instance segmentation 对当前少量细线样本效果很弱，因此另有一个单类轻量 U-Net 基线。它读取同一份
`datasets/video_v1/string_seg_v6` 审核门控数据，阈值只在 val 上选择，最后再评估 test：

```bash
.\.venv\Scripts\python.exe train_semantic_string_model.py --device cuda
.\.venv\Scripts\python.exe evaluate_semantic_string_model.py --weights runs/semantic/yoyo_string_semantic_v1/weights/best.pt --split test --device cuda
```

训练产物写入 `runs/semantic/<run_name>/`，包含 `best.pt`、`last.pt`、逐 epoch 历史、数据清单哈希、独立测试指标和预测复核图。上传视频追踪会自动识别这种 checkpoint，运行清单会记录 `string_model_kind=semantic`。semantic 输出仍然是 review-only，不能直接当作训练真值。

当前较新的审核实验保存在独立版本中：

- `datasets/video_v1/string_seg_v3`：57 条人工审核样本（train 44、val 6、test 7），其中新增了 14 条经视觉核验的 `not_visible` 负样本。
- `runs/semantic/yoyo_string_semantic_v3/weights/best.pt`：semantic v3。独立 test 的 exact Dice 为 0.241、3 px tolerant F1 为 0.335、image-presence F1 为 0.889、负样本平均误报为 158 像素；当前作为默认追踪候选，但每帧仍带有 `string_needs_review`。
- `datasets/video_v1/string_seg_v4`：在 v3 基础上加入 4 条人工逐帧核验的清晰正样本，共 61 条审核样本（train 48、val 6、test 7）。
- `runs/semantic/yoyo_string_semantic_v4_complete/weights/best.pt`：v4 实验 checkpoint（训练运行在第 26 epoch 退出，最佳 checkpoint 为 epoch 24）。独立 test 的 exact Dice 为 0.256、3 px tolerant F1 为 0.361、image-presence F1 为 0.727、负样本平均误报为 507 像素；负样本误报和 presence 指标劣于 v3，因此不作为默认模型。
- `runs/yolo/yoyo_string_v4/weights/best.pt`：使用 v3 数据集训练的 YOLO segmentation 实验；独立 test mask mAP 仍为 0，不推荐接入追踪。

显式指定 semantic v3（默认配置已指向该权重）：

```bash
.\.venv\Scripts\python.exe -m video_tracking.tracker <video> --string-weights runs/semantic/yoyo_string_semantic_v3/weights/best.pt --device cuda:0
```

semantic 训练的验证阈值搜索覆盖到 0.995，并在验证集排序时考虑负样本误报；test 仍只用于最终比较，不能反向调参。

追踪时若 semantic 模型在没有悠悠球且没有上一帧字符串锚点的片头/片尾产生全帧背景候选，系统会丢弃该候选，避免把背景写入 token 特征；已有轨迹在悠悠球暂时出画时仍可由光流传播，并标记 `string_without_yoyo` 和 `needs_review`。

## 命令行批量标注

如果不需要打开 Gradio 界面，可以直接运行：

```bash
.\.venv\Scripts\python.exe batch_annotate.py
```

等价的模块化命令：

```bash
.\.venv\Scripts\python.exe -m annotation.batch_annotate
```

默认会读取 `config.yaml` 中的配置：

- `dataset.image_input_dir`
- `dataset.annotation_output_dir`
- `model.default_model`
- `model.min_image_tokens`
- `model.max_image_tokens`
- `model.image_transport`

也可以临时覆盖参数：

```bash
.\.venv\Scripts\python.exe batch_annotate.py --input-dir dataset/Positive_Sample/1A --output-dir annotations --limit 10
```

查看全部参数：

```bash
.\.venv\Scripts\python.exe batch_annotate.py --help
```

`batch_annotate.py` 会在运行前检查 `annotations/` 与输入图片目录。若某张图片已经有完整输出，会自动跳过：

- `annotations/labels/...json`
- `annotations/visualizations/..._vis.png`
- `annotations/images/...`，当 `dataset.keep_source_images` 为 `true`

如果需要强制重新标注所有图片：

```bash
.\.venv\Scripts\python.exe batch_annotate.py --force
```

## 人工审核后同步清理

跑完一轮后，可以打开 `annotations/visualizations/` 查看可视化结果。如果发现某些图标得不好，直接删除对应的 `_vis.png`。

然后先 dry-run 查看将要删除的关联文件：

```bash
.\.venv\Scripts\python.exe sync_annotations.py
```

等价的模块化命令：

```bash
.\.venv\Scripts\python.exe -m annotation.sync_annotations
```

确认无误后执行清理：

```bash
.\.venv\Scripts\python.exe sync_annotations.py --apply --prune-empty-dirs
```

该脚本会对 `labels/`、`images/`、`visualizations/` 做互相验证。若发现某张图片的标注输出不完整，会删除同组仍存在的文件，例如：

- `annotations/labels/...json`
- `annotations/images/...`
- `annotations/visualizations/..._vis.png`

不会删除 `dataset/` 下的原始图片。

默认不会删除“输出三件套完整、但不在当前输入目录映射里”的历史结果。若确实要清理这类旧输出，可以加：

```bash
.\.venv\Scripts\python.exe sync_annotations.py --apply --delete-orphans
```

## YOLO11 训练

YOLO11 训练配置位于 `config.yaml` 的 `yolo:` 段：

```yaml
yolo:
  model_name: "yolo11n.pt"
  models_dir: "models"
  dataset_dir: "yolo_dataset"
  data_yaml: "yolo_dataset/data.yaml"
  class_names:
    - "yoyo"
  epochs: 100
  imgsz: 640
  batch: 8
  project: "runs/yolo"
  run_name: "yoyo_video_v1"
```

默认权重路径为：

```text
models/yolo11n.pt
```

下载默认 YOLO11n 权重：

```bash
.\.venv\Scripts\python.exe download_yolo_model.py
```

等价的模块化命令：

```bash
.\.venv\Scripts\python.exe -m yolo_training.download_model
```

指定下载模型和保存位置：

```bash
.\.venv\Scripts\python.exe download_yolo_model.py --model yolo11s.pt --models-dir models
```

如果 Ultralytics 资产地址变化，可以显式指定下载 URL：

```bash
.\.venv\Scripts\python.exe download_yolo_model.py --model yolo11n.pt --models-dir models --url https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt
```

先把 `annotations/` 转为 YOLO 数据集：

```bash
.\.venv\Scripts\python.exe prepare_yolo_dataset.py --clear
```

等价的模块化命令：

```bash
.\.venv\Scripts\python.exe -m yolo_training.prepare_dataset --clear
```

输出结构：

```text
yolo_dataset/
  data.yaml
  manifest.json
  images/train/
  images/val/
  labels/train/
  labels/val/
```

开始训练：

```bash
.\.venv\Scripts\python.exe train_yolo.py
```

等价的模块化命令：

```bash
.\.venv\Scripts\python.exe -m yolo_training.train
```

一条命令完成“准备数据 + 缺权重时自动下载 + 训练”：

```bash
.\.venv\Scripts\python.exe train_yolo.py --clear-dataset --auto-download
```

常用覆盖参数：

```bash
.\.venv\Scripts\python.exe train_yolo.py --epochs 50 --imgsz 640 --batch 8 --device 0
```

训练结果默认保存在：

```text
runs/yolo/yoyo_video_v1/
```

## 视频识别与跟踪

训练完成后，默认使用当前数据版本对应的 v2：

```text
runs/yolo/yoyo_video_v2/weights/best.pt
```

进行视频检测和跟踪。相关配置在 `config.yaml` 的 `tracking:` 段：

```yaml
tracking:
  weights_path: "runs/yolo/yoyo_video_v2/weights/best.pt"
  output_dir: "tracked_videos"
  confidence: 0.25
  iou: 0.7
  imgsz: 640
  device: ""
  string_weights_path: "runs/semantic/yoyo_string_semantic_v3/weights/best.pt"
  string_attachment_class: "hand_and_yoyo_attached"  # 当前 videos 主要为 1A
```

启动前端：

```bash
.\.venv\Scripts\python.exe app.py
```

在 `Video Tracking` 页签上传视频，选择训练后的权重文件，即可生成带检测框、轨迹 ID 和运动轨迹的可视化视频。

也可以命令行运行：

```bash
.\.venv\Scripts\python.exe track_video.py path\to\input.mp4 --weights runs/yolo/yoyo_video_v2/weights/best.pt --device 0
```

可用 `--string-attachment-class` 显式选择连接类别。默认 `unknown` 不会把颜色/光流结果强制连接到手或球；
只有 `hand_and_yoyo_attached` 会启用低置信度的手到球几何先验，该结果仍标记为 `needs_review`。

等价的模块化命令：

```bash
.\.venv\Scripts\python.exe -m video_tracking.tracker path\to\input.mp4 --weights runs/yolo/yoyo_video_v2/weights/best.pt --device 0
```
例如：
```bash
.\.venv\Scripts\python.exe track_video.py "path\to\input.mp4" --weights "runs\yolo\yoyo_video_v1\weights\best.pt" --device 0
```
输出默认保存在：

```text
tracked_videos/
```

每次运行会创建独立目录，包含 `tracked.mp4`、逐帧 `frames.jsonl`、`segments.json` 和 `run.json`。`run.json` 记录输入视频、球体权重、绳子权重的 SHA-256、参数、bad case 统计和输出文件。逐帧记录会区分 `visible`、`edge_clipped`、`likely_out_of_frame` 和 `not_visible_or_occluded`；后两者仍需人工确认，不能直接当作同一种负样本。默认使用 semantic v3 绳子权重。当前 `videos/` 全部是 1A，数据清单会写入 `current_action_group=1A`；因此 `string_attachment_class=hand_and_yoyo_attached` 时 semantic 组件必须在悠悠球附近才会成为绳子观测。工作台当前只提供 1A 和未知/需复核选项，未来 2A/3A/4A/5A 仍保留在底层 schema 与命令行接口中，但暂不参与训练。未知模式不会强行施加连接假设，并会把远离球体的组件标记为 `string_spatially_ambiguous`。权重不存在时会显式记录并回退到需要复核的颜色/光流估计。

候选招式片段由球体运动/手部距离启发式生成。运动阈值使用画面对角线/秒归一化（默认 `0.08`），不会再错误地乘 FPS；因此 4K/50fps 视频也能形成连续候选片段。片段默认限制为最多 180 秒（包含前后 padding），只作用于切分后导出的有效招式片段，不限制上传视频、源视频读取或整场追踪时长。片段仍标记为 `needs_review`，后续招式分类训练前需要人工确认起止点和招式标签。

## 从视频重新建立数据集

当前 `videos/` 全部属于 1A。需要为已有清单补齐或校正组别元数据时，先 dry-run，再显式应用：

```bash
.\.venv\Scripts\python.exe -m video_dataset.action_group --action-group 1A
.\.venv\Scripts\python.exe -m video_dataset.action_group --action-group 1A --apply
```

旧的 114 张静态图和旧训练运行已移入 `archive/legacy_static_20260720/`，当前流程从 `videos/` 重新开始。文件名按不透明 ID 处理，默认每个视频都是独立 `source_group`，因此不会把同一个视频的相邻帧拆到训练集和验证集。若确认多个文件属于同一选手或同一拍摄源，可以在 `datasets/video_v1/sources.json` 中手工设置相同的 `source_group` 后重新抽帧。

若发现源视频污染，先 dry-run 查看精确命中范围，再删除视频及派生记录：

```bash
.\.venv\Scripts\python.exe remove_video_source.py "videos\bad_video.mp4" --delete-video
.\.venv\Scripts\python.exe remove_video_source.py "videos\bad_video.mp4" --delete-video --apply
```

删除后重新运行 QA、YOLO 导出和 string segmentation 导出；不要使用 `--rebuild-sources`，否则其余视频可能被重新划分 split。

先生成源视频清单并抽取少量试验帧：

```bash
.\.venv\Scripts\python.exe build_video_dataset.py --sample-fps 1 --max-videos 2 --max-frames-per-video 20 --rebuild-sources
```

确认画面和划分后，再按需要抽取完整数据：

```bash
.\.venv\Scripts\python.exe build_video_dataset.py --sample-fps 1 --split all
```

抽帧记录初始状态为 `unreviewed`，空框不会自动视为悠悠球缺失。使用多模态模型进行球体、绳子折线、手点和 bad case 预标注：

```bash
.\.venv\Scripts\python.exe annotate_video_frames.py --dataset-dir datasets/video_v1 --split train --limit 100
```

如果完整抽帧数量较大，可以先用归档旧模型筛选候选帧，再只把候选帧送入多模态模型。候选框始终只是主动学习线索，不是训练真值：

```bash
.\.venv\Scripts\python.exe select_yoyo_candidates.py --dataset-dir datasets/video_v1 --weights archive/legacy_static_20260720/yoyo_yolo11_run/weights/best.pt --sample-fps 1 --max-videos 2 --max-candidates-per-video 30
.\.venv\Scripts\python.exe annotate_video_frames.py --dataset-dir datasets/video_v1 --candidates-only --split train --limit 100
```

输出标注的 `review_status` 是 `auto_labeled_needs_review`。人工审核后将其改为 `reviewed` 或 `approved`，再转换为 YOLO：

```bash
.\.venv\Scripts\python.exe prepare_yolo_dataset.py --annotations-dir datasets/video_v1/annotations --output-dir datasets/video_v1/yolo --clear
```

单个标注的审核状态可以显式修改：

```bash
.\.venv\Scripts\python.exe review_annotation.py datasets/video_v1/annotations/labels/train/<video_id>/frame_00000000.json approved --reviewer your_name
```

所有 VLM 标注都应先在 Gradio 的 `Annotation Review` 页签中查看可视化，再批准或拒绝。若 JSON 被手工修正，可重新生成带球框、绳线、手点和 bad case 标题的审核图：

```bash
.\.venv\Scripts\python.exe regenerate_annotation_visualizations.py --dataset-dir datasets/video_v1
```

也可以生成整批缩略图联系表，快速检查跨视频的误检和漏检：

```bash
.\.venv\Scripts\python.exe make_annotation_contact_sheet.py --dataset-dir datasets/video_v1 --split all
```

运行一致性 QA 后，`annotation_qa.csv` 会优先列出 VLM 与 bootstrap 框冲突或绳线几何异常的标注：

```bash
.\.venv\Scripts\python.exe validate_video_annotations.py --dataset-dir datasets/video_v1
.\.venv\Scripts\python.exe regenerate_annotation_visualizations.py --dataset-dir datasets/video_v1
```

数据集转换器默认跳过 `unreviewed` 和 `auto_labeled_needs_review`，只有显式审核通过的标注进入训练。只有在专门排查数据时才使用 `--include-unreviewed`。

审核通过的帧达到可训练规模后，使用同一份视频数据集训练并生成版本清单：

```bash
.\.venv\Scripts\python.exe train_yolo.py --annotations-dir datasets/video_v1/annotations --dataset-dir datasets/video_v1/yolo --clear-dataset --weights models/yolo11n.pt --project runs/yolo --name yoyo_video_v1 --epochs 100
```

训练完成后，`runs/yolo/yoyo_video_v2/` 下的 `run_manifest.json` 会记录数据清单哈希、初始权重哈希、训练参数和环境版本；追踪器的 `run.json` 会再记录实际输入视频和推理权重哈希。旧的 v1 仍保留在 `runs/yolo/yoyo_video_v1/` 作为回滚版本。

使用隔离测试集评估并保存 `test_metrics.json`：

```bash
.\.venv\Scripts\python.exe evaluate_yolo.py --weights runs/yolo/yoyo_video_v1/weights/best.pt --data datasets/video_v1/yolo/data.yaml --split test --device 0
```

## 标注输出

默认输出目录为 `annotations/`：

```text
annotations/
  images/              原图副本
  labels/              每张图对应一个 JSON 标注文件
  visualizations/      带框可视化图片
```

JSON 中会保存两种坐标：

- `bbox_2d`：模型返回的 0-999 归一化坐标
- `bbox_pixel`：换算后的像素坐标 `[x1, y1, x2, y2]`
- `string_polylines_2d` / `string_polylines_pixel`：一个或多个可见绳线 stroke；旧的单笔字段 `string_polyline_*` 保留兼容。
- `visibility` / `string_visibility`：球体和绳子的可见性；`out_of_frame`、`absent`、`not_visible` 不应被当成普通可见样本。
- `bad_case`：如 `yoyo_not_visible`、`motion_blur`、`string_ambiguous`、`non_trick_scene`。

示例：

```json
{
  "image_size": [1920, 1080],
  "bbox": [
    {
      "bbox_2d": [410, 320, 520, 455],
      "bbox_pixel": [787, 345, 999, 491],
      "label": "yoyo",
      "sub_label": "visible yoyo body"
    }
  ]
}
```

## Prompt

悠悠球检测 prompt 位于 `annotation/prompts.py` 的 `YOYO_DETECTION_PROMPT`。当前要求模型只返回 JSON，并尽量只框选可见的悠悠球主体，不包含手、线和背景物体。
