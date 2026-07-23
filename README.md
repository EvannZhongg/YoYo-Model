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

common/                   无业务依赖的共享基础设施
  files.py                文件枚举与流式 SHA-256 计算

workbench/                工作台应用服务层
  commands.py             数据集、训练和评估命令编排（不依赖 Gradio）
  review.py               标注审核队列、预览和统计查询
  tracking.py             跟踪审核画廊的视图数据组装

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

## 架构边界

代码按“界面/入口 -> 应用服务 -> 领域模块 -> 公共基础设施”单向依赖：

- `app.py` 只负责 Gradio 控件、事件绑定和界面适配；数据集构建、训练、评估等流程统一放在 `workbench/`。
- `annotation/`、`video_dataset/`、`video_tracking/`、`string_segmentation/`、`yolo_training/` 各自拥有本领域逻辑，领域之间只通过明确的函数或文件协议协作。
- `common/` 只能包含不依赖任何领域模块的稳定工具，禁止把业务规则放入其中。
- 根目录的一百字节左右脚本是已公开 CLI 的兼容入口；新实现应进入对应包内，兼容脚本只导入并调用包内 `main()`。
- 测试应直接导入所属领域或应用服务模块；只有界面适配行为才从 `app.py` 导入。

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
- `Video Workbench`：视频数据集的统一操作台，依次完成抽帧、候选筛选、VLM 预标注、hard-negative review-only 挖掘、QA、YOLO 导出和可视化审核。
- `Annotation Review`：兼容旧流程的独立审核页。

### 视频操作台审核规则

`Video Workbench` 中的审核是组件级的：`bbox` 和 `string` 必须分别选择并审核。VLM 结果只作为预标注，带有 `VLM REVIEW ONLY` 标记的可视化图不能直接视为训练真值。只有 bbox 状态为 `approved` 或 `reviewed` 的样本才会进入检测数据集；只有 string 状态为 `approved` 或 `reviewed` 的样本才会进入绳子分割数据集。
每次审核状态变更都会自动追加到 `datasets/video_v1/manual_review_log.jsonl`，保留组件、reviewer、可见性、场景标签和审核理由。

无法从当前帧可靠判断的组件使用 `unresolved`，不要用 `rejected` 或猜测出的正/负标签代替。`unresolved` 是可审计的终态：不会进入训练、不会继续占用主动学习队列，但仍可在工作台按状态筛选并在获得相邻帧或专家证据后重新处理。

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

绳子复核支持两种主动学习排序。工作台的 `Build String Review Queue` 默认只处理 `train`：`Uncertainty first` 会结合 QA、颜色提案失败、bad case、
视频来源覆盖和当前 semantic 模型不确定性寻找高风险样本；`Agreement first` 仅从无 QA warning、无高风险 bad case 且已有正样本几何的待审核标注中，按标注/模型的 tolerant F1、模型置信度和连通分量数优先展示较一致的样本。两种策略都会生成独立队列与 4x4 联系表，不会修改标注或自动批准模型输出。
工作台的 `Derived Holdout Source Groups` 会同时从候选筛选、VLM 预标注、颜色 proposal、通用组件审核、string review、hard-negative 和邻帧 mining 队列中排除这些来源，避免把 fresh holdout 帧带入补标或调参。旧的 `Annotation Review` 页签也默认使用该排除列表。
命令行等价操作：

```bash
.\.venv\Scripts\python.exe -m video_dataset.string_review_queue --split train --exclude-source-groups ab03bb7118b0 --limit 16 --with-model --weights runs/semantic/yoyo_string_semantic_v17_reviewed_expansion_hn005/weights/best.pt --device cuda
```

若要先复核模型与现有标注较一致的样本，在同一命令中加入 `--strategy agreement`。该策略必须启用 `--with-model`，agreement 数值只能作为复核提示，不能代替 raw/overlay/detail 的逐帧目视确认。

候选筛选、VLM 预标注和颜色 proposal 也支持相同的 `--exclude-source-groups` 参数；如果直接使用命令行，必须显式传入当前 fresh holdout 来源，不能只依赖工作台的默认值。

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

操作台每次启动都会为 YOLO、string 数据集、pipeline 评估和候选训练 run 生成带 UTC 时间戳的新名称。已有非空数据版本默认不可改写；只有显式勾选 `Allow replacing an existing dataset version` 才会执行 `--clear`。YOLO string 训练检测到已有 `manifest.json` 时使用 `--no-prepare` 只读训练，不会隐式重建数据；模型 run 名已存在时会拒绝启动，要求换一个新版本名。

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
.\.venv\Scripts\python.exe prepare_string_dataset.py --annotations-dir datasets/video_v1/annotations --output-dir datasets/video_v1/string_seg_candidate_NEW_VERSION --clear
```

导出 manifest 会分别统计 positive、negative 和 segmentation instances，并验证 `source_group` 没有跨 train/val/test 泄漏。开始训练：

  ```bash
.\.venv\Scripts\python.exe train_string_model.py --clear-dataset --auto-download --device 0
  ```

细绳训练默认使用 `imgsz=960`、`mask_ratio=1`，并降低平移、缩放和 mosaic 增强。可用
`--imgsz`、`--mask-ratio`、`--translate`、`--scale` 和 `--mosaic` 显式覆盖。独立测试集评估：

```bash
.\.venv\Scripts\python.exe evaluate_string_model.py --weights runs/yolo/yoyo_string_v1/weights/best.pt --data datasets/video_v1/string_seg_v9/data.yaml --split test --imgsz 960
```

评估 JSON 同时记录 bbox 与 mask 指标，以及 positive/negative 图像数；小型测试集指标只能作为版本间比较依据。

若 train 或 val 没有人工通过的可见绳线，训练会直接终止并报告样本数。训练结果默认写入 `runs/yolo/yoyo_string_v1/`，其中 `run_manifest.json` 记录数据集、初始化权重、参数和环境版本。

### 细绳语义分割实验

YOLO instance segmentation 对当前少量细线样本效果很弱，因此另有一个单类轻量 U-Net 基线。当前默认读取
`datasets/video_v1/string_seg_v17_reviewed_expansion` 审核门控数据，阈值只在 val 上选择，最后再评估 test：

```bash
.\.venv\Scripts\python.exe train_semantic_string_model.py --device cuda
.\.venv\Scripts\python.exe evaluate_semantic_string_model.py --weights runs/semantic/yoyo_string_semantic_v1/weights/best.pt --split test --device cuda
```

训练产物写入 `runs/semantic/<run_name>/`，包含 `best.pt`、`last.pt`、逐 epoch 历史、数据清单哈希、独立测试指标和预测复核图。上传视频追踪会自动识别这种 checkpoint，运行清单会记录 `string_model_kind=semantic`。semantic 输出仍然是 review-only，不能直接当作训练真值。

当前较新的审核实验保存在独立版本中：

- `datasets/video_v1/string_seg_v3`：57 条人工审核样本（train 44、val 6、test 7），其中新增了 14 条经视觉核验的 `not_visible` 负样本。
- `runs/semantic/yoyo_string_semantic_v3/weights/best.pt`：semantic v3 历史默认。其原生 7 帧 test 的 exact Dice 为 0.241、3 px tolerant F1 为 0.335、image-presence F1 为 0.889、负样本平均误报为 158 像素；每帧仍带有 `string_needs_review`。在 v8-v13 共用的 9 帧冻结 test 上重新评估时，对应指标为 0.180、0.312、0.923 和 158 像素。
- `datasets/video_v1/string_seg_v4`：在 v3 基础上加入 4 条人工逐帧核验的清晰正样本，共 61 条审核样本（train 48、val 6、test 7）。
- `runs/semantic/yoyo_string_semantic_v4_complete/weights/best.pt`：v4 实验 checkpoint（训练运行在第 26 epoch 退出，最佳 checkpoint 为 epoch 24）。独立 test 的 exact Dice 为 0.256、3 px tolerant F1 为 0.361、image-presence F1 为 0.727、负样本平均误报为 507 像素；负样本误报和 presence 指标劣于 v3，因此不作为默认模型。
- `datasets/video_v1/string_seg_v8`：85 条人工审核样本（train 68、val 8、test 9）；train 为 45 条正样本和 23 条负样本。v7 到 v8 只增加 train 真值，val/test 文件逐一 SHA-256 相同。
- `runs/semantic/yoyo_string_semantic_v8_balanced_v2/weights/best.pt`：使用 balanced val 选择器和轻量 hard-negative 项训练的 v8 实验。冻结 test 的 exact Dice 为 0.193、3 px tolerant F1 为 0.357、image-presence F1 为 0.857、负样本平均误报为 394 像素；绳线质量略升但 bad case 劣于 v3，因此不替换默认追踪权重。
- `datasets/video_v1/string_seg_v9`：86 条人工审核样本（train 69、val 8、test 9）；新增一条补全双横线与右侧下垂线的 train 正样本，val/test 与 v8 文件逐一 SHA-256 相同。
- `datasets/video_v1/string_seg_v10`：87 条人工审核样本（train 70、val 8、test 9）；新增 `9a86c6fcc304/frame_00001200` 一条人工确认的 partial string（悠悠球不在画面，但手间可见绳线），val/test 与 v9 文件逐一 SHA-256 相同。
- `runs/semantic/yoyo_string_semantic_v10_best_finalize/weights/best.pt`：v10 数据集的 warm-start finalize 实验。冻结 test 的 exact Dice 为 0.127、3 px tolerant F1 为 0.307、image-presence F1 为 0.667、负样本平均误报为 45.3 像素；泛化劣于 v3，不替换默认追踪权重。中断后塌缩的 `yoyo_string_semantic_v10_reviewed_partial` 与其 resume run 仅保留为 bad-case 记录。
- `datasets/video_v1/string_seg_v11`：加入 4 条人工确认的邻帧负样本后共 91 条审核样本，但随后发现 `350c71b59099` 的 400/600/700 帧被旧流程错误标成负样本；三帧实际都有清晰悠悠球和绳线。因此 v11 及其 run 只保留为错误标签 bad-case 记录，不能用于晋级。
- `datasets/video_v1/string_seg_v12`：修正上述三帧后仍为 91 条审核样本；train 为 50 条正样本、24 条负样本，共 121 个 string instances，val/test 分别为 8/9 帧。v10、v11、v12 的 val/test 共 34 个图像和标签文件逐一 SHA-256 相同。
- `runs/semantic/yoyo_string_semantic_v12_hardneg_ft_v3/weights/best.pt`：从 semantic v3 warm-start，val 选择 epoch 19 和阈值 0.9204。冻结 9 帧 test 的 exact Dice 为 0.202、3 px tolerant F1 为 0.381、image-presence F1 为 0.923、负样本平均误报为 439 像素；虽然细绳 tolerant 指标提高，但背景误报显著劣于 semantic v3，因此不替换默认权重。24 条 train 负样本诊断为 0 条阈值误检，说明仍存在明显的冻结集泛化差距。
- `datasets/video_v1/string_seg_v13`：94 条人工审核样本；新增三张“悠悠球握持可见、绳线不可见”的 train 负样本，均同时通过 detector proposal 的 4K bbox overlay 人工核验。train 为 50 条正样本、27 条负样本；val/test 与 v12 的 34 个文件逐一 SHA-256 相同。
- `runs/semantic/yoyo_string_semantic_v13_hardneg_ft_v3/weights/best.pt`：与 v12 使用相同 seed、初始化和超参数，val 选择 epoch 16。冻结 test 的 exact Dice 为 0.192、3 px tolerant F1 为 0.376、image-presence F1 为 0.857、负样本平均误报为 510 像素；该单变量数据实验仍未改善泛化，因此保留为失败 run，不替换 semantic v3。
- `datasets/video_v1/string_seg_v14`：99 条人工审核样本；从五个不同 train source 新增 5 张复杂正样本，包含多分支绳线、运动模糊环和长竖线。train 为 55 条正样本、27 条负样本，共 135 个 string instances；val/test 与 v13 的 34 个图像和标签文件逐一 SHA-256 相同。
- `runs/semantic/yoyo_string_semantic_v14_diverse_ft_v3/weights/best.pt`：沿用 v12/v13 的 seed、semantic v3 初始化和超参数，val 选择 epoch 14、阈值 0.9204。冻结 test 的 exact Dice 为 0.197、3 px tolerant F1 为 0.388、image-presence F1 为 0.923、负样本平均误报为 502 像素；tolerant F1 略升但 Dice 和背景误报仍不满足晋级条件，因此不替换 semantic v3，也不再重复相同 warm-start 配方。
- `runs/semantic/yoyo_string_lraspp_v14_transfer_v1/weights/best.pt`：在 v14 上从 ImageNet MobileNetV3 backbone 重新初始化的 LR-ASPP 迁移实验，val 选择 epoch 25、阈值 0.8707。冻结 test 的 exact Dice 为 0.155、3 px tolerant F1 为 0.367、image-presence F1 为 0.923、负样本平均误报为 1941 像素；跨场景背景结构误报显著恶化，因此保留为失败 run，不调 test 阈值、不继续相同配方，也不替换 semantic v3。
- `datasets/video_v1/string_seg_v15`：108 条人工审核样本；从 5 个 train source 新增 9 张逐帧原图核验的负样本，包括赛事叠字片头以及“悠悠球握持可见、绳线尚未展开”的非招式帧。train 为 55 条正样本、36 条负样本，共 135 个 string instances；val/test 的 34 个图像和标签文件与 v14 逐一 SHA-256 相同。
- `runs/semantic/yoyo_string_semantic_v15_crossscene_hardneg_ft_v3/weights/best.pt`：只把 v14 训练数据替换为 v15，其余 semantic v3 初始化、seed 和超参数保持相同，val 选择 epoch 18、阈值 0.9204。冻结 test 的 exact Dice 为 0.199、3 px tolerant F1 为 0.393、image-presence F1 为 0.923、负样本平均误报为 444 像素，四项均优于 v14，但误报仍远高于默认 semantic v3 的 158 像素，因此不晋级。该 checkpoint 在 36 张 train 负样本上为 0 像素误报，说明剩余问题是未见场景泛化，而不是继续增大同一 hard-negative loss 权重。
- `datasets/video_v1/string_seg_v16_fresh_holdout`：不改 canonical 标注的派生数据集，按完整来源隔离为 `75/8/14`，仅 `ab03bb7118b0` 进入 test，旧 test 被排除，跨 split 重复图像哈希为 0。
- `runs/semantic/yoyo_string_semantic_v16_fresh_hn005/weights/best.pt`：随机初始化、hard-negative 权重 0.005，val 选择 epoch 38、阈值 0.9701。fresh test 的 exact Dice 为 0.234、3 px tolerant F1 为 0.412、presence precision/recall/F1 为 0.846/0.917/0.880，2 张负样本平均误报 99.5 像素。接入 detector 和球体抑制后的 precision/recall/F1 为 1.000/0.750/0.857、Dice 0.203、tolerant F1 0.388，负样本误报为 0；曾晋级为 review-only 默认，现保留为回滚 checkpoint。hard-negative 权重 0.05 的 resume run 塌缩为全背景，只保留为失败记录。
- `datasets/video_v1/string_seg_v17_reviewed_expansion`：只增加 3 张经过 geometry 与 semantic/temporal 双重视觉复核的 train 正样本，导出为 `78/8/14`；train 含 46 张正样本、32 张负样本和 119 个 string instances。val/test 与 v16 逐文件相同，test 仍仅含冻结来源 `ab03bb7118b0`，跨 split 图像哈希交集为 0。
- `runs/semantic/yoyo_string_semantic_v17_reviewed_expansion_hn005/weights/best.pt`：与 v16 相同随机初始化配方，仅替换 train 数据，val 选择 epoch 38、阈值 0.9701。单次 frozen test 的 exact Dice 为 0.233、3 px tolerant F1 为 0.426、presence precision/recall/F1 为 0.846/0.917/0.880，2 张负样本平均误报 60.5 像素。相对 v16，presence 不变、tolerant F1 提高 0.013、负样本误报下降 39%，exact Dice 仅下降 0.0003；逐帧 prediction sheet 复核也显示片头和多张正样本的过分割减少，因此晋级为新的 review-only 默认。
- `datasets/video_v1/string_seg_v18_reviewed_expansion`：继续只增加 `ce89cc4914e6/frame_00001500` 与 `c43840972091/frame_00000700` 两张经过双角色原图、overlay、模型提示和相邻帧复核的 train 正样本，导出为 `80/8/14`；train 含 48 张正样本、32 张负样本和 126 个 string instances。v17 的 16 个 val 文件与 28 个 test 文件逐一 SHA-256 相同，旧 train 文件无删除或改写，跨 split 来源重叠为 0。
- `runs/semantic/yoyo_string_semantic_v18_reviewed_expansion_hn005/weights/best.pt`：与 v17 相同随机初始化和 hard-negative 配方，val 选择 epoch 18、阈值 0.15；exact Dice 0.289、3 px tolerant F1 0.738、presence precision/recall/F1 1.000/0.800/0.889，3 张负样本平均误报 0。val prediction sheet 已逐帧复核，四张命中正样本的预测沿可见绳线，仍漏检一张清晰正样本。由于 `ab03bb7118b0` frozen test 已用于 v17 的一次最终比较，v18 禁止再次评估该 test，当前只作为等待新独立 holdout 的候选，不替换 v17 默认。
- `datasets/video_v1/yolo_v4`：加入上述三张 reviewed bbox 后导出 41 个 train、5 个 val 图像，共 46 个框；冻结 val/test 的 22 个文件与 `yolo_v3` 逐一 SHA-256 相同。当前 detector 默认权重仍保持 `yoyo_video_v2`。
- `datasets/video_v1/yolo_v5`：加入五张跨 source 人工核验 bbox 后导出 46 个 train、5 个 val、6 个 test 图像，共 51 个框；val/test 的 22 个图像和标签文件与 v4 逐一 SHA-256 相同。
- `runs/yolo/yoyo_video_v5/weights/best.pt`：从与 detector v2 相同的 `models/yolo11n.pt`、seed 和 50 epoch 配方训练，仅改变 reviewed train 数据，val 选择 epoch 21。冻结 6 帧 test 的 precision 为 0.818、recall 为 0.905、mAP50 为 0.928、mAP50-95 为 0.667；虽然定位质量高于 v2 的 0.593，但 precision、recall 和 mAP50 均退化，且 test 很小，因此保留为候选实验，不替换 `yoyo_video_v2`。
- `datasets/video_v1/yolo_v6`：新增 6 张来自 3 个 train source 的“手持悠悠球、绳线未展开”人工核验框，共 52 个 train、5 个 val、6 个 test 图像和 57 个框；val/test 的 22 个图像和标签文件与 v5 逐一 SHA-256 相同。
- `runs/yolo/yoyo_video_v6/weights/best.pt`：沿用 detector v5 的初始化、seed 和 50 epoch 配方，仅改变 reviewed train 数据，val 选择 epoch 44。冻结 test 的 precision 为 0.988、recall 为 1.000、mAP50 为 0.995、mAP50-95 为 0.709；并在原 tracking bad case 上以 0.536 置信度恢复被 v2 漏检的手持悠悠球，因此晋级为默认 detector。
- `datasets/video_v1/yolo_v7`：在 v6 训练和冻结 test 评估完成后，再加入 2 张独立 train source 的人工核验手持悠悠球框，共 54 个 train、5 个 val、6 个 test 图像和 59 个框；val/test 的 22 个文件与 v6 逐一 SHA-256 相同。v7 只作为下一轮数据版本保存，未用已查看的 v6 test 结果反向训练或调参。
- `runs/yolo/yoyo_video_v8_fresh_holdout/weights/best.pt`：从干净 `models/yolo11n.pt` 初始化，在 `39/5/13` 的来源隔离数据上训练 50 epoch。fresh test precision/recall 为 0.995/0.818、mAP50 为 0.884、mAP50-95 为 0.568；部署组合的审核 bbox 子集 presence F1 为 0.900、平均 IoU 为 0.842。因其训练来源和 fresh test 明确隔离，晋级为默认 detector。
- `runs/yolo/yoyo_string_v4/weights/best.pt`：使用 v3 数据集训练的 YOLO segmentation 实验；独立 test mask mAP 仍为 0，不推荐接入追踪。

显式指定当前 semantic v17（默认配置已指向该权重）：

```bash
.\.venv\Scripts\python.exe -m video_tracking.tracker <video> --string-weights runs/semantic/yoyo_string_semantic_v17_reviewed_expansion_hn005/weights/best.pt --device cuda:0
```

semantic 训练的验证阈值搜索覆盖到 0.995。阈值与 checkpoint 都按 tolerant F1 和 image-presence F1 的调和分数选择，再比较 presence、负样本误报、tolerant F1 和 exact Dice；test 仍只用于最终比较，不能反向调参。

追踪融合把 semantic/YOLO string model 的“无组件”视为负证据：学习模型启用时不会再用更弱的 HSV/Hough 线段覆盖该结果。只有未启用学习模型时才允许 color/Hough fallback。该规则已在“悠悠球可见但绳线未展开”的 bad case 上验证为 `string: null`，同时保留已审核正样本上的 semantic string observation。

部署链路另有只读组合评估器。它在审核帧上依次运行 detector、基于预测球框的 semantic 锚定和 tracker suppression，报告 string presence、Dice、3 px tolerant F1、负样本误报，以及审核 bbox 子集的 detector presence/IoU；不会写回标注。工作台默认运行 val，选择 test 时必须显式勾选最终评估确认：

```bash
.\.venv\Scripts\python.exe -m video_tracking.evaluate_pipeline --split val --device 0
```

- `1x` 旧基线组合 `yoyo_video_v6 + semantic_v3` 在 `string_seg_v15` 的 val 上 string presence F1 为 0.571、Dice 为 0.191、tolerant F1 为 0.296、负样本误报为 0；冻结 test 对应为 0.833、0.180、0.306 和 132.3 像素。
- 当前默认使用 val 选择的 `2x` semantic 推理，并按倍率平方同步扩大最小连通域过滤。再抑制主要落在预测球框内部的组件后，val presence precision/recall/F1 为 1.000/0.800/0.889、Dice 为 0.220、tolerant F1 为 0.535，3 张负样本误报均为 0。加入球体内部抑制之前的一次冻结 test 为 presence F1 0.857、Dice 0.214、tolerant F1 0.334；由于之后查看失败帧并增加了抑制规则，旧 test 不再能证明最终规则的独立泛化，必须扩充新的未查看 holdout 后再作最终结论。
- semantic v10 按组合 val 排序成为候选（presence F1 0.889、tolerant F1 0.547、负样本误报 0），但冻结 test presence F1 降到 0.667、误报增至 2 帧、漏检增至 2 帧，因此被否决；当时的默认 string 权重仍保持 semantic v3。
- detector v6 在同一组合基准的 6 张审核 bbox test 帧上 presence precision/recall 均为 1.0，平均 IoU 为 0.838。逐帧 JSON 和复核图保存在 `runs/pipeline_eval/`，test 结果不能用于调阈值、锚定距离或选择新候选。
- fresh-holdout 最终组合 `yoyo_video_v8_fresh_holdout + semantic_v16_fresh_hn005` 在 14 张 string test 帧上 presence precision/recall/F1 为 1.000/0.750/0.857、Dice 为 0.203、tolerant F1 为 0.388，2 张负样本误报均为 0；13 张审核 bbox 帧的 detector presence precision/recall/F1 为 1.000/0.818/0.900，平均 IoU 为 0.842。三张 string 漏检为 100/350/600 帧，继续进入 bad-case 复核，不能用该 test 反向调阈值。派生 split 的 canonical bbox 标注仍位于原 train 目录，评估器会跨 split 唯一解析；本次 detector 子指标由冻结预测行补录真值，清单明确记录 `model_inference_rerun=false`。

旧 test 经失败分析后不再是未查看 holdout。为保持旧清单可复现而不改 canonical 标注，可在派生导出时把一个完整 train 来源撤出并路由到新 test，同时排除旧 test。当前 fresh-holdout 预留跨场景 BYPC 来源 `ab03bb7118b0`；导出器还按图像 SHA-256 去除跨 split 的重复赛事片头，优先保留 test，其次 val，最后 train：

```bash
.\.venv\Scripts\python.exe prepare_yolo_dataset.py --annotations-dir datasets/video_v1/annotations --output-dir datasets/video_v1/yolo_v8_fresh_holdout --clear --holdout-source-groups ab03bb7118b0 --exclude-original-test
.\.venv\Scripts\python.exe prepare_string_dataset.py --annotations-dir datasets/video_v1/annotations --output-dir datasets/video_v1/string_seg_v16_fresh_holdout --clear --holdout-source-groups ab03bb7118b0 --exclude-original-test
```

对应清单为 detector `39/5/13` 和 string `75/8/14`（train/val/test），三个 split 的图像内容哈希交集均为 0。fresh-holdout 模型必须从 `models/yolo11n.pt` 或随机 semantic 初始化重新训练；禁止 warm-start 任何见过 `ab03bb7118b0` 的旧项目权重。

长训练可启用基于 val 排名的 early stopping；`--early-stopping-min-epochs` 之前不会停止，连续指定轮数没有改善才结束。`run_manifest.json` 会记录完成 epoch 数、是否提前停止和停止原因：

```bash
.\.venv\Scripts\python.exe train_semantic_string_model.py --dataset-dir datasets/video_v1/string_seg_v13 --initial-weights runs/semantic/yoyo_string_semantic_v3/weights/best.pt --lr 0.0001 --hard-negative-weight 0.005 --epochs 20 --early-stopping-min-epochs 8 --early-stopping-patience 6 --device cuda
```

可用当前默认 semantic checkpoint 对已审核的 train `not_visible` 负样本生成 review-only hard-negative 队列；它只写诊断 JSON/CSV/contact sheet，不修改标注：

```bash
.\.venv\Scripts\python.exe -m video_dataset.hard_negative_queue --dataset-dir datasets/video_v1 --weights runs/semantic/yoyo_string_semantic_v17_reviewed_expansion_hn005/weights/best.pt --exclude-source-groups ab03bb7118b0 --device cuda --output-name string_hard_negative_queue_v17
```

队列按预测像素、连通组件数、最大概率降序排列，并记录 checkpoint SHA-256。模型响应只能用于人工排序；只有人工审核后的 `not_visible` 样本才可进入训练，val/test 不参与补标或调参。

从队列锚点提取邻帧时，默认只生成 JSON 和 contact sheet，不会写训练真值：

```bash
.\.venv\Scripts\python.exe -m video_dataset.hard_negative_candidates --dataset-dir datasets/video_v1 --queue datasets/video_v1/string_hard_negative_queue_v17.json --exclude-source-groups ab03bb7118b0 --offset-seconds=-1,-0.5,0.5,1 --top-anchors 8 --limit 32 --output-name hard_negative_neighbor_candidates_v17
```

默认只围绕模型实际误检的 train 负样本；需要扩大来源覆盖时可显式添加 `--include-clean-anchors`，把模型已正确判负的人工负样本也作为邻帧锚点。`Video Workbench > Hard-negative Mining` 提供相同控件，两个阶段仍只生成 review-only JSON 和 contact sheet。

逐帧查看原图和 contact sheet 后，只有确认画面中没有可见绳线的帧才能显式晋级。`--approve` 可重复，晋级事件会追加到人工审核日志；未列出的候选保持 review-only：

```bash
.\.venv\Scripts\python.exe -m video_dataset.hard_negative_candidates --dataset-dir datasets/video_v1 --queue datasets/video_v1/hard_negative_neighbor_candidates_v3.json --approve 635b990cee32:150 --reviewer manual --reason "Visually confirmed no visible string."
```

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

训练完成后，默认使用来源隔离的 fresh-holdout v8：

```text
runs/yolo/yoyo_video_v8_fresh_holdout/weights/best.pt
```

进行视频检测和跟踪。相关配置在 `config.yaml` 的 `tracking:` 段：

```yaml
tracking:
  weights_path: "runs/yolo/yoyo_video_v8_fresh_holdout/weights/best.pt"
  output_dir: "tracked_videos"
  confidence: 0.25
  iou: 0.7
  imgsz: 640
  device: ""
  visualization_max_width: 1920  # tracked preview only; metadata remains in source coordinates
  pose_weights_path: "models/yolo11n-pose.pt"
  enable_pose: true
  string_weights_path: "runs/semantic/yoyo_string_semantic_v17_reviewed_expansion_hn005/weights/best.pt"
  string_inference_scale: 2.0  # val-selected; 4x semantic pixels/compute versus 1x
  string_inference_fps: 10.0   # optical flow fills gaps; failed propagation triggers model reacquisition
  string_attachment_class: "hand_and_yoyo_attached"  # 当前 videos 主要为 1A
```

工作台默认启用 `models/yolo11n-pose.pt`。当画面中有多个人时，首帧先根据可见手腕、手腕到悠悠球的距离、可见关键点数量和姿态质量选择选手；后续帧优先匹配上一帧选手框，短时悠悠球或手腕遮挡不会立即切换到观众。若没有空间上合理的候选，旧参考会被丢弃并重新执行首帧选择。只把选中人物的 17 点骨架和左右手腕写入帧记录并绘制到审核视频。每帧的 `pose_person` 保存选择方法、人物索引、候选人数、参考年龄、框 IoU、中心位移和关键点统计，`run.json` 保存 pose 权重 SHA-256，便于复现实验。

多人首帧、旧人物参考被拒绝或低 IoU 续接会写入 `pose_person.review_reasons` 并标记 `pose_identity_needs_review`；该标记只是可视化复核提示，不会变成姿态真值。Tracking Visual Review 的每个格子显示选中的人物、候选人数、cold/temporal 模式、框 IoU 和复核原因。长视频联系表优先保留首尾、检测/绳线/bad-case/pose 状态切换帧，再补充均匀时间采样，不再因每帧都有悠悠球而只显示开头。

在一段 3840x2160 实拍赛事视频的两次 10 帧 CUDA 验证中，3 名 pose 候选里连续选中选手索引 0，10/10 帧均得到 2 个手腕和 17 个姿态点；即使其中 1 帧悠悠球 detector 漏检，人物选择仍通过上一帧参考保持正确（框 IoU 0.966、中心位移 16.83 个源像素）。启用 detector、semantic string 和 pose 的追踪循环为 0.70-0.81 fps，1920x1080 审核预览不会改变 4K 源坐标元数据。该测量不含模型加载和审核产物导出时间。

启动前端：

```bash
.\.venv\Scripts\python.exe app.py
```

在 `Video Tracking` 页签上传视频，选择训练后的权重文件，即可生成带检测框、轨迹 ID 和运动轨迹的可视化视频。

也可以命令行运行：

```bash
.\.venv\Scripts\python.exe track_video.py path\to\input.mp4 --weights runs/yolo/yoyo_video_v8_fresh_holdout/weights/best.pt --device 0
```

可用 `--string-attachment-class` 显式选择连接类别。默认 `unknown` 不会把颜色/光流结果强制连接到手或球；
只有 `hand_and_yoyo_attached` 会启用低置信度的手到球几何先验，该结果仍标记为 `needs_review`。

等价的模块化命令：

```bash
.\.venv\Scripts\python.exe -m video_tracking.tracker path\to\input.mp4 --weights runs/yolo/yoyo_video_v8_fresh_holdout/weights/best.pt --device 0
```
例如：
```bash
.\.venv\Scripts\python.exe track_video.py "path\to\input.mp4" --weights "runs\yolo\yoyo_video_v1\weights\best.pt" --device 0
```
输出默认保存在：

```text
tracked_videos/
```

每次运行会创建独立目录，包含 `tracked.mp4`、逐帧 `frames.jsonl`、`segments.json` 和 `run.json`。`run.json` 记录输入视频、球体权重、绳子权重的 SHA-256、参数、模型推理帧数、实际处理速度、bad case 统计和输出文件。固定宽度的 `frame_features` 当前使用 251 维 `yoyo_tracking_frame_features_v7`；v7 保留 v6 的主球侧绳段字段，并新增最多 8 个独立绳段、每段 4 个采样点、分量长度/存在位、手腕锚点状态和多段光流质量。分量各自采样，绝不跨分量插值。v6 曾新增 `bad_case_string_hand_anchor_mismatch`，v5 曾新增 `bad_case_pose_identity_needs_review`；下游实验必须按 manifest 的 schema/version 和 feature names 校验，不能与旧宽度直接拼接。逐帧记录会区分 `visible`、`edge_clipped`、`likely_out_of_frame` 和 `not_visible_or_occluded`；后两者仍需人工确认，不能直接当作同一种负样本。默认使用 source-isolated semantic v17 绳子权重。`string_inference_fps` 默认按 10 FPS 调度高分辨率 semantic 模型，中间帧保留逐帧 detector 并尝试前后向一致性光流传播；若已有锚定绳线无法传播且当前帧仍检测到悠悠球，会立即重新运行 semantic 模型，避免为提速牺牲绳线覆盖。每帧的 `string_model_inference.status/reason` 会区分定时推理、`flow_reacquire`、间隔跳过或模型不可用。设为 0 可恢复逐帧 semantic 推理。当前 `videos/` 全部是 1A，数据清单会写入 `current_action_group=1A`；因此 `string_attachment_class=hand_and_yoyo_attached` 时 semantic 组件必须在悠悠球附近才会成为绳子观测；若观测几何到可见手腕的最近距离超过 `max(48 px, 0.025 * frame diagonal)`，记录 `string_hand_anchor_mismatch` 供复核，并禁止该观测成为下一帧的光流锚点，但不使用手腕补画或扭曲绳线。tracking review sheet 会优先保留此类事件，直接显示观测到手腕的距离/阈值，并用红框标出 mismatch 帧。工作台当前只提供 1A 和未知/需复核选项，未来 2A/3A/4A/5A 仍保留在底层 schema 与命令行接口中，但暂不参与训练。未知模式不会强行施加连接假设，并会把远离球体的组件标记为 `string_spatially_ambiguous`。权重不存在时会显式记录并回退到需要复核的颜色/光流估计。

semantic 推理会先把 uint8 图像传到目标设备再归一化，并只在各连通分量的局部框内生成 contour 和 skeleton。三张 train 样本的优化前后 observation JSON 完全一致。Lucas-Kanade 只为绳线点周围构建带 192 px padding 的局部金字塔；真实接受帧与原全图结果的平均坐标差为 0.0001 px，悠悠球缺失时仍保留全图 fallback。同一段 4K/50fps、10 帧的端到端 tracking loop 从 20.16 秒降至 11.37 秒，同时保持 10/10 帧的 string output。该结果仍只有约 0.88 FPS，`run.json.performance` 是判断当前机器实际吞吐量的权威记录；完整赛事视频运行前应先用工作台的 `Preview Frame Limit` 检查模型与速度。

在已分析过的历史 test 来源 `db159b217457` 上，对 3-7 秒共 200 帧使用 detector v8、semantic v17、pose 和 10 FPS semantic cadence 复跑手腕锚点回归。新运行识别出 25 帧 `string_hand_anchor_mismatch`（距离 110.35-506.39 px，阈值 110.15 px）；逐一检查这些帧的下一帧后，0 帧继续由 `lucas_kanade_optical_flow` 从错误锚点传播。光流帧数由 31 降到 21，semantic 总推理帧数由 140 增到 145，处理速度由 1.1695 FPS 变为 1.1645 FPS。该来源只用于已知 bad-case 诊断，不作为新 holdout 或模型晋级证据；完整产物位于 `runs/tracking_validation/recycled_old_test_v17_long_hand_anchor_v6/`。

继续检查 frame 200 的原始 semantic 概率图后发现，模型实际预测了 6 个合格分量：2 个悠悠球本体误报、1 个球侧绳段和 3 个目视确认的手侧绳段；旧后处理只保留球侧绳段。tracking metadata v1.2 在 `hand_and_yoyo_attached` 模式下会保留球侧分量、手腕阈值内的已有分量以及与这些手侧分量相邻一跳的已有分量，所有缺口仍以独立 `polylines` 保存，绝不补线。多段光流逐条执行，主球侧分量丢失时立即触发 semantic 重捕获。frame 200 的输出由 1 段变为 4 段，对已审核真值的 Dice/tolerant F1 从 `0/0` 提升到 `0.152/0.280`；原分辨率 overlay 已确认没有跨缺口连线。

同一历史 200 帧诊断复跑后，110 个 semantic 输出中有 79 帧保留手腕支持分量，mismatch 从 25 帧降到 8 帧；23 个光流输出中 8 帧成功传播多段几何，10 帧明确记录部分分量丢失。semantic 推理帧数为 143，处理速度为 1.0767 FPS。review sheet 以及最高分量数的 f211/f257 原分辨率 overlay 均已目视检查，额外分量沿复杂绳形且未包含人物轮廓或背景线。运行产物位于 `runs/tracking_validation/recycled_old_test_v17_long_hand_components/`；这些结果仍是已知 bad-case 的工程诊断，不是模型 test 晋级指标。

该 200 帧运行已重新导出为 v7 frame features，NPZ shape 为 `200 x 251`，NPZ 与 manifest 的 251 个 feature names 逐项一致。frame 200 的 4 个独立分量分别进入 component 0-3，component 4 明确为缺失；整段最大观测分量数和最大编码分量数均为 8，没有发生超过固定槽位上限的静默截断。`frame_features.jsonl` 同时保留原始分量数、实际编码分量数、手腕支持分量数、锚点状态和部分光流丢失状态，供未来招式 clip-token 切片与训练前审计。

上传追踪完成后，操作台除 `tracking_review_sheet.jpg` 外还会读取同一 run 的 `tracking_review_index.json`，显示最多 24 个审核事件；每个事件按 source-coordinate raw frame、tracked overlay 的顺序成对展示，因此最多为 48 个可点击、可全屏 Gallery 项。原图保存在 `review_raw_frames/`，overlay 保存在 `review_frames/`，index 同时记录两者尺寸；4K 源视频在默认 preview 配置下会明确显示为 raw `3840x2160`、overlay `1920x1080`。Gallery 严格保持事件抽样顺序，caption 直接列出 frame/time、球体是否可见、string method/confidence、独立分量数、手腕距离/阈值、pose review、完整 bad cases 和 `view=raw|overlay`；图片路径必须位于当前 tracking run 内，损坏或越界的 index 项会被忽略。相邻抽样帧采用一次顺序解码，跨度很大的抽样才使用逐帧 seek；真实 200 帧/24 事件的原图复核导出约 15.1 秒。Gallery 仍只是模型输出复核入口，不会自动写入训练标注或批准任何 VLM/semantic 结果。

`visualization_max_width` 默认只把 `tracked.mp4` 限制为 1920 px 宽；模型、`frames.jsonl`、frame features 和导出的招式 clip 始终使用源视频坐标/分辨率，`run.json` 同时记录 source/output dimensions。10 帧样例的预览文件从 1.22 MB 降至 0.43 MB，便于工作台播放和批量保存，但不会加速 semantic 推理；需要逐像素查看时可在工作台设为 0 输出源分辨率预览。

候选招式片段由球体运动/手部距离启发式生成。运动阈值使用画面对角线/秒归一化（默认 `0.08`），不会再错误地乘 FPS；因此 4K/50fps 视频也能形成连续候选片段。片段默认限制为最多 180 秒（包含前后 padding），相邻候选的共享 padding 会在活动区间中点裁切，保证导出的候选区间互不重叠，并在 `segments.json` 保留 `active_start_frame`/`active_end_frame`。该逻辑只作用于切分后导出的有效招式片段，不限制上传视频、源视频读取或整场追踪时长。片段仍标记为 `needs_review`，后续招式分类训练前需要人工确认起止点和招式标签。

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
