# YOYO String Annotation VLM Assisted

本目录提供一套 VLM 辅助的悠悠球绳标注流程。配置的 VLM 只负责粗粒度分流：判断素材是否属于目标领域、场景是否明显为招式、是否存在明显坏例，以及安排复核优先级。绳子中心线、悠悠球框、手部位置、路径拓扑、招式方向和最终审核必须由视觉标注代理基于原图与相邻帧完成。

## 配置并使用模型

### 1. 准备运行环境

使用项目虚拟环境，不要使用全局 Python。首次使用前必须安装本目录 `requirements.txt` 中声明的依赖：

```powershell
& .\.venv\Scripts\python.exe -m pip install -r `
  skills\yoyo-string-annotation-vlm-assisted\requirements.txt
```

依赖包括 OpenCV、OpenAI、PyYAML、python-dotenv、Pillow、NumPy，以及 OpenCV 无法读取源视频时使用的 imageio/FFmpeg 回退解码器。安装完成后确认当前解释器和 OpenCV：

```powershell
& .\.venv\Scripts\python.exe -c "import sys,cv2; print(sys.executable); print(cv2.__version__)"
```

### 2. 配置 API Key

默认从本目录的 `.env` 读取 `API_KEY`：

```dotenv
API_KEY=your_dashscope_api_key
```

不要把 API Key 写入 `config.yaml`、标注 JSON、命令行参数或日志。需要使用其他环境变量名时，修改 `config.yaml` 中的 `model.api_key_env`。需要临时使用其他环境文件时，通过 `--env-file` 指定。

### 3. 配置模型

模型配置位于 `config.yaml`：

```yaml
model:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  api_key_env: "API_KEY"
  default_model: "qwen3.6-35b-a3b"
  min_image_tokens: 1024
  max_image_tokens: 9800
  max_response_tokens: 3000
  enable_thinking: false
  request_timeout_seconds: 180
  retries: 2

triage:
  promotion_confidence: 0.90
  quick_verify_confidence: 0.95
  notes_confidence: 0.70
  safe_bad_cases:
    - "motion_blur"
    - "low_contrast"
    - "edge_clipped"
```

| 配置项 | 含义 |
| --- | --- |
| `base_url` | OpenAI 兼容 API 地址；环境变量 `BASE_URL` 的优先级更高 |
| `api_key_env` | 保存 API Key 的环境变量名 |
| `default_model` | 默认视觉模型；`--model` 优先，其次是环境变量 `DEFAULT_MODEL` |
| `min_image_tokens` / `max_image_tokens` | 发送给模型的图像分辨率预算 |
| `max_response_tokens` | 单次响应的最大 token 数 |
| `enable_thinking` | 是否向兼容接口传递思考模式开关 |
| `request_timeout_seconds` | 单次请求超时时间 |
| `retries` | 请求失败后的重试次数 |
| `promotion_confidence` | 将安全粗分类写入草稿标注的最低置信度 |
| `quick_verify_confidence` | 进入快速确认队列的最低置信度 |
| `notes_confidence` | 写入 VLM 事实性备注的最低置信度 |
| `safe_bad_cases` | VLM 可安全提升的坏例标签 |

使用其他 OpenAI 兼容视觉模型时，修改 `base_url`、`api_key_env` 和 `default_model` 即可。接口必须支持图像输入和 `chat.completions`。

### 4. 采样并初始化标注

先从视频抽取 anchor 与相邻上下文帧，再初始化 `agent_yoyo_string_annotation_v4` 标注：

```powershell
& .\.venv\Scripts\python.exe `
  skills\yoyo-string-annotation-vlm-assisted\scripts\sample_video_frames.py `
  --videos INPUT_VIDEOS `
  --output ANNOTATION_PROJECT `
  --frames-per-video 12 `
  --oversample-factor 5 `
  --neighbor-offsets=-2,-1,1,2 `
  --separate-context `
  --hash-cache DATASET_ROOT\source_video_sha256_cache.json

& .\.venv\Scripts\python.exe `
  skills\yoyo-string-annotation-vlm-assisted\scripts\annotation_pipeline.py init `
  --images ANNOTATION_PROJECT\images `
  --output ANNOTATION_PROJECT `
  --min-approvals 2
```

### 5. 运行 VLM 分流

```powershell
& .\.venv\Scripts\python.exe `
  skills\yoyo-string-annotation-vlm-assisted\scripts\vlm_triage.py run `
  --labels ANNOTATION_PROJECT\labels `
  --output ANNOTATION_PROJECT\triage `
  --config skills\yoyo-string-annotation-vlm-assisted\config.yaml
```

临时覆盖模型可增加 `--model MODEL_NAME`。脚本按模型和提示版本缓存结果；只有源图、模型、提示契约或缓存结果确实失效时才使用 `--force`。

运行后生成：

- `triage/triage_manifest.json`：模型、提示版本、处理数量和队列统计。
- `triage/results/<source_group>/*.json`：标准化判断、置信度、警告和安全提升结果。
- `triage/agent_handoff.json`：按 `quick_verify`、`clear_candidate`、`standard`、`hard_case` 排序的视觉代理任务。

VLM 可以提升高置信度的 `scene_label`、安全 `bad_case` 和事实性 `notes`。它不能填写或批准 `string_visibility`、`trick_orientation`、坐标、框、手部、中心线、mask、路径、拓扑、审核结论等字段。

## 当前标注字段

当前标注 schema 为 `agent_yoyo_string_annotation_v4`。所有 `*_pixel` 坐标以原始图像像素为准；自动生成的 `*_2d` 字段是 0–999 归一化镜像，不是标注真值。

### 身份与视频溯源

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 标注 schema 版本 |
| `created_at_utc` / `updated_at_utc` | 创建与最近修改时间 |
| `source_image` | 当前标注对应的图像路径 |
| `image_sha256` | 图像内容校验值 |
| `image_size` | 原图 `[width, height]` |
| `source_video` | 原始视频路径 |
| `source_video_sha256` | 原始视频内容校验值 |
| `source_group` | 稳定的视频/来源组标识，供下游隔离划分数据集 |
| `video_id` | `source_group` 的兼容镜像 |
| `frame_index` | 在原视频中的零基帧号 |
| `timestamp_s` | 在原视频中的时间戳，单位为秒 |
| `sequence_id` | 以 anchor 为中心的采样序列标识 |
| `sampling_role` | `anchor` 或 `temporal_context` |
| `anchor_frame_index` | 当前序列的 anchor 帧号 |
| `sampling_manifest_sha256` | 采样清单的身份校验值 |

### 目标、绳子与场景

| 字段 | 含义与取值 |
| --- | --- |
| `visibility` | 悠悠球可见性：`visible`、`partially_visible`、`occluded`、`out_of_frame`、`absent`、`uncertain` |
| `yoyo_bbox_pixel` | 悠悠球原图像素框 `[x1,y1,x2,y2]` |
| `yoyo_bbox_2d` | 悠悠球框的 0–999 镜像 |
| `bbox` | 兼容旧消费者的框数据 |
| `string_visibility` | 绳子状态：`visible`、`partial`、`not_visible`、`uncertain` |
| `string_polylines_pixel` | 多段可见绳子中心线；遮挡和不确定区间必须断开 |
| `string_polylines_2d` | 多段中心线的 0–999 镜像 |
| `string_polyline_pixel` / `string_polyline_2d` | 第一段中心线的兼容镜像字段 |
| `string_mask_polygons_pixel` | 可选的可见绳区域多边形 |
| `hands_pixel` | 左右手原图像素位置：`{"left": point|null, "right": point|null}` |
| `hands_2d` | 左右手位置的 0–999 镜像 |
| `string_attachment_class` | `hand_and_yoyo_attached`、`yoyo_detached`、`hand_detached`、`unknown` |
| `scene_label` | `trick`、`transition`、`non_trick`、`unknown` |
| `trick_orientation` | `normal`、`horizontal`、`unknown`、`not_applicable` |
| `string_path` | 绳路拓扑、锚点、有序点、边证据和未解决间隙 |
| `bad_case` | 事实性问题列表，如 `motion_blur`、`partial_occlusion`、`low_contrast`、`edge_clipped`、`ambiguous_string` |
| `notes` | 简短事实性备注 |

`string_path` 的结构为：

| 子字段 | 含义与取值 |
| --- | --- |
| `topology` | `open`、`loop`、`branched`、`multiple`、`uncertain` |
| `reconstruction_status` | `complete`、`partial`、`uncertain`、`not_applicable` |
| `paths` | 有序路径数组 |
| `paths[].path_id` | 路径标识 |
| `paths[].start_anchor` / `end_anchor` | `left_hand`、`right_hand`、`yoyo` 或 `unknown` |
| `paths[].points_pixel` | 原图像素坐标中的有序路径点 |
| `paths[].edges` | 相邻点之间的边，包含 `from`、`to`、`evidence` 和 `confidence` |
| `paths[].edges[].evidence` | `observed`、`temporal` 或 `inferred` |
| `unresolved_gaps` | 无法可靠重建的遮挡或歧义区间说明 |

### 审核与版本记录

| 字段 | 含义 |
| --- | --- |
| `review_status` | 整体审核状态 |
| `bbox_review_status` | 悠悠球框审核状态 |
| `string_review_status` | 绳子标注审核状态；只有 `approved` 或 `reviewed` 可导出 |
| `reviewed_at_utc` | 最终审核时间，审核前可能不存在或为 `null` |
| `reviewer` | 最终审核者，审核前可能不存在或为 `null` |
| `quality.revision` | 当前修订号 |
| `quality.min_model_approvals` | 所需独立模型审核数量 |
| `quality.history` | 每次内容修改的审计历史 |
| `quality.reviews` | 审核者、角色、决定及内容摘要绑定记录 |

最终便携导出还会增加或改写：

| 字段 | 含义 |
| --- | --- |
| `source_image_original` | 导出前的原始图像路径 |
| `source_image` | 导出目录中的相对图像路径 |
| `visualization` | 最终审核叠加图的相对路径 |
