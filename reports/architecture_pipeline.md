# YoYo Model 项目流程与结构

状态：2026-09-07，基于 `main` 分支提交 `cc7aeaa`。

## 项目结构图

```mermaid
flowchart TB
    subgraph Entry["交互与命令入口"]
        APP["app.py<br/>Gradio Workbench"]
        CLI["cli/<br/>数据、训练、评估、追踪命令"]
    end

    subgraph Orchestration["业务编排层"]
        WB["workbench/<br/>标注、训练调度、追踪审核"]
        DATASET["training_v3/<br/>统一数据集、任务视图、manifest"]
        TRACKER["video_tracking/<br/>逐帧多模型融合"]
    end

    subgraph Models["模型能力层"]
        DET["yoyo_detection/<br/>悠悠球检测"]
        STRING_RUNTIME["string_tracking/<br/>绳线运行入口"]
        STRING_MODEL["string_segmentation/<br/>语义分割、几何提取、评估"]
        ORIENTATION["yoyo_orientation/<br/>ROI 方向分类"]
        POSE["video_tracking/rtmpose_backend.py<br/>可选人体与手部姿态"]
    end

    subgraph Shared["共享基础"]
        CONFIG["config.yaml + config.py<br/>模型路径与运行参数"]
        COMMON["common/<br/>文件哈希与方向语义"]
    end

    subgraph Storage["数据与产物"]
        ANN["annotations/<br/>审核后的标注"]
        DS["datasets/1Ayoyo_dataset/<br/>canonical 与任务视图"]
        RUNS["runs/<br/>run_manifest.json 与 checkpoint"]
        VIDEOS["videos/<br/>输入视频"]
        OUTPUTS["tracked_videos/<br/>视频、JSONL、审核产物"]
    end

    APP --> WB
    APP --> TRACKER
    CLI --> DATASET
    CLI --> DET
    CLI --> STRING_RUNTIME
    CLI --> ORIENTATION
    CLI --> TRACKER

    WB --> ANN
    WB --> DET
    WB --> STRING_RUNTIME
    WB --> ORIENTATION
    WB --> TRACKER

    ANN --> DATASET --> DS
    DS --> DET
    DS --> STRING_MODEL
    DS --> ORIENTATION
    STRING_RUNTIME --> STRING_MODEL

    DET --> RUNS
    STRING_MODEL --> RUNS
    ORIENTATION --> RUNS
    RUNS --> TRACKER
    VIDEOS --> TRACKER
    TRACKER --> DET
    TRACKER --> STRING_RUNTIME
    TRACKER --> ORIENTATION
    TRACKER -. "显式启用" .-> POSE
    TRACKER --> OUTPUTS

    CONFIG -. "配置" .-> WB
    CONFIG -. "配置" .-> TRACKER
    COMMON -. "共享语义与工具" .-> DATASET
    COMMON -. "共享语义与工具" .-> Models
```

### 目录职责

| 目录或文件 | 主要职责 |
| --- | --- |
| `app.py` | 组装标注、训练/评估、计分标注和视频追踪页签 |
| `cli/` | `python -m` 命令入口，薄封装后转发到正式模块 |
| `workbench/` | Workbench 页面后端、子进程命令编排和逐帧人工审核 |
| `training_v3/` | 构建 canonical 数据集、任务视图、来源隔离拆分和 manifest |
| `yoyo_detection/` | YOLO 悠悠球检测训练、推理和评估 |
| `string_segmentation/` | MobileNetV3-FPN 绳线分割、中心线提取和静态/连续集评估 |
| `string_tracking/` | 绳线训练、评估和运行时加载 facade |
| `yoyo_orientation/` | 悠悠球 ROI 方向分类训练、推理和评估 |
| `video_tracking/` | 检测、ByteTrack、绳线、方向、可选姿态的逐帧融合 |
| `config.yaml` / `config.py` | 独立模型权重和追踪 compositor 参数 |
| `reports/` / `tests/` | 当前有效方案、训练结论和关键链路测试 |

## 端到端流程图

```mermaid
flowchart TD
    SOURCE["annotations/*/labels<br/>多个审核标注源"]
    SCORE_UI["Workbench Score Annotation"]
    SCORE["annotations/score_annotations/<br/>独立计分标注 pipeline"]
    VALIDATE["schema 与质量审核校验<br/>图像 SHA-256 全局去重"]
    SPLIT["按 source_group 隔离<br/>train / val / test"]
    ROOT_MANIFEST["datasets/1Ayoyo_dataset/manifest.json<br/>数据身份与拆分血缘"]

    SOURCE --> VALIDATE --> SPLIT --> ROOT_MANIFEST
    SCORE_UI --> SCORE

    ROOT_MANIFEST --> CANONICAL["canonical/<br/>统一图像、标签和 index.jsonl"]
    ROOT_MANIFEST --> DET_VIEW["detection/<br/>YOLO data.yaml"]
    ROOT_MANIFEST --> STRING_VIEW["string_segmentation/<br/>data.yaml + view manifest"]
    ROOT_MANIFEST --> ORI_BUILD["orientation view 构建<br/>悠悠球方形 ROI"]
    ORI_BUILD --> ORI_VIEW["orientation_roi/<br/>view manifest"]

    DET_VIEW --> DET_TRAIN["yoyo_detection.train<br/>YOLO detection"]
    STRING_VIEW --> STRING_TRAIN["string_tracking.train<br/>MobileNetV3-FPN semantic string"]
    ORI_VIEW --> ORI_TRAIN["yoyo_orientation.train<br/>YOLO classification"]

    DET_TRAIN --> DET_RUN["detection run<br/>run_manifest.json + weights/best.pt"]
    STRING_TRAIN --> STRING_RUN["string run<br/>run_manifest.json + threshold + weights/best.pt"]
    ORI_TRAIN --> ORI_RUN["orientation run<br/>run_manifest.json + weights/best.pt"]

    DET_RUN --> EVAL["同协议独立 test 评估<br/>校验数据 manifest 与权重 SHA-256"]
    STRING_RUN --> EVAL
    ORI_RUN --> EVAL
    EVAL --> PROMOTE{"候选满足晋升门槛?"}
    PROMOTE -->|是| DEFAULTS["更新 config.yaml 与 config.py<br/>三类默认权重"]
    PROMOTE -->|否| EXPERIMENT["保留最小实验依据"]

    VIDEO["输入视频"] --> TRACK["video_tracking.tracker.track_video"]
    DEFAULTS --> TRACK
    TRACK --> RUN_OUTPUT["tracked.mp4 + frames.jsonl + run.json<br/>review sheet + review frames"]
    RUN_OUTPUT --> REVIEW["Workbench 逐帧审核<br/>tracking_frame_reviews.jsonl"]
```

`manifest.json` 是训练数据身份、来源拆分和任务视图的权威记录。每个独立训练运行再以 `run_manifest.json` 绑定数据 manifest、初始化权重、参数和最终 checkpoint；评估器据此选择对应任务并校验血缘。

## 视频逐帧追踪流程图

```mermaid
flowchart TD
    FRAME["输入视频帧"] --> DETECT["YOLO 悠悠球检测<br/>默认 imgsz 1024"]
    DETECT --> BYTE["ByteTrack 分配稳定 ID"]
    BYTE --> SELECT["选择当前悠悠球<br/>置信度、位置连续性、历史 track ID"]

    SELECT --> STRING_GATE{"到达绳线采样帧<br/>且存在当前、最近球体或绳轨?"}
    STRING_GATE -->|是| SEMANTIC["语义模型前向<br/>960x544 checkpoint 按 1.125 倍推理"]
    STRING_GATE -->|否| NO_FRESH["本帧无新语义观测"]
    SEMANTIC --> THRESHOLD["阈值取 max<br/>checkpoint 验证阈值, config floor"]
    THRESHOLD --> CENTERLINE["滞回掩码、连通域、骨架化<br/>最多 32 个中心线组件"]
    FRAME --> AUGMENT["饱和色 / 亮脊 Hough<br/>曲线 ridge fallback"]
    CENTERLINE --> UNION["语义概率门控后<br/>合并补充组件"]
    AUGMENT --> UNION
    UNION --> STRING_FUSE["estimate_string<br/>新鲜语义观测优先"]
    NO_FRESH --> STRING_FUSE
    PREVIOUS["前一帧可信绳线状态"] -.-> STRING_FUSE
    STRING_FUSE --> REACQUIRE{"绳线仍为空<br/>但当前悠悠球存在?"}
    REACQUIRE -->|是| RETRY["立即补跑语义模型并再次融合"]
    RETRY --> FLOW["最终绳线结果<br/>光流可补回次级组件，纯传播最多 12 帧"]
    REACQUIRE -->|否| FLOW

    SELECT --> ORI_GATE{"到达方向采样帧?"}
    ORI_GATE -->|是| ROI["悠悠球方形 ROI"]
    ROI --> ORI_MODEL["三类或四类方向模型"]
    ORI_MODEL --> MAP["四类 presentation 可映射为<br/>三类 trick_orientation"]
    MAP --> FILTER["EMA、切换滞回与确认<br/>稳定 5 FPS / 模糊时 25 FPS"]
    ORI_GATE -->|否| CARRY["沿用最近方向状态"]

    FRAME -. "enable_pose=true" .-> POSE["YOLOX 人体检测 + RTMPose WholeBody<br/>人物选择、身体与手部关键点"]
    SELECT -.-> POSE

    DETECT --> RECORD["组装 schema 1.2 逐帧记录<br/>检测、球体、绳线、方向、姿态、bad case"]
    FLOW --> RECORD
    FILTER --> RECORD
    CARRY --> RECORD
    POSE -.-> RECORD
    RECORD -. "可信状态" .-> PREVIOUS

    RECORD --> JSONL["frames.jsonl"]
    RECORD --> DRAW["绘制 bbox、轨迹、绳线和方向"]
    DRAW --> VIDEO_OUT["tracked.mp4"]
    RECORD --> SUMMARY["汇总 FPS、bad case、方向和绳线几何统计"]
    SUMMARY --> RUN_JSON["run.json"]
    JSONL --> REVIEW_ASSETS["tracking_review_sheet.jpg<br/>tracking_review_index.json + review frames"]
```

追踪时三类模型权重分别来自 `detection`、`string_tracking` 和 `orientation` 配置区块；`tracking` 只保存多模型编排、采样频率、时序滤波和可视化参数。RTMPose 默认关闭，仅在显式启用时补充审核元数据。

## 图表维护

目录、入口、模块依赖或配置归属变化时更新“项目结构图”；标注协议、数据视图、manifest 血缘、训练评估或模型晋升流程变化时更新“端到端流程图”；`video_tracking.tracker` 的逐帧分支、采样门控、时序状态、输出 schema 或审核产物变化时更新“视频逐帧追踪流程图”。维护时以实际代码和 `config.yaml` 为准，同步更新文档顶部日期与基线提交，检查图中的默认尺寸、阈值规则、采样频率和产物名称，并使用当前支持的 Mermaid 版本完成解析或渲染验证，最后执行 `git diff --check`。
