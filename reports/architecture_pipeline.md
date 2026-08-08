# 悠悠球识别与追踪 Pipeline

状态：2026-08-08 当前生产架构。历史实验与失效分支不在此文档保留。

```mermaid
flowchart TD
    A[视频帧] --> B[YOLO11s 悠悠球检测<br/>imgsz 1024]
    B --> C[ByteTrack ID 跟踪]

    C --> D{存在当前/最近悠悠球<br/>或已有绳线轨迹?}
    D -->|否| E[跳过语义前向]
    D -->|是| F[单 MobileNetV3-FPN<br/>960x544 / threshold 0.6471]
    F --> G[语义概率与多组件中心线]
    A --> H[语义支持邻域内<br/>饱和色/亮脊 Hough]
    G --> I[沿线概率门控与组件并集]
    H --> I

    I --> J{当前帧有新鲜观测?}
    J -->|是| K[直接保留当前几何]
    J -->|否| L[Lucas-Kanade 前后向光流<br/>最多短时传播 12 帧]

    C --> M[仅悠悠球方形 ROI]
    M --> N[三分类方向模型<br/>5/25 FPS 自适应采样<br/>EMA 与切换滞回]
    A -. 显式开启 .-> O[RTMPose-m WholeBody<br/>仅审核元数据]

    C --> P[逐帧结果]
    K --> P
    L --> P
    N --> P
    O -.-> P
    P --> Q[tracked.mp4 / frames.jsonl / run.json]
```

FPN 融合 MobileNetV3 编码器的五个空间尺度，以一个模型替代原来的普通域双 LR-ASPP、弱域模型和分档路由。Workbench 与 CLI 默认读取 `config.yaml` 中的同一候选权重；旧 ensemble/adaptive 参数只保留为显式兼容接口，不参与默认执行。
