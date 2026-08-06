# 悠悠球识别与追踪 Pipeline

状态：持续更新，仅追加架构版本。

## 2026-08-03 版本1

```mermaid
flowchart TD
    A[视频帧] --> B[YOLO 悠悠球检测]
    B --> C[ByteTrack ID 跟踪]
    A --> D[YOLO Pose 手腕与人体关键点]
    A --> E[LR-ASPP 绳线语义分割<br/>960x544 / 每帧推理]
    E --> F[语义中心线与多组件提取]
    A --> G[颜色 / Hough 线候选]
    E --> H[候选沿线语义概率门控<br/>mean >= 0.40<br/>P >= 0.10 覆盖率 >= 0.50]
    G --> H
    H --> I[语义 + 颜色组件并集]
    F --> I
    I --> J[端点桥接候选<br/>距离与切线方向门控]
    J --> K[连续 2 帧确认后合并]
    A --> L[Lucas-Kanade 光流传播<br/>前后向误差门控]
    K --> M[当前观测与光流加权融合]
    L --> M
    C --> M
    D --> N[审核元数据]
    M --> N
    N --> O[三分类方向识别]
    O --> P[视频 / JSONL / 审核图 / run.json]
```

## 2026-08-04 版本1

```mermaid
flowchart TD
    A[视频帧] --> B[YOLO 悠悠球检测]
    B --> C[ByteTrack ID 跟踪]
    A --> D[YOLO Pose 手腕与人体关键点]
    A --> E[LR-ASPP 绳线语义分割<br/>960x544 / 每帧推理]
    E --> F[语义中心线与多组件提取]
    A --> G[颜色 / Hough 线候选]
    E --> H[候选沿线语义概率门控<br/>mean >= 0.40<br/>P >= 0.10 覆盖率 >= 0.50]
    G --> H
    H --> I[语义 + 颜色组件并集]
    F --> I
    C --> J[最近悠悠球上下文<br/>0.25 s 宽限]
    J --> K[允许非冷启动语义绳线重建状态]
    I --> K
    A --> L[Lucas-Kanade 光流传播<br/>前后向误差门控 / 最多 12 帧]
    K --> M{当前帧是否有新鲜观测}
    L --> M
    M -->|是| N[保留当前语义几何<br/>光流仅记录一致性 / 冲突]
    M -->|否| O[使用光流传播几何]
    D --> P[审核元数据]
    N --> P
    O --> P
    P --> Q[三分类方向识别]
    Q --> R[视频 / JSONL / 审核图 / run.json]
```

## 2026-08-04 版本2

```mermaid
flowchart TD
    A[视频帧] --> B[YOLO 悠悠球检测]
    B --> C[ByteTrack ID 跟踪]
    A --> D[YOLO Pose 手腕与人体关键点]
    A --> E[主 LR-ASPP<br/>960x544 / 每帧推理]
    A --> F[副 LR-ASPP<br/>960x544 / 每帧推理]
    E --> G[主语义概率图<br/>阈值原点 0.3985]
    F --> H[副语义概率图<br/>阈值原点 0.50]
    G --> I[阈值相对 logit 校准]
    H --> I
    I --> J[校准 logit 加权融合<br/>主 0.7 / 副 0.3]
    J --> K[融合语义概率图<br/>候选阈值 0.50]
    K --> L[语义中心线与多组件提取]
    A --> M[颜色 / Hough 线候选]
    K --> N[候选沿线融合概率门控<br/>mean >= 0.40<br/>P >= 0.10 覆盖率 >= 0.50]
    M --> N
    L --> O[语义 + 颜色组件并集]
    N --> O
    C --> P[最近悠悠球上下文<br/>0.25 s 宽限]
    O --> Q[非冷启动语义绳线观测]
    P --> Q
    A --> R[Lucas-Kanade 光流传播<br/>前后向误差门控 / 最多 12 帧]
    Q --> S{当前帧是否有新鲜观测}
    R --> S
    S -->|是| T[保留当前观测几何<br/>光流仅记录一致性 / 冲突]
    S -->|否| U[使用光流传播几何]
    D --> V[审核元数据]
    T --> V
    U --> V
    V --> W[三分类方向识别]
    W --> X[视频 / JSONL / 审核图 / run.json]
```

## 2026-08-05 版本3

图中“观测优先”表示新鲜语义分割几何不会被光流结果直接形变；光流只在没有新鲜观测时作为短时补全，并记录前后向一致性与冲突元数据供审核。

```mermaid
flowchart TD
    A[视频帧<br/>保留源分辨率] --> B[YOLO11s 悠悠球常规检测<br/>imgsz 1024 / augment=false]
    B --> E[常规检测候选]
    E --> F[可信上一帧框 + 置信度联合选择<br/>多悠悠球时优先时间连续性]
    F --> G[ByteTrack 稳定 track id]

    A --> H[YOLO Pose<br/>人体与手腕关键点]
    G --> I[悠悠球上下文<br/>最近可信框 / 0.25 s 宽限]
    H --> J[时间连续的人体身份选择]

    A --> K[默认主 LR-ASPP<br/>960x544]
    A --> L[固定副 LR-ASPP<br/>960x544]
    A --> KA[弱域主 LR-ASPP<br/>常驻 / 默认不推理]
    K --> M[当前主概率图<br/>默认阈值原点 0.3985]
    KA --> M
    L --> N[副概率图<br/>阈值原点 0.50]
    M --> O[相对阈值 logit 校准]
    N --> O
    O --> P[默认主 0.7 / 副 0.3 加权融合<br/>弱域主 0.5 / 副 0.5<br/>候选阈值 0.50]
    F --> Q[语义 mask 观测<br/>中心线 / 多组件提取]
    J --> Q
    P --> Q
    A --> R[颜色 / Hough 几何候选]
    P --> S[沿线概率门控<br/>mean >= 0.40<br/>P >= 0.10 覆盖率 >= 0.50]
    R --> S
    S --> T[语义 + 颜色组件并集<br/>review-only]
    Q --> T
    T --> GA[最近 12 次语义观测门控<br/>颜色通过=0 / mean conf&lt;0.82<br/>mean distance ratio&gt;0.018]
    GA --> GB{联合条件满足?}
    GB -->|否| K
    GB -->|是| GC[记录触发帧<br/>下一帧单向切换弱域主]
    GC --> KA

    T --> U{当前帧有新鲜绳线观测?}
    V[上一帧绳线 + 灰度帧] --> W[Lucas-Kanade 光流<br/>ROI / 全帧回退<br/>前后向误差 <= 4 px]
    W --> U
    U -->|是| X[保留新鲜观测几何<br/>记录 flow 一致性 / 冲突]
    U -->|否| Y[使用光流传播几何<br/>最多 12 帧]
    X --> Z[绳线状态与审核元数据]
    Y --> Z
    Z --> V

    F --> AA[方向 ROI：悠悠球 + 手腕 + 绳线<br/>方形 union crop]
    J --> AA
    Z --> AA
    AA --> AB[三分类方向模型<br/>每秒 5 帧 / imgsz 320]
    AB --> AC[跨帧携带最近方向结果<br/>记录 carried / error]

    G --> AD[帧级记录]
    H --> AD
    Z --> AD
    AC --> AD
    AD --> AE[tracked.mp4<br/>frames.jsonl<br/>审核图 / run.json]
```

## 2026-08-06 版本4

```mermaid
flowchart TD
    A[视频帧] --> B[YOLO11s 悠悠球检测]
    B --> C[ByteTrack ID 跟踪]

    A --> D[RTMPose-m WholeBody<br/>人体与 133 点姿态]
    D --> E[时间连续的人体选择]

    A --> F[双路 LR-ASPP 绳线分割]
    F --> G[概率校准、颜色候选门控与中心线提取]
    G --> H[观测优先时序与光流短时补全]

    C --> I[仅悠悠球方形 ROI]
    I --> J[三分类方向模型]

    C --> K[逐帧结果]
    E --> K
    H --> K
    J --> K
    K --> L[tracked.mp4 / frames.jsonl / run.json]
```

## 2026-08-07 版本5（当前默认 Pipeline）

RTMPose-m WholeBody 保留为 Workbench 和 CLI 可选审核分支，默认关闭。当前悠悠球检测、
绳线追踪和仅悠悠球 ROI 的方向分类均不依赖姿态输出。

```mermaid
flowchart TD
    A[视频帧] --> B[YOLO11s 悠悠球检测]
    B --> C[ByteTrack ID 跟踪]

    A --> D[双路 LR-ASPP 绳线分割]
    D --> E[概率校准、颜色候选门控与中心线提取]
    E --> Q{当前帧有新鲜观测?}
    Q -->|是| F[直接保留当前观测]
    Q -->|否| R[Lucas-Kanade 前后向光流]
    R --> P[短时传播几何]

    C --> G[仅悠悠球方形 ROI]
    G --> H[三分类方向模型]

    A -. 显式开启 .-> I[RTMPose-m WholeBody]
    I -. 人体与手部审核元数据 .-> J

    C --> J[逐帧结果]
    F --> J
    P --> J
    H --> J
    J --> K[tracked.mp4 / frames.jsonl / run.json]
```
