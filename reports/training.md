# 悠悠球模型与追踪报告：2026-08-05

本报告只覆盖悠悠球检测、绳线分割、三分类方向识别，以及连续视频中的悠悠球/绳线追踪。所有训练、评估和测试均在 `.venv` 中执行；`runs/` 中保留 checkpoint、manifest 和评估 JSON，模型文件使用 SHA-256 标识。

## 当前数据与协议

- canonical 数据集：`datasets/1Ayoyo_dataset`，dataset id `yoyo_unified_396ce5fa8e73`。
- 当前样本数为 train/val/test = `310/61/60`；训练集包含 24 张新增南京审核图，val/test 内容和来源组保持冻结。
- 主 manifest SHA-256：`e920b8742f0946ccd5550b281f7f8db031402151d36b7c1a436124422e668543`。
- 绳线 view manifest SHA-256：`0ff7d829e1274ef5a49a3a0b225299128ae0df275adb04d1c29ac732886f0167`。
- 方向 ROI view id：`orientation_roi_aee683aec99d`，由当前 canonical manifest 重建；view manifest SHA-256 可从 `datasets/1Ayoyo_dataset/orientation_roi/manifest.json` 校验。
- split 之间按 source group 隔离。连续追踪集 `datasets/1Ayoyo_consecutive` 包含四个视频共 334 帧。

## 悠悠球检测

当前生产权重：`runs/candidates/yoyo_unified_396ce5fa8e73_detection_yolo11s_s-current-capacity60/weights/best.pt`。

- 权重 SHA-256：`ac76000388bc81f442e860b6aac68487406205f27e89f31aab16e0c52e82f705`；YOLO11s，约 9.4M 参数。
- 该候选在冻结的 `datasets/1Ayoyo_dataset` test（60 张、52 个正样本）达到 Precision `0.9492`、Recall `0.8846`、mAP50 `0.9431`、mAP50-95 `0.5411`，固定 yoyo recall `0.9038`。相对旧生产权重的 mAP50-95 从 `0.4353` 提升到 `0.5411`。
- Workbench 默认输入尺寸为 `1024`。候选在 1280 的单图推理约 `12.1 ms`，降到 1024 约 `8.9 ms`；1024 test 仍为 mAP50 `0.9240`、mAP50-95 `0.5168`、Recall `0.8269`，仍超过旧生产模型。
- 1024 常规推理在 334 帧四段连续审核区间上 pooled Recall `0.9901`、F1 `0.9950`、mean IoU `0.8402`；关闭 TTA 的 detector-only 速度为 `5.25 FPS`。启用 TTA 只额外恢复 1 帧，却引入 3 个误检，pooled F1 降至约 `0.992`，因此生产默认 `yoyo_tta_rescue=false`。
- 训练运行：`runs/candidates/yoyo_unified_396ce5fa8e73_detection_yolo11s_s-current-capacity60/run_manifest.json`；最佳 epoch `16`，训练因 15 epoch 无改善于 epoch `31` 早停。`config.yaml`、`config.py` 和 Workbench 视频测试均已切换到该模型与 1024 输入。

当前 detector-only 连续帧复验（reviewed yoyo box）：

| sequence | Recall | F1 | mean IoU | p95 center error | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 周博文，72 帧 | 0.9722 | 0.9859 | 0.8303 | 9.75 px | 2.35 |
| 唐浩翔，100 帧 | 1.0000 | 1.0000 | 0.8350 | 7.42 px | 7.68 |
| DSCF7145，95 帧 | 1.0000 | 1.0000 | 0.8518 | 9.18 px | 7.83 |
| 池高宇，67 帧 | 0.9851 | 0.9925 | 0.8473 | 28.21 px | 8.47 |
| 四组 pooled | **0.9901** | **0.9950** | **0.8402** | - | **5.25** |

## 绳线分割

当前生产候选：`runs/candidates/yoyo_unified_0ff7d829e127_semantic_string_adaptive-lr5e6-v1/`。

该候选为自包含的三权重、双路逐帧 LR-ASPP 概率融合：

- 主权重：`weights/primary.pt`，SHA-256 `72bfa24275261248f69ada0325f81876067909468c30105a8f93c92bada508f3`。
- 副权重：`weights/secondary.pt`，SHA-256 `640c4ac5b59c2f70aee1c45ebca78774b78983e4251d03d065267576310223df`。
- 弱域主权重：`weights/adaptive.pt`，SHA-256 `690f3e653c837fe92afcb814d01d55f5ba77c45fae0c85636a4f837459fd8c70`；由原主权重以 `lr=5e-6` 温启动训练，选择 epoch 15。
- 主模型验证阈值 `0.3985`，副模型校准阈值 `0.50`。
- 两路概率先转为相对各自阈值原点的 logit，再按主 `0.7`、副 `0.3` 融合，融合候选阈值为 `0.50`。
- 融合后的概率图继续进入语义中心线、多组件提取和颜色/Hough 几何候选概率门控。双语义模型不使用显式绳色色相标签；颜色/Hough 分支只作为受语义概率约束的补充几何候选。

当前 canonical test 共 60 张，其中 54 张有绳、6 张无绳：

| pipeline | Pixel Dice | tolerant F1, 3 px | presence F1 | 负图平均误检 |
| --- | ---: | ---: | ---: | ---: |
| 单 LR-ASPP | 0.5843 | 0.8585 | 0.981818 | 14.2 px |
| 默认校准双模型融合 | 0.593939 | 0.868082 | **0.981818** | 10.0 px |
| 弱域主模型 + 原副模型，alpha=0.50 | **0.601069** | **0.873775** | **0.981818** | **6.167 px** |

正式候选 manifest：`runs/candidates/yoyo_unified_0ff7d829e127_semantic_string_adaptive-lr5e6-v1/run_manifest.json`。

当前冻结 test 重评估：`runs/experiments/semantic_calibrated_ensemble_a30_current_static_baseline/test_semantic_metrics_external_0ff7d829e127.json`。

弱域主模型不能全局替换原主模型：直接用于所有连续序列会造成旧域回退。因此默认仍使用原主/副模型 `alpha=0.30`；最近 12 次语义观测同时满足颜色概率候选通过数为 0、平均语义 confidence `<0.82`、平均 `distance_to_yoyo_px / frame_diagonal >0.018` 时，才从下一帧单向切换到弱域主模型，并与原副模型按 `alpha=0.50` 融合。每帧仍只推理一个主模型和一个副模型；代价是第三个 checkpoint 常驻显存。

Workbench 默认选择该候选主权重时自动启用配套副权重和弱域主权重；手动选择其他绳线模型时两者均关闭，避免混用不匹配 checkpoint。

## 连续视频追踪

连续帧评估使用 reviewed yoyo box 隔离绳线几何，不混入 detector 定位误差。当前 Pipeline 为：双模型校准融合、概率门控颜色/Hough 补线、观测优先时序、Lucas-Kanade 缺帧传播和最近悠悠球上下文宽限。

| sequence | pipeline | F1@8 | Precision@8 | Recall@8 | Chamfer px | mean components |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 周博文，72 帧 | 旧生产单模型 | 0.455690 | 0.661079 | 0.347673 | 37.6674 | 2.3472 |
| 周博文，72 帧 | 校准双模型融合 | **0.468227** | **0.670969** | **0.359576** | **36.2188** | 2.3472 |
| 唐浩翔，100 帧 | 旧生产单模型 | 0.649530 | 0.775825 | 0.558597 | 34.8305 | 3.2200 |
| 唐浩翔，100 帧 | 校准双模型融合 | **0.660163** | **0.786047** | **0.569033** | **33.8345** | 3.2400 |
| DSCF7145，95 帧 | 旧生产单模型 | 0.402525 | 0.483110 | 0.344981 | 104.5161 | 4.0211 |
| DSCF7145，95 帧 | 校准双模型融合 | **0.476806** | **0.560270** | **0.414985** | **85.1243** | 3.3368 |

三组 pooled 结果：

| pipeline | Precision@8 | Recall@8 | F1@8 | 帧均 Chamfer |
| --- | ---: | ---: | ---: | ---: |
| 旧生产单模型 | 0.641857 | 0.435180 | 0.518688 | 60.3900 px |
| 校准双模型融合 | **0.674939** | **0.470354** | **0.554373** | **52.7267 px** |

周博文和唐浩翔未参与副模型训练，是连续视频晋升的独立证据。DSCF7145 参与过副模型训练，仅作为已审核序列上的行为检查。

正式连续评估：`runs/experiments/semantic_calibrated_ensemble_a30_temporal_all/summary.json`。

新增池高宇序列后，使用未改动的生产模型与参数重新审核四组连续帧：

| sequence | F1@8 | Precision@8 | Recall@8 | Chamfer px | mean components |
| --- | ---: | ---: | ---: | ---: | ---: |
| 周博文，72 帧 | 0.468227 | 0.670969 | 0.359576 | 36.2188 | 2.3472 |
| 唐浩翔，100 帧 | 0.660163 | 0.786047 | 0.569033 | 33.8345 | 3.2400 |
| DSCF7145，95 帧 | 0.476806 | 0.560270 | 0.414985 | 85.1243 | 3.3368 |
| 池高宇，67 帧 | 0.212071 | 0.520588 | 0.133158 | 97.0596 | 2.8806 |
| 四组 pooled | 0.491748 | 0.662165 | 0.391095 | 61.6198 | - |

池高宇是当前最弱域：人工目标平均 4.194 个可见组件，生产模型平均输出 2.881 个组件，主要瓶颈是复杂多段绳线召回，而不是误检或颜色补线。当前四组重评估：`runs/experiments/semantic_calibrated_ensemble_a30_current_temporal_all/summary.json`。

弱域滑动门控只在池高宇序列触发，其他三段保持原生产路径和指标不变：

| sequence | gate | F1@8 | Chamfer px | motion error px |
| --- | --- | ---: | ---: | ---: |
| 周博文，72 帧 | 未触发 | 0.468227 -> 0.468227 | 36.2188 -> 36.2188 | 81.4570 -> 81.4570 |
| 唐浩翔，100 帧 | 未触发 | 0.660163 -> 0.660163 | 33.8345 -> 33.8345 | 158.4252 -> 158.4252 |
| DSCF7145，95 帧 | 未触发 | 0.476806 -> 0.476806 | 85.1243 -> 85.1243 | 154.6699 -> 154.6699 |
| 池高宇，67 帧 | **触发** | **0.212071 -> 0.218912** | **97.0596 -> 90.9256** | **166.4074 -> 130.2796** |

门控末窗可解释性检查：周博文 confidence `0.8858`、距离比 `0.00846`；唐浩翔颜色通过数 `11`；DSCF7145 颜色通过数 `10`；池高宇 confidence `0.6637`、距离比 `0.05954` 且颜色通过数 `0`。四段任意滑动窗口检查也仅池高宇满足联合条件。打包权重正式复验位于 `runs/experiments/semantic_adaptive_candidate_0ff7d829e127_temporal_all/summary.json`，三个权重 SHA 和每组触发状态、F1@8、Chamfer、motion error 均与晋升实验一致。

同一四组 source-video 区间还用于 detector/TTA 的正式 A/B。baseline 与 cascade 除 TTA rescue 外参数完全一致，绳线、pose 和方向模型均关闭；评估直接对齐 reviewed yoyo box。

| sequence | mode | presence recall | mean IoU | p95 center error | longest miss | ID switches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 周博文，72 帧 | baseline | 0.944444 | 0.763842 | 94.3947 px | 2 | 1 |
| 周博文，72 帧 | cascade | **0.958333** | **0.767245** | **89.7593 px** | 2 | 1 |
| 唐浩翔，100 帧 | baseline | 1.000000 | 0.822435 | 7.5581 px | 0 | 0 |
| 唐浩翔，100 帧 | cascade | 1.000000 | 0.822435 | 7.5581 px | 0 | 0 |
| DSCF7145，95 帧 | baseline | 0.609375 | 0.785900 | 52.9159 px | 14 | 0 |
| DSCF7145，95 帧 | cascade | **0.890625** | **0.841259** | **11.2817 px** | **4** | 1 |
| 池高宇，67 帧 | baseline | 1.000000 | 0.756096 | 46.8398 px | 0 | 0 |
| 池高宇，67 帧 | cascade | 1.000000 | **0.776846** | **40.0224 px** | 0 | 0 |

四组 pooled：presence precision 保持 `1.0`，recall `0.904290 -> 0.966997`，F1 `0.949740 -> 0.983221`，mean IoU `0.786472 -> 0.802675`，IoU50 rate `0.941606 -> 0.959044`，p95 center error `35.7026 -> 25.4175 px`，最长缺失 `14 -> 4` 帧。ID switch 总数 `1 -> 2`，来自 DSCF 新恢复区间；同时有效 track-id 帧数 `237 -> 274`。周博文新增一个较低 IoU 的有效匹配，单序列 IoU50 rate `0.926471 -> 0.913043`，但 recall、mean IoU 和 p95 center error 均改善，不构成实质回退。

以上旧 detector/TTA cascade 是上一代模型的历史证据；当前 YOLO11s 晋升的同口径常规 detector-only 复验见上方检测节，正式产物位于 `runs/experiments/det_s_candidate_1024_*`。旧证据保留用于说明 TTA 策略为何被新模型移除。

## 三分类方向识别

当前生产权重：`runs/candidates/yoyo_unified_582cde69ebb8_orientation_roi_d88241f08b82_best_nyyc36-warm-freeze10-lr1e4-v1/weights/best.pt`。

- 权重 SHA-256：`673786ce7cf1e6a38570f43be5d9db7031705a9f3a3b8f0a47ff6f8a9a705035`。
- 当前 test：Top1 `0.8000`，macro recall `0.6542`。
- 各类 recall：horizontal `0.5000`、normal `0.9070`、not_applicable `0.5556`。
- 当前 canonical 重建 ROI view 的复核结果：`runs/candidates/yoyo_unified_582cde69ebb8_orientation_roi_d88241f08b82_best_nyyc36-warm-freeze10-lr1e4-v1/test_metrics_external_694eeeca30a7.json`。

## 默认 Workbench 视频 smoke test

使用 `videos/NYPC1A/NYPC 嘉宾表演 周博文.mp4` 的连续审核区间，以 YOLO11s 1024 默认 detector、关闭 TTA、双绳模型、pose、概率门控补线和方向模型运行 30 帧。产物位于：

`runs/experiments/workbench_default_smoke_yolo11s_1024/NYPC 嘉宾表演 周博文_20260805T072545Z_37589c5f/`

- MP4、逐帧 JSONL、审核图和 `run.json` 均成功生成。
- `run.json` 记录新 detector 权重 SHA-256 `ac76000388bc81f442e860b6aac68487406205f27e89f31aab16e0c52e82f705`、`imgsz=1024`、`yoyo_tta_rescue=false`，以及 `string_model_kind=semantic_adaptive_ensemble`；MP4 和 JSONL 均成功生成。
- `string_ensemble_alpha=0.3`、融合候选阈值 `0.5`、颜色概率门控、最多 12 帧光流传播和 15 帧最近悠悠球上下文宽限均启用。
- pose 成功启用；绳线模型执行 30 帧，方向模型按 cadence 执行 3 帧并输出 `normal` 汇总。
- TTA 未触发；30 帧完整 pipeline 速度约 `1.91 FPS`。绳线模型执行 30 帧，方向模型按 cadence 执行并输出汇总。

adaptive 候选的 Workbench smoke test：

- 周博文 30 帧：`runs/experiments/workbench_adaptive_smoke_zhou/NYPC 嘉宾表演 周博文_20260805T052129Z_847cb790/`；模型种类为 `semantic_adaptive_ensemble`，三份权重 SHA 与候选 manifest 一致，门控未触发（末窗 confidence `0.8903`、距离比 `0.00718`），完整 pipeline `1.85 FPS`。
- 池高宇 30 帧：`runs/experiments/workbench_adaptive_smoke_chi/池高宇_20260805T052323Z_db659049/`；门控在观测帧 `4757` 触发，下一帧 `4758` 首次使用 adaptive 主模型，末窗 confidence `0.7061`、距离比 `0.06065`，完整 pipeline `3.45 FPS`。
- 两次 smoke 均成功生成 MP4、逐帧 JSONL、审核图和 `run.json`；每帧仍为一个主模型加一个副模型推理。

## 复现命令

```powershell
.\.venv\Scripts\python.exe -m cli.training.evaluate runs\candidates\yoyo_unified_396ce5fa8e73_detection_yolo11s_s-current-capacity60 --dataset-dir datasets\1Ayoyo_dataset --split test --device 0

.\.venv\Scripts\python.exe -m cli.training.evaluate_semantic --weights runs\candidates\yoyo_unified_0ff7d829e127_semantic_string_adaptive-lr5e6-v1\weights\adaptive.pt --ensemble-weights runs\candidates\yoyo_unified_0ff7d829e127_semantic_string_adaptive-lr5e6-v1\weights\secondary.pt --ensemble-alpha 0.50 --ensemble-candidate-threshold 0.50 --dataset-dir datasets\1Ayoyo_dataset\string_segmentation --split test --device cuda --allow-dataset-mismatch --output-dir runs\experiments\semantic_current_domain_warm_lr5e6_static_ensemble_a050

.\.venv\Scripts\python.exe -m string_segmentation.evaluate_consecutive --weights runs\candidates\yoyo_unified_0ff7d829e127_semantic_string_adaptive-lr5e6-v1\weights\primary.pt --ensemble-weights runs\candidates\yoyo_unified_0ff7d829e127_semantic_string_adaptive-lr5e6-v1\weights\secondary.pt --ensemble-alpha 0.30 --ensemble-candidate-threshold 0.50 --adaptive-weights runs\candidates\yoyo_unified_0ff7d829e127_semantic_string_adaptive-lr5e6-v1\weights\adaptive.pt --adaptive-ensemble-alpha 0.50 --adaptive-warmup-frames 12 --adaptive-max-color-accepts 0 --adaptive-max-mean-confidence 0.82 --adaptive-min-mean-distance-ratio 0.018 --dataset-dir datasets\1Ayoyo_consecutive --output-dir runs\experiments\semantic_adaptive_candidate_0ff7d829e127_temporal_all --device cuda --color-augment --color-probability-min-mean 0.40 --color-probability-min-fraction 0.50 --temporal --max-propagation-frames 12 --unanchored-semantic-grace-frames 12

.\.venv\Scripts\python.exe -m cli.tracking.track_video "videos\NYPC1A\NYPC 嘉宾表演 周博文.mp4" --output-dir runs\experiments\workbench_default_smoke_yolo11s_1024 --device 0 --pose --start-seconds 92.66 --max-frames 30

.\.venv\Scripts\python.exe -m cli.dataset.prepare_orientation_view --dataset-dir datasets\1Ayoyo_dataset --clear

.\.venv\Scripts\python.exe -m cli.training.evaluate runs\candidates\yoyo_unified_582cde69ebb8_orientation_roi_d88241f08b82_best_nyyc36-warm-freeze10-lr1e4-v1 --dataset-dir datasets\1Ayoyo_dataset\orientation_roi --split test --device 0 --allow-dataset-mismatch

.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## 验证状态

`.\.venv\Scripts\python.exe -m unittest discover -s tests` 共运行 130 项测试，全部通过。语义训练数据加载已改用字节解码，Windows 中文来源组可正常进入训练。Gradio 测试退出时仍会输出未关闭 event loop 的 `ResourceWarning`，不影响测试结果。
