# 悠悠球模型与追踪报告：2026-08-07

本报告只覆盖悠悠球检测、绳线分割、三分类方向识别，以及连续视频中的悠悠球/绳线追踪。所有训练、评估和测试均在 `.venv` 中执行；`runs/` 中保留 checkpoint、manifest 和评估 JSON，模型文件使用 SHA-256 标识。

## 悠悠球检测

当前生产权重：`runs/candidates/yoyo_unified_396ce5fa8e73_detection_yolo11s_s-current-capacity60/weights/best.pt`。

- 权重 SHA-256：`ac76000388bc81f442e860b6aac68487406205f27e89f31aab16e0c52e82f705`；YOLO11s，约 9.4M 参数。
- 当前扩展 test（65 张、57 个正样本）重评估为 Precision `0.9522`、Recall `0.8596`、mAP50 `0.9415`、mAP50-95 `0.5425`，固定 yoyo recall `0.8772`；结果位于候选目录的 `test_metrics_external_14e51d6c595f.json`。原生 60 张 test 的 mAP50-95 为 `0.5411`，扩展后保持稳定。
- Workbench 默认输入尺寸为 `1024`。候选在 1280 的单图推理约 `12.1 ms`，降到 1024 约 `8.9 ms`；1024 test 仍为 mAP50 `0.9240`、mAP50-95 `0.5168`、Recall `0.8269`，仍超过旧生产模型。
- 1024 常规推理在 454 帧五段连续审核区间上 pooled Precision `1.0`、Recall `0.990544`、F1 `0.995249`、mean IoU `0.837197`，detector-only 汇总速度约 `5.9737 FPS`。
- 当前 YOLO11s 的五段严格 A/B 中，低置信度 TTA 共尝试 41 帧、接受 9 帧，只净增 1 个 TP，却引入 3 个 FP；pooled Precision/Recall/F1 均为 `0.992908`，速度降到 `5.7466 FPS`。其 F1 和速度均低于常规推理，运行时 TTA 组件、配置、CLI 和元数据字段已全部移除。
- 训练运行：`runs/candidates/yoyo_unified_396ce5fa8e73_detection_yolo11s_s-current-capacity60/run_manifest.json`；最佳 epoch `16`，训练因 15 epoch 无改善于 epoch `31` 早停。`config.yaml`、`config.py` 和 Workbench 视频测试均已切换到该模型与 1024 输入。

当前 detector-only 连续帧复验（reviewed yoyo box）：

| sequence | Recall | F1 | mean IoU | p95 center error | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 周博文，72 帧 | 0.9722 | 0.9859 | 0.8303 | 9.75 px | 2.35 |
| 唐浩翔，100 帧 | 1.0000 | 1.0000 | 0.8350 | 7.42 px | 7.68 |
| DSCF7145，95 帧 | 1.0000 | 1.0000 | 0.8518 | 9.18 px | 7.83 |
| 池高宇，67 帧 | 0.9851 | 0.9925 | 0.8473 | 28.21 px | 8.47 |
| 池高宇前场，120 帧 | 0.9917 | 0.9958 | 0.8297 | 9.00 px | 9.72 |
| 五段 pooled | **0.9905** | **0.9952** | **0.8372** | - | **5.97** |

## 绳线分割

当前生产候选：`runs/candidates/yoyo_unified_0ff7d829e127_semantic_string_adaptive-lr5e6-v1/`。

该候选为自包含的三权重、双路逐帧 LR-ASPP 概率融合：

- 主权重：`weights/primary.pt`，SHA-256 `72bfa24275261248f69ada0325f81876067909468c30105a8f93c92bada508f3`。
- 副权重：`weights/secondary.pt`，SHA-256 `640c4ac5b59c2f70aee1c45ebca78774b78983e4251d03d065267576310223df`。
- 弱域主权重：`weights/adaptive.pt`，SHA-256 `690f3e653c837fe92afcb814d01d55f5ba77c45fae0c85636a4f837459fd8c70`；由原主权重以 `lr=5e-6` 温启动训练，选择 epoch 15。
- 主模型验证阈值 `0.3985`，副模型校准阈值 `0.50`。
- 两路概率先转为相对各自阈值原点的 logit，再按主 `0.7`、副 `0.3` 融合，融合候选阈值为 `0.50`。
- 融合后的概率图继续进入语义中心线、多组件提取和颜色/Hough 几何候选概率门控。双语义模型不使用显式绳色色相标签；颜色/Hough 分支只作为受语义概率约束的补充几何候选。

当前 canonical test 共 65 张，其中 59 张有绳、6 张无绳：

| pipeline | Pixel Dice | tolerant F1, 3 px | presence F1 | 负图平均误检 |
| --- | ---: | ---: | ---: | ---: |
| 单 LR-ASPP | 0.583385 | 0.859735 | 0.983333 | 14.167 px |
| 默认校准双模型融合 | 0.592413 | 0.868790 | **0.983333** | 10.0 px |
| 弱域主模型 + 原副模型，alpha=0.50 | **0.598920** | **0.873985** | **0.983333** | **6.167 px** |

正式候选 manifest：`runs/candidates/yoyo_unified_0ff7d829e127_semantic_string_adaptive-lr5e6-v1/run_manifest.json`。

当前 test 重评估：`runs/experiments/semantic_current_2b0cfca_default_ensemble_a030/test_semantic_metrics_external_42086e82249d.json` 和 `runs/experiments/semantic_current_2b0cfca_adaptive_ensemble_a050/test_semantic_metrics_external_42086e82249d.json`。

弱域主模型不能全局替换原主模型：直接用于所有连续序列会造成旧域回退。因此默认仍使用原主/副模型 `alpha=0.30`；最近 12 次语义观测同时满足颜色概率候选通过数为 0、平均语义 confidence `<0.82`、平均 `distance_to_yoyo_px / frame_diagonal >0.018` 时，才从下一帧单向切换到弱域主模型，并与原副模型按 `alpha=0.50` 融合。每帧仍只推理一个主模型和一个副模型；代价是第三个 checkpoint 常驻显存。

Workbench 默认选择该候选主权重时自动启用配套副权重和弱域主权重；手动选择其他绳线模型时两者均关闭，避免混用不匹配 checkpoint。

## 连续视频追踪

连续帧评估使用 reviewed yoyo box 隔离绳线几何，不混入 detector 定位误差。当前 Pipeline 为：双模型校准融合、语义邻域预筛选后的概率门控颜色/Hough 补线、观测优先时序、仅在无新鲜观测时执行的 Lucas-Kanade 缺帧传播，以及最近悠悠球上下文宽限。

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

当前六组 552 帧使用未改动的生产权重复验；颜色/Hough 在搜索前先将 `p>=0.10` 的 960x544 语义支持区映射到源 ROI，并以 31x31 核膨胀。两段池高宇区间仍触发弱域门控：

| sequence | F1@8 | Precision@8 | Recall@8 | Chamfer px | mean components |
| --- | ---: | ---: | ---: | ---: | ---: |
| 周博文，72 帧 | **0.570148** | 0.668538 | **0.497004** | **19.0606** | 3.1806 |
| 唐浩翔，100 帧 | **0.696003** | 0.798797 | **0.616649** | **23.4859** | 3.4500 |
| DSCF7145，95 帧 | **0.495249** | 0.585347 | **0.429188** | **81.8179** | 3.4211 |
| 池高宇，67 帧 | **0.219108** | 0.562262 | **0.136065** | **90.8411** | 2.7015 |
| 池高宇前场，120 帧 | **0.299817** | 0.536124 | **0.208095** | **47.2157** | 2.2750 |
| 华南赛，98 帧 | **0.621008** | 0.792823 | **0.510398** | **16.2471** | 1.3061 |
| 六组 pooled | **0.490075** | **0.670162** | **0.386274** | **45.2064** | - |

池高宇仍是当前最弱域，主要瓶颈是复杂多段绳线召回。当前正式复验：`runs/experiments/semantic_color_prefilter_d31_552/summary.json`。

弱域滑动门控仍只在两个池高宇区间触发，周博文、唐浩翔、DSCF7145 和华南赛四组均不触发；语义邻域预筛选没有改变主模型切换范围。弱域权重、12 帧联合门控和下一帧单向激活规则均保持不变。

推理流程优化：主/副 LR-ASPP 现在共享一次 letterbox、归一化和 GPU 输入张量，只执行两次模型前向。四段 334 帧的四个 `frames.jsonl` SHA-256 与优化前逐字节一致；双模型微基准中位耗时由 `18.88 ms` 降到 `18.20 ms`（约 `3.6%`），没有改变任何绳线指标。Workbench 30 帧 smoke 仍成功生成 MP4/JSONL，优化证据目录为 `runs/experiments/semantic_shared_preprocess_equivalence_temporal_all/` 和 `runs/experiments/workbench_shared_preprocess_smoke/`。

颜色补线的语义参考线重采样已从逐 Hough 候选移到逐帧执行。真实 4K 帧的颜色观测微基准由 `161.6 ms` 降到 `141.8 ms`；454 帧五段评估墙钟时间由 `83.8 s` 降到 `79.4 s`。前三段及池高宇前场的 `frames.jsonl` SHA-256 与优化前一致，先前被同名文件覆盖的 67 帧池高宇区间则验证全部汇总指标对象一致。30 帧无 pose 对照的 JSONL 逐字节一致，跟踪循环由 `17.36 s` 降到 `13.15 s`；证据位于 `runs/experiments/semantic_adaptive_current_454_color_reference_hoist/` 和 `runs/experiments/workbench_color_reference_hoist_smoke/`。

进一步等价优化移除了 Hough 每个候选的临时端点数组和重复范数计算，四个端点距离只计算一次；同时跟踪器和连续评估复用当前帧灰度图，不再在光流估计后重复转换 4K 帧。真实帧 20 次颜色观测微基准的中位耗时由 `172.2 ms` 降到 `161.4 ms`，输出 SHA-256 一致。同机 30 帧无 pose/方向严格 A/B 为 `11.87 s -> 11.35 s`（约 `4.4%`），JSONL 逐字节一致；454 帧完整评估由 `79.4 s` 降到 `75.8 s`（约 `4.5%`），五个 `frames.jsonl` 全部逐字节一致，五份 metrics 除输出目录路径外无差异。证据位于 `runs/experiments/workbench_scalar_gray_ab_baseline/`、`runs/experiments/workbench_scalar_gray_ab_candidate/` 和 `runs/experiments/semantic_adaptive_scalar_gray_reuse_454/`。

最新流程将 Lucas-Kanade 从“每帧先计算、仅在缺观测时采用”改为真正的按需缺帧传播：有新鲜语义/颜色观测时直接保留观测，无观测时才执行前后向光流。扩展后的 6 组 552 帧严格 A/B 中，六组完整指标对象和逐帧绳线几何字段全部一致，墙钟时间由 `108.6057 s` 降至 `85.9556 s`（约 `20.86%`）。仅移除了 88 帧不参与几何输出的观测/光流分歧审核字段；对应的无作用 `fusion_distance` 配置、CLI 参数和 bad-case 分支也已删除。默认 30 帧视频 smoke 的悠悠球、绳线几何和方向输出同样逐帧一致，tracking loop 为 `24.2451 s -> 14.6973 s`（`1.2374 -> 2.0412 FPS`），且所有 MP4/JSONL/审核产物成功生成。证据位于 `runs/experiments/semantic_flow_defer_ab_baseline/`、`runs/experiments/semantic_flow_defer_ab_candidate/` 和 `runs/experiments/flow_defer_default_smoke/`。

在上述按需光流基础上，进一步延迟 4K 灰度转换：有新鲜语义/颜色观测的帧不再执行无用的 `cvtColor`，只在需要缺帧传播时从当前帧和上一帧原图生成灰度。六组 552 帧严格 A/B 的六个 `frames.jsonl` SHA-256 全部逐字节一致；同一设备、同一参数墙钟由 `61.8628 s` 降至 `59.5663 s`（约 `3.71%`），六组 F1@8、Recall@8、Chamfer 和组件数均不变。默认 Workbench 30 帧 smoke 的 JSONL 也与优化前逐字节一致，tracking loop 为 `6.5039 s`（`4.6126 FPS`），完整产物生成成功。证据位于 `runs/experiments/semantic_lazy_gray_ab_baseline/`、`runs/experiments/semantic_lazy_gray_ab_candidate_timed/` 和 `runs/experiments/lazy_gray_default_smoke/`。

双 LR-ASPP 的阈值校准融合现保留在模型设备上执行，只将最终一张融合概率图传回 CPU；原路径会先传回两张概率图，再以 NumPy 逐像素执行 `clip + logit + sigmoid`。65 张 canonical test 的 `p>=0.50`、`p>=0.10` 二值图均零像素变化，全部静态指标一致。六组 552 帧完整 metrics 全部一致，JSONL 仅两处审核浮点值变化（最大绝对差 `1e-4`），没有几何、组件、方法或自适应门控差异；同机墙钟由 `59.4119 s` 降至 `51.3375 s`（约 `13.59%`）。默认 Workbench 30 帧 JSONL 逐字节一致，tracking loop `6.5039 -> 5.8240 s`（`4.6126 -> 5.1511 FPS`）。证据位于 `runs/experiments/semantic_gpu_fusion_ab_baseline/`、`runs/experiments/semantic_gpu_fusion_ab_candidate/` 和 `runs/experiments/gpu_fusion_default_smoke/`。

颜色 mask 现只在语义支持包围盒及形态学所需的 4 像素依赖边界内执行 HSV、阈值与 3×3 开闭运算，再放回原尺寸供 Hough 使用；支持区外在原路径中本来就会被清零。随机图像边界测试逐像素一致，六组 552 帧的六个 JSONL SHA-256 也全部一致；同机墙钟 `52.3525 -> 48.0834 s`（约 `8.15%`）。默认 Workbench 30 帧 JSONL 逐字节一致，tracking loop `5.8240 -> 5.2586 s`（`5.1511 -> 5.7050 FPS`）。证据位于 `runs/experiments/semantic_color_mask_crop_ab_baseline/`、`runs/experiments/semantic_color_mask_crop_ab_candidate/` 和 `runs/experiments/color_mask_crop_default_smoke/`。

颜色/Hough 现先由语义支持邻域裁掉无关饱和舞台线，再进行候选搜索和原有沿线概率门控。同机 552 帧 A/B 的每组 F1@8、Recall@8 均上升、Chamfer 均下降，pooled F1 `0.461017 -> 0.490075`、Recall `0.356170 -> 0.386274`、帧均 Chamfer `50.5553 -> 45.2064 px`；presence 六组不变。墙钟由 `80.5 s` 降至 `64.2 s`（约 `20.25%`）。默认 Workbench 30 帧真实视频中，悠悠球与方向逐帧一致、bad-case 计数一致，tracking loop `14.7743 -> 7.6644 s`（`2.0306 -> 3.9142 FPS`）。证据位于 `runs/experiments/semantic_color_prefilter_d31_ab_baseline_552/`、`runs/experiments/semantic_color_prefilter_d31_552/`、`runs/experiments/color_semantic_prefilter_workbench_ab_baseline/` 和 `runs/experiments/color_semantic_prefilter_workbench_ab_candidate/`。

连续评估产物改用唯一 `group_id` 加短 SHA-256 命名，修复同一 `source_group` 的两个区间互相覆盖 `frames.jsonl`/metrics 的问题；模型推理与指标计算不变。

## 三分类方向识别

最终视图位于 `datasets/1Ayoyo_dataset/orientation_roi`，view id 为 `orientation_roi_9cd9d9361ab5`。训练/验证/测试分别为 810/65/65 张分类图像，且 manifest 明确声明不依赖手部和绳线输入。

| 方案 | Top-1 | Macro recall | horizontal | normal | not_applicable |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧上下文 ROI + 旧模型 | 0.8000 | 0.6562 | 0.5000 | 0.9130 | 0.5556 |
| 仅悠悠球 ROI + 旧模型零样本 | 0.7846 | 0.7048 | 0.6000 | 0.8478 | 0.6667 |
| 仅悠悠球 ROI + 最终模型 | **0.9231** | **0.8818** | **0.8000** | **0.9565** | **0.8889** |

最终权重：

`runs/candidates/yoyo_unified_2b0cfca8743a_orientation_roi_9cd9d9361ab5_best_yoyo-only-final-warm-freeze10-lr1e4-v1/weights/best.pt`

权重 SHA-256：`f00e3766c05d9ae7dc3fe13a9cd45faf3507aab4c9a9acfa6df73b155ff7cd91`。正式 test 结果位于同一 run 的 `test_metrics.json`，数据集 manifest 与 checkpoint 完全匹配。

### 连续帧方向稳定化

单帧权重和 ROI 保持不变；本轮只优化连续视频中的因果时序推理。评估直接使用 `datasets/1Ayoyo_consecutive` 的 6 组 reviewed 标注、552 帧和 reviewed yoyo box，避免 detector 框误差污染方向策略比较。晋升门槛要求 pooled accuracy、Macro recall、每组 accuracy 均不下降，并且预测切换数不增加。

| 运行路径 | 方向推理次数 | Accuracy | Macro recall | 预测切换 | 超额切换 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧默认：固定 5 FPS、直接 carry | 57 | 0.882246 | 0.850290 | 13 | 9 |
| **当前默认：5/25 FPS 自适应 + EMA/滞回** | **117** | **0.942029** | **0.922041** | **5** | **1** |

当前默认使用以下因果策略：

- 三类概率 EMA `alpha=0.4`；候选类相对当前类至少领先 `0.05`，连续 3 次确认后切换。
- 当候选类置信度至少 `0.9` 且相对当前类领先 `0.2` 时允许快速切换。
- 稳定状态保持 5 FPS；初始化、低于 `0.5` 置信度、原始/稳定标签冲突、待确认或刚切换时升到 25 FPS，连续 4 次稳定后回到 5 FPS。
- 六组序列 accuracy 逐组均未下降；pooled recall 从 horizontal/normal/not_applicable `0.710059/0.965812/0.875000` 提升为 `0.834320/0.994302/0.937500`。

真实 detector 框 A/B 使用 DSCF7145 的 95 帧双边界片段。旧路径方向推理 10 次，当前自适应路径 21 次，其中 14 次处于 burst：

| 路径 | Accuracy | Macro recall | 平均/最大边界延迟 | 输出切换 | tracking loop FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧 5 FPS | 0.947368 | 0.935988 | 2.5 / 3 帧 | 2 | 6.9078 |
| **自适应 EMA/滞回** | **0.968421** | **0.968246** | **1.5 / 2 帧** | **2** | 6.8767 |

两次运行的悠悠球 presence、定位和运动指标完全一致；在关闭 pose 和语义绳模型的同配置 tracking loop 中，自适应方向的吞吐差为 `0.45%`。正式离线证据位于 `runs/experiments/orientation_temporal_adaptive_final_20260807/metrics.json`，真实视频证据位于 `runs/experiments/orientation_runtime_ab/`。Workbench 默认继续使用上述最新方向权重，并自动启用自适应滤波。

## RTMPose 运行时评估

30 帧连续视频 smoke test 使用项目内 RTMPose-m WholeBody 与 YOLOX-m ONNX 模型：

- pose 成功：30/30 帧。
- 双手完整 21 点：60/60 组。
- 完整 pipeline：1.7081 FPS，17.5632 秒。
- 旧运行时姿态基线：2.0642 FPS，14.5332 秒。

RTMPose 约慢 17%，但提供 133 点 WholeBody 输出；可视审核确认选中了前景表演者。所有下载模型均位于 `models/rtmpose`，C 盘 RTMLib checkpoint 缓存为空。

切换最终方向权重并统一运行时裁剪后，又执行了 30 帧完整 smoke：pose 30/30 帧成功、双手 30/30 帧完整、方向模型按 cadence 推理 3 次，所有方向记录均使用 `yoyo_bbox_square_3p0_min_12pct; no_yoyo_center_square_28pct`。本次完整 pipeline 为 1.5526 FPS，产物位于 `runs/experiments/rtmpose_yoyo_only_orientation_final_smoke`。

由于当前方向模型只读取悠悠球 ROI，绳线几何也不使用手部硬门控，RTMPose 不再参与本报告四项主任务的预测。当前代码在同一 30 帧片段上的严格开关 A/B 中，关闭 RTMPose 后悠悠球、绳线、方向以及排除 pose-only 标记后的 bad-case 输出逐帧完全一致；tracking loop 从 `0.9989 FPS` 提升至 `1.6117 FPS`（`+61.35%`，`30.0327 s -> 18.6136 s`）。证据位于 `runs/experiments/pose_default_ab_on/` 和 `runs/experiments/pose_default_ab_off/`。

因此 Workbench 和 CLI 继续完整保留 RTMPose 选项与 `--pose/--no-pose` 开关，但默认配置改为关闭，避免为不参与当前主任务输出的审核元数据支付每帧推理成本。需要人体和手部关键点时可在 Workbench 中显式勾选。不传姿态开关的 30 帧默认 smoke manifest 已确认 `pose_enabled=false`，全部视频、JSONL 和审核产物成功生成；其悠悠球、绳线与方向逐帧哈希均与显式 `--no-pose` 运行一致，证据位于 `runs/experiments/pose_default_off_smoke/`。

## 验证状态

- `compileall` 通过。
- `pytest -q`：147 项全部通过；`pytest.ini` 将项目测试范围固定为 `tests/`。
- 结构化扫描：两个数据集共 914 个 canonical JSON，pose/手部键残留数为 0。

## 复现命令

```powershell
.\.venv\Scripts\python.exe -m training_v3.download_rtmpose_models
.\.venv\Scripts\python.exe -m training_v3.strip_pose_annotations
.\.venv\Scripts\python.exe -m training_v3.orientation_view --dataset-dir datasets\1Ayoyo_dataset --clear
.\.venv\Scripts\python.exe -m training_v3.evaluate runs\candidates\yoyo_unified_2b0cfca8743a_orientation_roi_9cd9d9361ab5_best_yoyo-only-final-warm-freeze10-lr1e4-v1 --device 0
.\.venv\Scripts\python.exe -m cli.tracking.evaluate_orientation --output-dir runs\experiments\orientation_temporal_adaptive_20260807 --device 0
.\.venv\Scripts\python.exe -m pytest -q
```
