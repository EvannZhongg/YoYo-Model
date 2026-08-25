# 悠悠球模型与追踪报告：2026-08-26

## 2026-08-26 绳线模型增量微调晋升

在更新后的 `datasets/1Ayoyo_dataset`（string manifest `7f661d3a5cfd3a3f8ae1cf2576192097c0a44643d75493cd3847b1397d3a5a7c`）上，以当前 MobileNetV3-FPN 权重为初始化，使用 `lr=1e-5`、backbone 学习率倍率 `0.05`、前 3 epoch 冻结 backbone、负样本采样权重 `4` 微调 13 个 epoch，验证集 best 为 epoch 11、阈值 `0.3488`。晋升权重为：

`runs/experiments/semantic_new116_ft_r1_lr1e5/weights/best.pt`

SHA-256：`ab2049c2f8abe0b1d43a77a9093db05114a19ee1e0adc1bf3f8deee3489af0de`。

新 test split 同口径结果：pixel Dice `0.727252 -> 0.736401`，Tolerant F1@3 `0.936931 -> 0.944534`，Presence F1 `0.980198 -> 0.980198`，负图平均误检像素 `75.333 -> 45.889`。连续集 856 帧 pooled F1@8 `0.607372 -> 0.625574`，Chamfer `82.58 -> 42.33 px`，HD95 `262.66 -> 123.05 px`；最弱邬聪聪组 F1@8 `0.637564 -> 0.818828`、Chamfer `474.78 -> 141.46 px`。唐浩翔、华南赛和池高宇 100 帧组 F1 有 `0.005-0.013` 的局部回退，但 Chamfer/HD95 与 pooled 指标显著改善，按全局收益规则保留并记录。`960x544` 单次 GPU 前向约 `7.62 -> 6.77 ms`；使用检测、绳线和方向默认配置对 `videos/2023华南赛1Afinal/2023 Southern China Yoyo Contest 1A Final 2nd 杨云帆 _ Film by CYML.mp4` 前 120 帧的完整追踪 loop 为 `24.897 FPS`，参数量和推理流程不变。

实验 manifest、test 指标和连续集逐组结果分别保留在 `runs/experiments/semantic_new116_ft_r1_lr1e5/run_manifest.json`、`runs/experiments/semantic_new116_ft_r1_lr1e5/test_semantic_metrics.json` 和 `runs/experiments/semantic_new116_ft_r1_lr1e5_consecutive856/summary.json`。默认 `tracking.string_weights_path` 已同步更新。

## 2026-08-26 新增数据复评与检测器晋升

`datasets/1Ayoyo_dataset` 已更新为 manifest `yoyo_unified_57935af9dc69`（722 张，新增审核样本 116 张）。旧生产检测器在同一新 test split 上为 Precision `0.987401`、Recall `0.898990`、mAP50 `0.959212`、mAP50-95 `0.574515`。以该权重初始化、在完整 train split 上以 AdamW、`lr0=5e-6` 微调 8 epoch 后，验证集选择旧/新权重线性 soup `alpha=0.75`，得到默认检测权重：

`runs/experiments/det_soup_new116_a0p75/weights/best.pt`

SHA-256：`13b66a5c6eda03deb7f5bc4b1efc60a273df5aae8d314424a79bf2fb5b7029b8`。

新 test split 同口径结果为 Precision `0.976794`、Recall `0.898990`、mAP50 `0.968342`、mAP50-95 `0.589861`；相对旧权重 mAP50-95 提升 `0.015346`，召回不变。`datasets/1Ayoyo_consecutive` 的 856 帧直接检测复核中，Presence F1 `0.976280`、Mean IoU `0.792355`、中心误差 `20.7396 px`；旧生产候选对应的复核为 Presence F1 `0.958678`、Mean IoU `0.771622`、中心误差 `33.0983 px`。模型结构和参数量不变，1024 输入推理约 `7.5 ms/frame`，未引入额外推理组件。验证、test 与连续帧结果分别保留在 `runs/experiments/yoyo_unified_57935af9dc69_detection_best_incremental_new116_r1_lr5e6/test_metrics.json`、`runs/experiments/det_soup_new116_a0p75/test_metrics.json` 和 `tmp/det_soup_a75_consecutive`。

方向模型在 856 帧连续集上的新基线复评仍为 Accuracy `0.855140`、Macro Recall `0.803199`；现有时序候选为 `0.903037`/`0.864272`，但逐组非下降门禁未通过，因此保持原方向权重与时序配置。

本报告只记录当前有效结论，覆盖悠悠球检测、绳线分割、三分类方向识别，以及连续视频中的悠悠球/绳线追踪。训练与评估均在 `.venv` 中执行；详细 checkpoint、manifest 和逐帧结果保留在 `runs/`。

## 当前默认方案

| 任务 | 当前方案 | 默认行为 |
| --- | --- | --- |
| 悠悠球检测 | YOLO11s，约 9.4M 参数 | `imgsz=1024`，常规单次推理 |
| 绳线分割 | 单 MobileNetV3-FPN，约 3.0M 参数 | `960x544` 单次推理，验证阈值 `0.6471` |
| 绳线追踪 | 语义分割 + 概率门控颜色/Hough + 按需光流 | 新鲜观测优先，仅缺帧时传播 |
| 方向识别 | 仅悠悠球 ROI 的三分类模型 | 5/25 FPS 自适应采样 + EMA/滞回 |
| RTMPose | 可选审核分支 | 默认关闭，不参与上述四项主任务输出 |

Workbench 默认模型和参数与 `config.yaml`、`config.py` 一致。

## 悠悠球检测

生产权重（本机发布产物，GitHub 提交不包含二进制文件）：

`runs/candidates/yoyo_detection_hardneg_4f4fb0ee4e66_detection_yolo11s_soup-a20-v1/weights/best.pt`

SHA-256：`4913dcf70784c75229282a1a31d1ed124a65f7b334d1a1cc4fb7775f117cabcd`。部署机器需将该本机权重放在上述路径；仓库只提交 manifest 与指标记录。

候选仅从 train split 挖掘上一生产权重在人工悠悠球框 `IoU<0.1`、置信度 `>=0.25` 的误检，目视核对后生成 22 张背景强化图；训练视图为 424 张原训练图加 22 张硬负样本，val/test 与来源组均保持不变。YOLO11s 以 `lr=5e-6` 微调 3 epoch，并在 val 上选择上一生产权重 `0.8` 与微调权重 `0.2` 的参数插值。初始化 lineage 与 val/test source group 无交集。当前 91 张 test（82 个正样本）、`imgsz=1024` 的同口径对比如下：

| 权重 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| 上一生产 YOLO11s | 0.981028 | 0.853659 | 0.947575 | **0.566066** |
| **当前硬负样本 YOLO11s** | **0.986283** | **0.876891** | **0.952891** | 0.561644 |

未参与微调的 856 张 reviewed 连续帧上，Presence F1 `0.953295 -> 0.958678`、Recall `0.919753 -> 0.930864`、Mean IoU `0.769834 -> 0.771622`、中心误差 `36.3070 -> 33.0983 px`。FP `8 -> 9`，但 FN `65 -> 56`；test mAP50-95 的 `-0.004422` 是本次晋升保留的局部代价。

邬聪聪 99 帧完整 pipeline A/B 只替换检测权重：悠悠球 Presence F1 `0.828025 -> 0.832298`、Recall `0.755814 -> 0.779070`、Mean IoU `0.270465 -> 0.320123`、中心误差 `280.0470 -> 267.0471 px`；FP `6 -> 8`。绳线 F1@8 `0.611659 -> 0.615666`、Chamfer `476.6630 -> 473.4851 px`。同架构连续集推理耗时 `7.3065 -> 7.2953 ms/frame`，参数量与生产输入不变。

## 绳线分割

生产候选（本机发布产物，GitHub 提交不包含二进制文件）：

`runs/candidates/yoyo_unified_b36a77f2e354_semantic_string_mobilenetv3-fpn-single-v1/weights/best.pt`

SHA-256：`8b953763c6102e767d060fc7af7d13dcff353fa8ef5c73d03cdf36ee4dda4468`。

模型以 ImageNet MobileNetV3-Large 为编码器，用 `/2、/4、/8、/16、/32` 特征构建轻量 FPN。训练前 5 个 epoch 冻结编码器；随后解冻，编码器学习率为解码器的 `0.05x`。小 batch 训练期间固定编码器 BatchNorm 统计，避免随机解码头扰动预训练分布。最佳 epoch 为 36，参数量约 `3.02M`。

65 张 canonical test（59 张有绳、6 张无绳）的代表性结果：

| 方案 | Pixel Dice | Tolerant F1@3 | Presence F1 | 负图平均误检 |
| --- | ---: | ---: | ---: | ---: |
| 单 LR-ASPP | 0.583385 | 0.859735 | 0.983333 | 14.167 px |
| 普通域校准双路 | 0.592413 | 0.868790 | 0.983333 | 10.000 px |
| 旧弱域权重选择结果 | 0.599546 | 0.877097 | 0.983333 | **4.333 px** |
| **单 MobileNetV3-FPN** | **0.713177** | **0.925824** | 0.983333 | 43.333 px |

FPN 在像素和容差几何上有大幅提升，59 张正图的 presence recall 为 `1.0`。6 张负图中旧方案和新方案都误判 2 张；新方案误检面积更大，因此负图像素误检是明确局部回退。生产流程仍要求悠悠球/历史绳线锚点，并在连续集上单独验证误检与几何。

## 连续视频追踪

评估使用 `datasets/1Ayoyo_consecutive` 的 reviewed yoyo box，以隔离绳线策略和 detector 误差。当前流程为：单 FPN 语义分割、语义支持邻域内的饱和色/亮脊 Hough、沿线概率门控、观测优先状态更新，以及只在缺少新鲜观测时执行的 Lucas-Kanade 前后向光流。不再加载副模型、弱域模型或执行分域路由。

最终 8 组、757 帧结果：

| 序列 | 帧数 | 旧 F1@8 | 新 F1@8 | 旧 Chamfer | 新 Chamfer |
| --- | ---: | ---: | ---: | ---: | ---: |
| 周博文 | 72 | 0.584083 | **0.657621** | 18.8245 | **17.3310** |
| 唐浩翔 | 100 | 0.697869 | **0.703379** | **23.2869** | 33.1537 |
| DSCF7145 | 95 | **0.495102** | 0.479661 | **81.9063** | 128.3117 |
| 池高宇，67 帧 | 67 | 0.413072 | **0.485520** | 64.7856 | **43.1967** |
| 池高宇前场 | 120 | 0.526797 | **0.562186** | **24.0786** | 29.3198 |
| 华南赛 | 98 | 0.692544 | **0.744656** | **11.6624** | 11.6659 |
| 池高宇 f4200-f4299 | 100 | 0.199450 | **0.470934** | 446.0609 | **24.3131** |
| namdongxun | 105 | 0.655378 | **0.735113** | 22.7764 | **13.5036** |

Pooled Precision/Recall/F1@8 从 `0.711532/0.443424/0.546359` 提升至 `0.810104/0.459328/0.586252`；帧均 Chamfer/HD95 约从 `88.29/195.68 px` 降至 `37.30/118.46 px`。最弱的池高宇 f4200-f4299 区间 F1 提升 `136%`，Chamfer 降低 `94.5%`。

DSCF7145 的 F1 和 Chamfer 明确回退；唐浩翔、池高宇前场的 Chamfer 也有局部回退。保留候选的依据是 pooled 五项指标全部提升、最弱域大幅改善、6/8 组 F1 提升，以及模型数和运行成本同时下降。结果位于 `runs/experiments/semantic_mobilenetv3_fpn_single_production_757/summary.json`。

### 运行时优化

`960x544` 的直接 GPU 基准中，单 FPN 为 `7.05 ms`，两个 LR-ASPP 前向合计约 `8.85 ms`。`namdongxun` 105 帧真实 detector Workbench A/B 中，旧三模型为 `8.5255/8.5335 FPS`，单 FPN 为 `10.0651/8.8566 FPS`；两次候选输出 MP4 SHA-256 均为 `e40854b86e9651c73e7f027b7e76e441cf4aca8d2ba248b32dfe349ce9125241`。

继续保留的通用优化包括：检测期间并行 semantic letterbox、语义支持 ROI 内颜色/Hough、饱和色/亮脊缓存复用、按需光流、异步有界 MP4 写入，以及无锚点时跳过语义前向。旧 LR-ASPP 专用的 GPU 融合、双路 CUDA Graph、弱域门控和三档校准不再进入默认流程。

## 三分类方向识别

数据视图：`datasets/1Ayoyo_dataset/orientation`，训练/验证/test 为 810/65/65 张，只使用悠悠球 ROI。

生产权重：

`runs/candidates/yoyo_unified_2b0cfca8743a_orientation_roi_9cd9d9361ab5_best_yoyo-only-final-warm-freeze10-lr1e4-v1/weights/best.pt`

SHA-256：`f00e3766c05d9ae7dc3fe13a9cd45faf3507aab4c9a9acfa6df73b155ff7cd91`。

| 方案 | Top-1 | Macro recall | horizontal | normal | not_applicable |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧上下文 ROI | 0.8000 | 0.6562 | 0.5000 | 0.9130 | 0.5556 |
| 当前 manifest，仅悠悠球 ROI（91 张 test） | **0.9231** | **0.8732** | **0.8462** | **0.9552** | **0.8182** |

当前默认采用三类概率 EMA `alpha=0.5`、切换 margin `0.05`、连续 4 次确认，强切换置信度 `0.9`、margin `0.1`。稳定状态以 5 FPS 推理，不稳定状态提升至 25 FPS，连续 4 次稳定后恢复低频。该配置不增加模型或前向次数。

新增邬聪聪组加入后，连续集为 9 组、856 帧。使用旧生产方向模型和新时序配置的离线回放结果为：

| 策略 | Accuracy | Macro recall | 输出切换 | 超额切换 |
| --- | ---: | ---: | ---: | ---: |
| 旧模型默认配置 | 0.896028 | 0.833289 | 13 | 1 |
| **旧模型 + 新时序配置** | **0.903037** | **0.864272** | **13** | **1** |

使用上一检测器的邬聪聪 99 帧真实 pipeline A/B 中，方向 Accuracy `0.454545 -> 0.494949`、Macro Recall `0.424866 -> 0.448121`、输出切换 `9 -> 5`；悠悠球和绳线指标逐项不变。检测器同步晋升后，当前完整 pipeline 的方向 Accuracy 为 `0.676768`、Macro Recall 为 `0.487478`。完整证据位于 `runs/experiments/orientation_baseline_consecutive856_20260812/metrics.json`、`runs/experiments/orientation_temporal_selected_consecutive856_20260812/metrics.json`、`runs/experiments/orientation_temporal_old_pipeline_wu99_20260812/metrics.json`、`runs/experiments/detection_hardneg_r2_pipeline_wu99/metrics.json`。

## RTMPose

RTMPose-m WholeBody 与 YOLOX-m 仍作为可选人体/手部审核分支保留，但不参与当前检测、绳线或仅悠悠球 ROI 的方向预测。严格开关 A/B 中，关闭 RTMPose 后四项主输出不变，tracking loop 从 `0.9989` 提升至 `1.6117 FPS`，因此 Workbench 与 CLI 默认关闭 pose；需要 133 点姿态审核时可显式启用 `--pose`。

## 验证状态

- 本轮完整验证：`pytest -q`，168 项及 2 个子测试通过。
- `compileall` 与 `git diff --check` 通过。
- 两个数据集共 1462 个 canonical JSON，pose/手部键残留数为 0。

## 复现命令

```powershell
.\.venv\Scripts\python.exe -m training_v3.evaluate runs\candidates\yoyo_detection_hardneg_4f4fb0ee4e66_detection_yolo11s_soup-a20-v1 --dataset-dir datasets\experiments\detection_hardneg_r1 --device 0
.\.venv\Scripts\python.exe -m training_v3.evaluate runs\candidates\yoyo_unified_2b0cfca8743a_orientation_roi_9cd9d9361ab5_best_yoyo-only-final-warm-freeze10-lr1e4-v1 --dataset-dir datasets\1Ayoyo_dataset\orientation_roi --allow-dataset-mismatch --device 0
.\.venv\Scripts\python.exe -m cli.training.train_semantic --dataset-dir datasets\1Ayoyo_dataset\string_segmentation --project runs\experiments --name semantic_degradation_aug_lr5e6_v1 --epochs 20 --input-width 960 --input-height 544 --batch 2 --lr 0.000005 --architecture lraspp_mobilenet_v3 --initial-weights runs\candidates\yoyo_unified_f5775b248d3b_semantic_string_lraspp_soup-a25-v1\weights\best.pt --degradation-augment --early-stopping-patience 5 --early-stopping-min-epochs 6 --device cuda
.\.venv\Scripts\python.exe -m cli.tracking.evaluate_orientation --raw-predictions runs\experiments\orientation_baseline_consecutive856_20260812\raw_predictions.json --output-dir runs\experiments\orientation_temporal_selected_consecutive856_20260812 --device 0
.\.venv\Scripts\python.exe -m pytest -q
```
