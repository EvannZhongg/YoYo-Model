# clDice 消融复核（2026-08-31）

## 目的

验证 clDice 是否在当前绳线语义模型中提供独立收益。对照固定为同一
`datasets/1Ayoyo_dataset/string_segmentation/manifest.json`（SHA-256
`f79c9805dae3c91df2ad49eb61f96db31a3236291e505c0925e3aad31f307964`）、
MobileNetV3-FPN、`960x544`、seed `20260830`、12 epochs、batch 2、
冻结 backbone 3 epochs、`hard_negative_weight=0.2`、
`negative_sample_weight=4.0`、`min_mask_width_px=2`。唯一变量是
`cldice_weight`：

- `semantic_ablation_nocldice_r1`: `0.0`
- `semantic_ablation_cldice_r1`: `0.1`

两次训练均为 ImageNet backbone foundation 初始化，未使用 test 来源。评估
阈值只由各自 validation 选择并冻结；连续集协议固定为阈值、颜色/亮脊增强、
语义预筛、最多 12 帧光流传播、前后向误差 4 px、8 个组件、最小组件 8 px、
最多 64 个折线点。

## 结果

| 模型 | test centerline F1@8 | test tolerant F1@3 | test Presence F1 | 连续集 pooled F1@8 | pooled precision / recall | Chamfer (px) | Presence F1 | 最弱来源组 F1@8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 无 clDice | n/a* | 0.934842 | 0.980545 | 0.594536 | 0.759570 / 0.501213 | 32.4349 | 0.992341 | Jakub 0.387233 |
| clDice=0.1 | 0.395391 | 0.932288 | 0.992126 | 0.585197 | 0.758774 / 0.476250 | 36.1325 | 0.992877 | Jakub 0.421168 |
| 当前生产 soup | -- | -- | -- | 0.619236 | 0.779341 / 0.513702 | 29.6636 | 0.988253 | Jakub 0.409559 |

\* 无 clDice 的历史 test JSON 未记录中心线 F1@8；不将其补算或用于排名。

无 clDice 行的静态 test 产物是在中心线指标固化前生成的，因此 test centerline
F1@8 不可用（记为 n/a）；连续集主指标和
几何指标来自 `tmp/semantic_cldice_ablation_summary_20260831.json`。
clDice-only 的 test 结果来自
`tmp/semantic_ablation_cldice_r1_t01749_test/test_semantic_metrics_threshold_0p1749.json`，
连续集逐来源结果与运行元数据来自
`tmp/semantic_ablation_cldice_r1_t01749_consecutive/summary.json`。

clDice-only 连续集安全门槛：最长缺失段 4 帧，最大恢复延迟 4 帧，零预测帧
16；最弱来源组为 Jakub（F1@8 `0.421168`，Chamfer `51.7457 px`）。

## 归因与决定

在完全相同的 foundation 训练配置下，加入 clDice 使连续集 pooled F1@8 从
`0.594536` 变为 `0.585197`（`-0.009339`），recall 从 `0.501213` 降至
`0.476250`，Chamfer 从 `32.4349` 升至 `36.1325 px`；Presence F1 基本不变。
这表明当前 clDice 与已有人为膨胀 mask、Dice/Focal 目标之间存在明显的目标
重叠或竞争，未证明其对中心线追踪有独立正收益。最弱来源组没有改善到生产
护栏以上，不能晋升，也不修改 `config.yaml` 或默认权重。

后续如继续优化，应保持该协议，优先一次只改变一个几何监督项；任何候选都
必须同时通过连续集 pooled F1@8、最弱来源组、Presence/缺失恢复门槛和推理
吞吐检查。
