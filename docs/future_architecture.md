# 悠悠球比赛自动评分模型开发指南

## 基于 Element Tokenizer 的运动语言建模方案

---

# 1. 总体设计思想

本项目将悠悠球比赛视频视为一种连续的**运动语言（Motion Language）**。

传统动作识别通常依赖：

```
视频
 ↓
动作分类
 ↓
动作名称
 ↓
评分
```

但悠悠球存在：

* 动作数量庞大；
* 自创动作较多；
* 元素命名不统一；
* Combo 组合复杂；
* 同一技术不同执行方式评分不同。

因此不将问题定义为：

> 识别某个固定 Trick 类别。

而定义为：

> 从连续悠悠球运动流中学习具有技术意义的运动元素 Token，并根据 Token 的结构和组合关系预测比赛价值。

整体流程：

```
比赛视频

    ↓

视觉感知系统

    ↓

Frame Motion Representation

    ↓

Element Tokenizer

    ↓

Element Token Sequence

    ↓

Element Transformer

    ↓

Score Prediction

    ↓

Competition Score
```

其中：

* Frame Representation 描述每一帧发生了什么；
* Element Tokenizer 负责将连续运动流切分成技术元素；
* Element Token 表示一个完整技术事件；
* Element Transformer 理解元素之间的组合关系；
* Score Model 输出技术价值。

---

# 2. 核心建模理念

## 2.1 将悠悠球动作视为“运动词语”

自然语言：

```
字符流

↓

Tokenizer

↓

Word Token

↓

Sentence Representation
```

悠悠球：

```
Frame Stream

↓

Element Tokenizer

↓

Element Token

↓

Competition Representation
```

对应关系：

| 语言模型            | 悠悠球模型                     |
| --------------- | ------------------------- |
| 字符              | 视频帧                       |
| Tokenizer       | Element Boundary Detector |
| Token Embedding | Element Embedding         |
| Sentence        | 比赛流程                      |
| Language Model  | 比赛评分模型                    |

---

# 3. 视频 Frame Representation

原始视频首先经过视觉模型转换为结构化运动状态。

每一帧：

[
X_t
]

包含：

```
String State
+
YoYo State
+
Pose Context
+
Orientation Context
```

形成：

[
X_1,X_2,...,X_T
]

---

# 4. String-YoYo State Representation（核心）

## 4.1 绳线状态

绳线是主要技术信息来源。

输入来自绳线识别模型：

* segmentation mask；
* polyline；
* curve；
* keypoints；
* fragment。

不要求每帧恢复完整绳线。

原因：

实际比赛存在：

* 遮挡；
* 快速运动；
* 交叉；
* 检测碎片化。

因此采用：

```
Frame

↓

String Fragment Set
```

表示：

[
S_t=
{f_1,f_2,...,f_n}
]

其中：

每个 fragment 包含：

### 几何信息

* 曲线点；
* 长度；
* 方向；
* 曲率；
* 端点位置。

### 结构信息

* fragment 间距离；
* fragment 连接概率；
* crossing 关系。

---

## 4.2 不强制恢复完整绳线

传统方式：

```
fragment

↓

拼接

↓

完整绳线

↓

特征
```

存在风险。

因为错误连接会产生错误拓扑。

因此采用：

```
fragment

↓

soft relation

↓

模型学习真实结构
```

即：

不判断：

```
A和B一定是一根绳
```

而表示：

```
P(A,B属于同一结构)=0.8
```

---

# 5. YoYo Representation

悠悠球不是普通目标检测结果，而是绳线系统中的核心节点。

包含：

## 基础状态

* 位置；
* 速度；
* 加速度；
* 运动方向。

## 关系状态

重点：

```
YoYo ↔ String
```

包括：

* 悠悠球距离绳段；
* 穿越绳线的位置；
* 相对于 crossing 的位置；
* 轨迹变化。

最终：

[
State_t=
String_t+YoYo_t
]

作为主要技术表示。

---

# 6. Pose 与 Orientation Context

人体姿态和方向不作为主要动作表示。

它们描述：

> 动作是在什么条件下完成。

---

## Pose Context

不直接输入全部关键点，而提取语义关系：

例如：

```
手-身体关系

悠悠球-身体关系

绳线-身体区域关系
```

包括：

* 背后；
* 腿下；
* 身体侧方；
* 头顶区域。

---

## Orientation Context

例如：

```
normal
horizontal
unknown
```

或者概率：

```
normal:0.1
horizontal:0.85
unknown:0.05
```

---

# 7. Frame Encoder

所有视觉信息转换为 Frame Embedding。

输入：

```
String Fragment Tokens

+

YoYo Token

+

Pose Token

+

Orientation Token
```

通过：

* Transformer；
* Set Transformer；
* Graph Transformer。

输出：

[
h_t
]

其中：

[
h_t
]

表示：

> 当前时刻悠悠球运动状态。

---

# 8. Element Tokenizer 总体结构

Element Tokenizer 不作为单一模型，而拆成三个模块：

```
Element Tokenizer

    |
    |
    +----------------+
    |                |
    ↓                ↓

Boundary Model    Element Encoder


                     |
                     ↓

              Element Embedding

                     |
                     ↓

              Semantic Model
```

---

# 9. Module 1：Element Boundary Detector

## 目标

学习：

> 连续运动流中的元素边界。

输入：

[
h_1,h_2,...,h_T
]

输出：

两个概率：

### Start Probability

[
P_s(t)
]

### End Probability

[
P_e(t)
]

例如：

```
Frame:

1 2 3 4 5 6 7 8


Start:

0 0 1 0 0 0 1 0


End:

0 0 0 0 1 0 0 1
```

得到：

```
Element 1:

Frame 3-5


Element 2:

Frame 7-8
```

---

## Boundary Model 不负责评分

它只负责：

```
哪里形成一个技术事件
```

类似：

NLP tokenizer：

```
字符
↓

词边界
```

---

# 10. Module 2：Element Encoder

Boundary Model 输出：

```
Element Segment
```

例如：

```
Frame 120-180
```

重新编码。

输入：

[
h_{120},...,h_{180}
]

通过：

Temporal Encoder：

* Transformer；
* Attention Pooling。

输出：

[
z_i
]

其中：

[
z_i
]

就是：

## Element Token Embedding

它表示：

一个完整技术元素的运动语义。

包含：

* 绳线结构变化；
* 悠悠球运动过程；
* 技术复杂度；
* 执行过程。

---

# 11. Element Token 不等于动作名称

系统不需要：

```
Token001 = Eli Hops
Token002 = Brain Twister
```

而学习：

```
Token embedding
```

例如：

```
Element A

[0.23,-0.15,...]


Element B

[0.21,-0.12,...]
```

表示：

两个技术结构相似。

---

# 12. Module 3：Element Semantic Model

该模块负责学习：

> Element Token 的技术价值。

输入：

[
z_i
]

输出：

## Score

[
Score_i=f(z_i)
]

例如：

```
Element Token

↓

Score Head

↓

4.5
```

---

同时可以加入：

## Ranking Learning

对于：

[
Score_A>Score_B
]

要求：

[
\hat{Score}_A>\hat{Score}_B
]

学习：

* 技术复杂度；
* 完成质量；
* 难度差异。

---

# 13. Element Sequence Modeling

比赛不是单个元素，而是一系列：

[
Z=
[z_1,z_2,...,z_n]
]

因此建立：

## Element Transformer

输入：

```
Element Token Sequence


[z1]

[z2]

[z3]

[z4]
```

学习：

* Combo；
* 连续动作关系；
* 技术组合；
* 整体表现。

类似：

语言模型：

```
word1 → word2 → word3
```

悠悠球：

```
element1 → element2 → element3
```

---

# 14. Competition Score Prediction

最终：

单元素：

[
s_i=f(z_i)
]

比赛：

[
S=
\sum_i s_i
]

或者：

使用 Element Transformer：

[
S=f(z_1,z_2,...,z_n)
]

考虑：

* 元素组合；
* 重复动作；
* 整体流程。

---

# 15. 数据标注设计

不要求标注动作名称。

只需要：

```json
{
"video":"xxx",

"elements":[

{
"start_frame":120,
"end_frame":180,
"score":4
},

{
"start_frame":250,
"end_frame":320,
"score":6
}

]
}
```

核心监督：

```
元素时间范围
+
对应分值
```

---

# 16. Feature Flow

完整数据流：

```
Video Frame


↓

Perception Models


↓

Frame State


-----------------

String Fragment

YoYo State

Pose Context

Orientation


-----------------

↓

Frame Encoder


↓

Frame Embedding Sequence


↓

Boundary Detector


↓

Element Segment


↓

Element Encoder


↓

Element Token


↓

Element Transformer


↓

Score


```

---

# 17. 为什么采用模块化 Element Tokenizer

相比单一端到端模型：

```
Video

↓

自动Token

↓

Score
```

模块化方案具有：

## 可解释性

可以查看：

```
模型认为哪里产生元素
```

以及：

```
每个元素embedding
```

---

## 可替换性

Boundary 不好：

替换 Boundary Model。

Score 不好：

替换 Score Head。

不需要整体重新训练。

---

## 更符合数据特点

已有：

```
start/end
score
```

标注。

可以充分利用。

---

# 18. 最终系统结构

```
                         Competition Video

                                |

                                ↓

                    String / YoYo Perception

                                |

                                ↓

                    Frame Motion Encoder

                                |

                                ↓

                   Frame Embedding Sequence


                                |

              +-----------------+----------------+

              |                                  |

              ↓                                  ↓


       Boundary Detector                 Temporal Features


              |

              ↓


        Element Segments


              |

              ↓


        Element Encoder


              |

              ↓


        Element Tokens


              |

              ↓


     Element Transformer


              |

              ↓


          Score Prediction


              |

              ↓


       Competition Score
```

---

# 核心目标总结

最终模型学习的不是：

> “这个动作叫什么名字”。

而是：

[
\boxed{
连续绳线-悠悠球状态变化
\rightarrow
运动元素Token
\rightarrow
技术价值
}
]

其中：

* 绳线变化决定主要技术语义；
* 悠悠球轨迹补充动作结构；
* 姿态和方向提供执行条件；
* Element Token 表示抽象运动单元；
* Element Transformer 建模比赛整体表现。

最终形成一个不依赖固定 Trick 词典、能够适应自创动作和未知动作的悠悠球智能评分系统。
