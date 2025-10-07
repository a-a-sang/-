# Lecture 5: Deep Neural Networks

## 第 1 页：封面（Cover）



*   **标题**：媒体与认知（Media and Cognition）- 第 5 讲：深度神经网络（Lecture 5: Deep Neural Networks）

*   **出品方**：清华大学电子工程系（Dept. of EE, Tsinghua University）

*   **主讲人**：方璐（Lu FANG）

*   **配图**：VICE News|HBO 相关画面（无额外文本信息）

## 第 2 页：空白页（Blank Page）



*   无任何文本与图像内容，为过渡页。

## 第 3 页：人工智能的爆发与寒冬（Burst and Winter of AI）

### 核心时间线与阶段划分



| 时间区间      | 阶段（Stage）                | 关键事件（Key Events）                                                           |
| --------- | ------------------------ | -------------------------------------------------------------------------- |
| 1956      | 人工智能诞生（Birth of AI）      | 达特茅斯会议（Dartmouth Conference）首次提出 “人工智能” 概念，标志学科诞生。                         |
| 1956-1974 | 第一波爆发（The First Burst）   | 首个智能软件（逻辑理论家程序）、早期机器人（如工业机器人）问世，感知机（Perceptron）提出（1957）。                   |
| 1974-1980 | 第一次寒冬（The First Winter）  | 早期 AI 技术局限显现（如无法解决非线性问题），科研信心不足，资金削减（No confidence, no funding）。           |
| 1980-1987 | 第二波爆发（The Second Burst）  | 专家系统（Expert System）落地（如 XCON 用于设备配置），神经网络因反向传播（Back-Propagation）突破。        |
| 1987-1993 | 第二次寒冬（The Second Winter） | 专家系统维护成本高、适应性差，神经网络受算力限制，研究资金因缺乏经济效益（lack of practical economic effect）削减。 |
| 2006 - 至今 | 第三波爆发（The Third Burst）   | 大数据（Big Data）推动深度学习（Deep Learning）发展，语音与视觉识别（如人脸识别）能力突破。                   |

### 关键技术里程碑（按时间排序）



1.  **1957 年**：感知机（Perceptron）提出，为人工神经网络奠定基础；

2.  **1986 年**：反向传播算法（Back-Propagation）用于多层神经网络训练；

3.  **1995 年**：自适应学习系统（Adaptive Learning System）、归纳感知机（Inductive Perceptron）结合逻辑系统；

4.  **2006 年**：深度神经网络（Deep Neural Network）突破，具备深度学习能力；

5.  **2013 年**：语音与视觉识别（Voice and Visual Identification）技术成熟，进入实用阶段。

### 英文术语标注

达特茅斯会议（Dartmouth Conference）、感知机（Perceptron）、专家系统（Expert System）、反向传播（Back-Propagation）、深度学习（Deep Learning）、大数据（Big Data）

## 第 4 页：AlphaGo（2016）

### AlphaGo 核心成就与技术架构



*   **核心成就**：2016 年成为首个击败人类围棋冠军的 AI 程序，使用 176 块 GPU 与 1202 块 CPU，在《Nature》发表论文《Mastering the game of Go with deep neural networks and tree search》（Silver et al., 2016）。

*   **技术核心**：结合深度神经网络（Deep Neural Networks）与蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS），包含策略网络（Policy Network）与价值网络（Value Network）：

1.  **策略网络（Policy Network）**：预测落子位置概率，输入为围棋棋盘特征；

2.  **价值网络（Value Network）**：评估当前局面胜率，辅助 MCTS 剪枝，减少无效搜索。

### 硬件可扩展性测试（Hardware Scalability Test）



| AlphaGo 类型  | 搜索线程（Search Threads） | CPU 数量 | GPU 数量 | Elo 评分（Elo Rating） |
| ----------- | -------------------- | ------ | ------ | ------------------ |
| Distributed | 12                   | 428    | 64     | 2937               |
| Distributed | 24                   | 764    | 112    | 3079               |
| Distributed | 40                   | 1202   | 176    | 3140               |
| Distributed | 64                   | 1920   | 280    | 3168               |



*   **测试说明**：每步思考时间上限 2 秒，Elo 评分通过 BayesElo 计算，结果显示硬件资源（CPU/GPU/ 线程数）与棋力呈正相关。

### 网络结构示意图



*   含输入层（INPUT LAYER）、隐藏层（HIDDEN LAYERS）、输出层（OUTPUT LAYER），标注 “Polp (als)”“V (S)”（分别对应策略网络与价值网络输出）。

*   参考链接：[https://www.deepmind.com/research/highlighted-research/alphago](https://www.deepmind.com/research/highlighted-research/alphago)

### 英文术语标注

蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）、策略网络（Policy Network）、价值网络（Value Network）、Elo 评分（Elo Rating）、BayesElo（贝叶斯 Elo 评分方法）

## 第 5 页：超越 AlphaGo（Beyond AlphaGo）

### AlphaGo 系列模型演进（按时间排序）



| 模型（Model）    | 发布时间        | 覆盖领域（Domains）              | 核心创新（Core Innovation）                             | 知识依赖（Knowledge Dependency）                      | 参考文献（Reference）                                                                                                               |
| ------------ | ----------- | -------------------------- | ------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| AlphaGo      | 2016 年 1 月  | 围棋（Go）                     | 首次用神经网络 + 树搜索掌握围棋，击败人类冠军                          | 依赖人类棋谱（Human data/knowledge）+ 已知规则（Known rules） | Silver et al., Nature 2016: "Mastering the game of Go with deep neural networks and tree search"                              |
| AlphaGo Zero | 2017 年 10 月 | 围棋（Go）                     | 无监督自我对弈（Unsupervised Self-Play），零人类知识输入，仅通过规则自主学习 | 仅依赖已知规则（Known rules）                            | Silver et al., Nature 2017: "Mastering the game of go without human knowledge"                                                |
| AlphaZero    | 2018 年 12 月 | 围棋、国际象棋、将棋（Go/Chess/Shogi） | 单一算法（Single Algorithm）掌握 3 种完全信息博弈，无需任务特定优化       | 仅依赖各游戏已知规则（Known rules）                         | Silver et al., Science 2018: "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" |
| MuZero       | 2020 年 12 月 | 围棋、国际象棋、将棋、Atari 游戏        | 学习游戏规则（Learned Rules），无需已知动态环境信息，可掌握未知动力学环境任务     | 无规则输入（Learns rules from interaction）            | Schrittwieser et al., Nature 2020: "Mastering atari, go, chess and shogi by planning with a learned model"                    |

### 演进核心趋势



1.  **知识依赖减少**：从 “依赖人类数据” 到 “自主学习规则”，模型自主性大幅提升；

2.  **任务通用性增强**：从 “单一围棋任务” 到 “多领域通用”，打破任务壁垒；

3.  **环境适应性提升**：从 “已知规则环境” 到 “未知动力学环境”，泛化能力扩展。

### 英文术语标注

无监督自我对弈（Unsupervised Self-Play）、完全信息博弈（Perfect Information Games）、未知动力学环境（Unknown Dynamics）、自我对弈（Self-Play）

## 第 6 页：深度学习的核心驱动力（What makes deep learning work?）

### 三大核心要素



1.  **深度模型（Big Models with Many Layers）**

*   定义：含多层隐藏层的神经网络（如 BERT、ResNet），通过层级结构捕捉高阶非线性特征；

*   示例：BERT 模型含多层注意力机制，可捕捉文本上下文语义；ResNet 通过残差连接支持千层网络训练。

1.  **大规模数据集（Large Datasets with Various Examples）**

*   需求：为模型提供充足学习信号，覆盖多样场景（如 ImageNet 含 1400 万张图像，2 万 + 类别）；

*   作用：缓解过拟合（Overfitting），提升泛化能力（Generalization Ability），是深度学习爆发的基础。

1.  **高性能计算（High-Performance Computing to Support）**

*   硬件支撑：GPU（如 NVIDIA A100）、TPU（谷歌张量处理器）、分布式集群，加速矩阵运算（神经网络核心操作）；

*   软件支撑：深度学习框架（PyTorch、TensorFlow）、优化库（CUDA、cuDNN），降低大模型训练门槛。

### 配图说明



*   含模型结构示意图（输入层 - 隐藏层 - 输出层）、数据集示例（ImageNet 等）、GPU 硬件图，标注 “nature”“nature photonics” 等期刊名称，体现技术学术认可度。

### 英文术语标注

深度模型（Big Models）、大规模数据集（Large Datasets）、高性能计算（High-Performance Computing）、泛化能力（Generalization Ability）、GPU（Graphics Processing Unit）、TPU（Tensor Processing Unit）

## 第 7 页：人类大脑与深度学习模型对比（How about human brain?）

### 人类大脑核心特性



1.  **神经元规模**：约 860 亿个神经元（86,000,000,000 Neurons），远超当前深度学习模型参数规模（如 GPT-4 约 1.76 万亿参数，且 “参数”≠“神经元”）。

2.  **生理结构（Physiological Architecture）**

*   脑区划分：前额叶（FRONTAL）、顶叶（PARIETAL）、颞叶（TEMPORAL）、枕叶（OCCIPITAL），分别负责决策、感知、语言、视觉等功能；

*   模型类比：对应深度学习的 “输入层 - 隐藏层 - 输出层”，但结构复杂度远超人工模型。

1.  **数据与先验**

*   大规模 “数据集”：人类通过一生的视觉、听觉等感知数据学习，且包含先天先验（Innate Priors，如视觉系统天生的边缘检测能力）；

*   与 AI 差异：AI 需人工标注数据，缺乏人类的 “常识” 与 “小样本学习” 能力。

1.  **计算特性**

*   低功耗（Low-power Computing）：功耗仅 10\~23 瓦，远低于 AI 硬件（单 GPU 约 300 瓦）；

*   高效机制：通过局部激活（仅相关神经元工作）、动态连接（无用连接休眠）实现节能。

### 深度学习模型局限性



*   结构简化：仅模拟 “加权求和 + 激活”，缺乏神经递质、突触可塑性等生物细节；

*   数据依赖：需海量人工标注数据，无先天先验；

*   能耗高昂：大模型训练碳排放相当于 5 辆美国汽车终身排放量（MIT Technology Review 数据）。

### 英文术语标注

神经元（Neuron）、前额叶（Frontal Lobe）、顶叶（Parietal Lobe）、颞叶（Temporal Lobe）、枕叶（Occipital Lobe）、先天先验（Innate Priors）、低功耗计算（Low-power Computing）

## 第 8 页：人类大脑：智能与高效（Human Brain: Intelligent and Efficient）

### 大脑生理结构与功能



1.  **神经元连接机制**

*   连接方式：神经元通过轴突（Axons）与树突（Dendrites）相互连接，形成复杂网络；

*   信号传递：电信号通过神经递质（Neurotransmitters）在突触（Synapse）间传递，调控认知行为（如学习、记忆）。

1.  **突触与神经递质特性**

*   突触多样性：26 种突触类别（26 categories），差异达 8%（8% difference），支持不同信号传递需求；

*   神经递质作用：如多巴胺（Dopamine，奖励机制）、谷氨酸（Glutamate，兴奋性信号），实现精细化信号调控。

1.  **大脑高效性**

*   动态连接：通过 “动态链接（dynamic link）”“动态神经网络（dynamic neural network）” 调整连接强度，适应不同任务；

*   低功耗：功耗仅 10\~23 瓦，依赖 “按需激活” 机制，避免无效计算。

### 关键数据与图示



*   标注 “86\~100 billion neurons”（860\~1000 亿神经元），脑区结构示意图（标注 “Axon”“Dendrite”“Synapse”）。

### 英文术语标注

轴突（Axon）、树突（Dendrite）、突触（Synapse）、神经递质（Neurotransmitter）、多巴胺（Dopamine）、谷氨酸（Glutamate）、动态连接（Dynamic Connection）

## 第 9 页：大脑建模与人工神经网络（Modelling the Brain）

### 早期神经网络与大脑的关联



1.  **人工神经元的大脑模拟**

*   结构类比：


    *   树突（Dendrites）→人工神经元输入（Inputs）：接收上游信号；
    
    *   细胞体（Soma）→加权求和（Sum）：整合输入信号；
    
    *   轴突（Axon）→人工神经元输出（Output）：传递信号到下游；
    
    *   激活机制→非线性激活函数（Non-Linearity）：判断是否 “激活”（输出信号强度）。

1.  **人工模型的局限性**

*   简化本质：仅粗略模拟神经元部分功能，缺失生物细节（如神经递质、突触可塑性、细胞间复杂信号通路）；

*   核心差异：人工神经元无 “自主调节连接强度” 的生物机制，需通过反向传播人工更新权重。

### 配图说明



*   左侧为大脑神经元结构（标注 “Dendrites”“Soma”“Axon”“Synapse”），右侧为人工神经元结构（标注 “Inputs”“Weights”“Sum”“Non-Linearity”“Output”），直观对比两者对应关系。

### 英文术语标注

人工神经元（Artificial Neuron）、加权求和（Sum）、非线性激活函数（Non-Linearity）、突触可塑性（Synaptic Plasticity）

## 第 10 页：从大脑建模到神经网络演进（From Brain Modeling to Neural Network）

### 大脑科学与神经网络的交叉里程碑



| 时间     | 研究者（Researcher）  | 成果（Achievement）                                | 领域（Field）              | 意义（Significance）                     | 文献来源（Source）                                      |
| ------ | ---------------- | ---------------------------------------------- | ---------------------- | ------------------------------------ | ------------------------------------------------- |
| 1890 年 | W. James         | 神经元激活理论（Activation of Neurons）                 | 心理学（Psychology）        | 奠定 “神经元激活” 的理论基础，为人工神经网络提供思想源头。      | 《Principles of Psychology》                        |
| 1962 年 | D. Hubel         | 猫脑视觉感受野（Visual Receptive Field of Cat Brain）   | 神经科学（Neuroscience）     | 发现视觉皮层细胞对特定刺激（如边缘）敏感，启发 CNN 的局部连接设计。 | Journal of Neurophysiology                        |
| 1981 年 | C. Bruce         | 猴脑下颞皮质功能（Inferior Temporal Cortex of Monkey）   | 神经科学（Neuroscience）     | 揭示下颞皮质在物体识别中的作用，为视觉神经网络的层级结构提供生物依据。  | Journal of Physiology                             |
| 1989 年 | R. Albin         | 脑基底神经节研究（Basal Ganglia）                        | 神经科学（Neuroscience）     | 探索运动控制相关脑区，为机器人学的运动规划提供参考。           | Brain                                             |
| 2002 年 | D.P. Buxhoeveden | 脑皮层微柱结构（Cerebral Cortex Micropillar Structure） | 神经科学（Neuroscience）     | 发现皮层的微柱组织方式，启发神经网络的局部特征提取思路。         | Trends in Neurosciences                           |
| 2007 年 | T. Poggio        | HMAX 模型（HMAX Model）                            | 计算机视觉（Computer Vision） | 模拟视觉皮层的层级加工，是早期视觉神经网络的重要模型。          | Proceedings of the National Academy of Sciences   |
| 2017 年 | G. Hinton        | 胶囊网络（Capsule Network）                          | 深度学习（Deep Learning）    | 模拟大脑的 “部分 - 整体” 识别机制，提升小样本识别能力。      | Advances in Neural Information Processing Systems |

### 神经网络技术演进



1.  **1958 年（F. Rosenblatt）**：感知机（Perceptron）提出，首个可训练的人工神经网络；

2.  **1990 年（Y. LeCun）**：卷积神经网络（Convolutional Neural Network），模拟视觉感受野，用于手写数字识别；

3.  **2004 年（Y. Bengio）**：强化学习与神经网络结合（Reinforcement Learning with Neural Network），拓展神经网络应用场景；

4.  **2017 年（G. Hinton）**：胶囊网络（Capsule Network），进一步贴近大脑的物体识别机制。

### 英文术语标注

视觉感受野（Visual Receptive Field）、下颞皮质（Inferior Temporal Cortex）、脑基底神经节（Basal Ganglia）、脑皮层微柱（Cerebral Cortex Micropillar）、HMAX 模型（HMAX Model）、胶囊网络（Capsule Network）、感知机（Perceptron）、卷积神经网络（Convolutional Neural Network）

## 第 11 页：核心术语定义（Some Terms to Know）

### 四大核心神经网络术语



1.  **ANN（Artificial Neural Network，人工神经网络）**

*   定义：受生物神经网络启发的计算系统，是深度学习模型的统称（涵盖所有深度模型）；

*   特点：通过神经元（节点）与连接（权重）模拟信息传递，可拟合复杂函数。

1.  **DNN（Deep Neural Network，深度神经网络）**

*   定义：输入层与输出层之间含**多层隐藏层**的 ANN，是 “深度学习” 的核心载体；

*   与 ANN 关系：DNN 是 ANN 的子集，强调 “深度”（多层结构），区别于 “浅层神经网络”（如单隐层网络）。

1.  **CNN（Convolutional Neural Network，卷积神经网络）**

*   定义：含 “共享权重（Shared-weight）” 架构的 DNN，专为处理网格数据（如图像）设计；

*   核心操作：卷积层（Convolutional Layer）提取局部特征，池化层（Pooling Layer）降维，在计算机视觉领域应用最广泛。

1.  **RNN（Recurrent Neural Network，循环神经网络）**

*   定义：节点间存在循环连接（Cycle）的 DNN，可处理序列数据（如文本、语音）；

*   变体：LSTM（Long Short-Term Memory，长短期记忆网络）、GRU（Gated Recurrent Unit），解决 RNN 的梯度消失问题。

### 配图说明



*   含 DNN 结构示意图（输入层 - 多层隐藏层 - 输出层）、LSTM 细胞结构示意图，标注 “h4-1”“LSTM cell”，直观展示术语对应的模型结构。

### 英文术语标注

人工神经网络（Artificial Neural Network, ANN）、深度神经网络（Deep Neural Network, DNN）、卷积神经网络（Convolutional Neural Network, CNN）、循环神经网络（Recurrent Neural Network, RNN）、长短期记忆网络（Long Short-Term Memory, LSTM）、共享权重（Shared-weight）、序列数据（Sequential Data）

## 第 12 页：感知机与前向传播（The Perceptron, Forward Propagation）

### 感知机：深度学习的基础单元



1.  **结构组成**

*   输入（Inputs）：$x_1, x_2, ..., x_n$（如特征向量）；

*   权重（Weights）：$w_1, w_2, ..., w_n$（调节输入信号强度，通过训练更新）；

*   求和（Sum）：$\sum_{i=1}^n w_i x_i + b$（$b$为偏置，调整激活阈值）；

*   非线性激活（Non-Linearity）：激活函数$g(\cdot)$，引入非线性，使网络可拟合复杂函数；

*   输出（Output）：$y = g(\sum w_i x_i + b)$。

1.  **前向传播（Forward Propagation）**

*   定义：从输入层到输出层，按 “输入→权重→求和→激活→输出” 的顺序计算信号，是模型预测的核心流程；

*   示例：单感知机的前向传播即 “输入加权求和→激活→输出”，多层网络则需逐层计算隐藏层输出。

### 配图说明



*   感知机结构流程图（标注 “Inputs”“Weights”“Sum”“Non-Linearity”“Output”），箭头表示信号传递方向。

### 英文术语标注

感知机（Perceptron）、前向传播（Forward Propagation）、权重（Weights）、偏置（Bias）、非线性激活（Non-Linearity）、激活函数（Activation Function）

## 第 13 页：激活函数（Activation Functions）

### 常用激活函数公式与特性



| 激活函数（Activation Function）    | 公式（Formula）                                                                  | 特点（Characteristics）                                               | 适用场景（Application Scenario）       |
| ---------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------- | -------------------------------- |
| Sigmoid                      | $\sigma(x) = \frac{1}{1 + e^{-x}}$                                           | 输出范围 \[0,1]，可表示概率；但 x 绝对值大时导数接近 0（梯度消失），输出均值非 0。                  | 二分类输出层（如逻辑回归）、早期网络隐藏层（现已少用）      |
| Tanh（双曲正切）                   | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$                               | 输出范围 \[-1,1]，均值为 0，缓解梯度消失；但 x 绝对值大时仍梯度消失。                         | 早期 RNN 隐藏层（如 LSTM 的门控）、需零均值输出的场景 |
| ReLU（Rectified Linear Unit）  | $ReLU(x) = \max(0, x)$                                                       | 计算简单（仅判断 x 是否 > 0），x>0 时导数 = 1（缓解梯度消失）；但 x≤0 时导数 = 0（死亡 ReLU 问题）。 | CNN 隐藏层、DNN 隐藏层（当前主流）、需高效计算的场景   |
| ELU（Exponential Linear Unit） | $ELU(x) = \begin{cases} x & x \geq 0 \\ \alpha(e^x - 1) & x < 0 \end{cases}$ | 解决死亡 ReLU（x≤0 时仍有梯度），输出均值接近 0；但计算复杂度高于 ReLU（需指数运算）。               | 对梯度敏感的网络（如小样本训练、高精度要求场景）         |

### 激活函数的核心作用



*   **无激活函数（w/o activation）**：无论网络层数多少，输出始终是输入的线性组合（线性模型），无法解决非线性问题；

*   **有激活函数（w/activation）**：引入非线性，使深层网络可近似任意复杂函数（Universal Approximation Theorem，通用近似定理），是深度学习解决复杂任务的关键。

### 配图说明



*   含各激活函数的函数图像（x 轴为输入，y 轴为输出），标注 “101”（可能为坐标刻度），直观展示函数形态差异。

### 英文术语标注

Sigmoid 函数（Sigmoid Function）、Tanh 函数（Hyperbolic Tangent Function）、ReLU（Rectified Linear Unit）、ELU（Exponential Linear Unit）、梯度消失（Vanishing Gradient）、通用近似定理（Universal Approximation Theorem）

## 第 14 页：构建单隐层神经网络（Building Neural Networks with Perceptrons - Single Layer）

### 单隐层神经网络结构



1.  **组成部分**

*   输入层（Inputs）：$x_1, x_2, ..., x_d$（d 为输入维度，如特征数）；

*   隐藏层（Hidden Layer）：含多个感知机（如 3 个），每个感知机的输出为$a_1 = g(w_{11}x_1 + w_{12}x_2 + ... + w_{1d}x_d + b_1)$，$a_2, a_3$类似；

*   输出层（Outputs）：对隐藏层输出加权求和并激活，如$y = g(w_{o1}a_1 + w_{o2}a_2 + w_{o3}a_3 + b_o)$；

*   激活函数（g (・)）：隐藏层与输出层可使用不同激活函数（如隐藏层用 ReLU，输出层用 Sigmoid）。

1.  **前向传播流程**


    1.  输入层→隐藏层：计算每个隐藏单元的线性变换$z_1 = w_1^T x + b_1$，激活后得到$a_1 = g(z_1)$；
    
    2.  隐藏层→输出层：计算输出单元的线性变换$z_o = w_o^T a + b_o$，激活后得到最终输出$y = g(z_o)$。

### 配图说明



*   网络结构示意图：输入层（3 个节点）→隐藏层（3 个节点，标注 “g (・)”）→输出层（1 个节点，标注 “g (・)”），箭头标注权重传递方向，标注 “Hidden states”（隐藏层状态）。

### 英文术语标注

单隐层神经网络（Single Layer Neural Network）、输入层（Input Layer）、隐藏层（Hidden Layer）、输出层（Output Layer）、隐藏层状态（Hidden States）、前向传播（Forward Propagation）

## 第 15 页：单隐层神经网络细节（Building Neural Networks with Perceptrons - Single Layer Details）

### 结构重复与补充说明



*   与第 14 页结构一致，进一步强调 “隐藏层状态（Hidden states）” 的作用：隐藏层是 “输入特征的非线性变换”，将原始输入映射到更高维特征空间，为输出层提供更具区分性的特征。

### 关键细节补充



*   **权重矩阵表示**：输入层到隐藏层的权重矩阵$W^{(1)} \in \mathbb{R}^{h \times d}$（h 为隐藏层节点数，d 为输入维度），隐藏层到输出层的权重矩阵$W^{(2)} \in \mathbb{R}^{1 \times h}$；

*   **偏置向量**：隐藏层偏置$b^{(1)} \in \mathbb{R}^h$，输出层偏置$b^{(2)} \in \mathbb{R}^1$，确保网络可平移激活阈值，提升拟合能力。

### 配图说明



*   与第 14 页示意图一致，无新增图像，仅文本补充权重矩阵与偏置的数学表示，强化数学建模视角。

### 英文术语标注

权重矩阵（Weight Matrix）、偏置向量（Bias Vector）、特征空间（Feature Space）、非线性变换（Non-Linear Transformation）

## 第 16 页：构建深度神经网络（Building Neural Networks with Perceptrons - Deep Layer）

### 深度神经网络结构（含多层隐藏层）



1.  **组成部分**

*   输入层（Inputs）：$x_1, x_2, ..., x_d$（原始特征）；

*   隐藏层（Hidden Layers）：至少 2 层（如 2 层），每层含多个感知机，逐层对特征进行非线性变换；


    *   隐藏层 1 输出：$a^{(1)} = g^{(1)}(W^{(1)}x + b^{(1)})$；
    
    *   隐藏层 2 输出：$a^{(2)} = g^{(2)}(W^{(2)}a^{(1)} + b^{(2)})$；

*   输出层（Outputs）：$y = g^{(L)}(W^{(L)}a^{(L-1)} + b^{(L)})$（L 为总层数）。

1.  **“深度” 的意义**

*   层级特征提取：浅层隐藏层提取低维特征（如图像的边缘、纹理），深层隐藏层提取高维语义特征（如图像的物体部件、类别）；

*   效率优势：深度结构比浅层结构用更少参数拟合复杂函数（如 10 层网络比 1 层网络用更少节点拟合相同函数）。

### 配图说明



*   网络结构示意图：输入层（3 个节点）→隐藏层 1（3 个节点，标注 “g (・)”）→隐藏层 2（3 个节点，标注 “g (・)”）→输出层（1 个节点，标注 “g (・)”），标注 “Hidden states”（各隐藏层状态），箭头表示信号传递方向。

### 英文术语标注

深度神经网络（Deep Neural Network, DNN）、多层隐藏层（Multiple Hidden Layers）、层级特征提取（Hierarchical Feature Extraction）、语义特征（Semantic Feature）

## 第 17 页：深度神经网络的优化方法（How to optimize Deep Neural Network?）

### 核心优化需求



*   问题：深度网络参数规模大（如百万级），暴力计算梯度复杂度极高，需高效优化方法；

*   解决方案：**误差反向传播（Backpropagation of Error, BP）+ 梯度下降（Gradient Descent）**，通过链式法则高效计算梯度，降低复杂度。

### 反向传播与梯度下降的核心逻辑



1.  **前向传播（Forward Pass）**

*   计算流程：$x \to z^{(1)} = W^{(1)}x + b^{(1)} \to a^{(1)} = ReLU(z^{(1)}) \to z^{(2)} = W^{(2)}a^{(1)} + b^{(2)} \to L$（损失）；

*   目的：得到模型预测值与损失$L$，为反向传播提供 “误差来源”。

1.  **反向传播（Backward Pass）**

*   核心思想：从输出层到输入层，利用链式法则（Chain Rule）计算损失对各层参数（$W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}$）的梯度；

*   优势：


    *   降低计算复杂度：避免暴力计算的$O(n^3)$复杂度，降至$O(n^2)$；
    
    *   复用中间结果：前向传播的$z^{(1)}, a^{(1)}, z^{(2)}$可直接用于反向传播，减少重复计算。

1.  **梯度下降（Gradient Descent）**

*   参数更新：$W = W - \alpha \cdot \frac{\partial L}{\partial W}$，$b = b - \alpha \cdot \frac{\partial L}{\partial b}$（$\alpha$为学习率）；

*   目标：沿梯度负方向更新参数，最小化损失$L$。

### 配图说明



*   前向传播与反向传播流程图：标注 “forward pass”“backward pass” 方向，用箭头连接各计算步骤，标注梯度计算路径（如$\frac{\partial L}{\partial a^{(1)}} \to \frac{\partial L}{\partial z^{(1)}} \to \frac{\partial L}{\partial W^{(1)}}$）。

### 英文术语标注

误差反向传播（Backpropagation of Error, BP）、梯度下降（Gradient Descent）、链式法则（Chain Rule）、前向传播（Forward Pass）、反向传播（Backward Pass）、学习率（Learning Rate）

## 第 18 页：神经网络训练目标与损失函数（Training Neural Networks - Purpose & Loss）

### 训练目标（Purpose of Training）



*   核心：找到最优网络权重（Network Weights），使**损失函数（Loss Function）** 最小化 —— 损失函数衡量 “模型预测值” 与 “真实标签（Ground Truth）” 的差异。

### 损失函数（Loss Formulation）



*   定义：损失函数是网络权重的函数，记为$L(W, b; x^{(i)}, y^{(i)})$，其中：


    *   $W, b$：网络所有层的权重与偏置（待优化变量）；
    
    *   $x^{(i)}, y^{(i)}$：第 i 个训练样本的输入与真实标签。

*   常用损失函数类型：

1.  **均方误差（Mean Squared Error, MSE）**：适用于回归任务，$L = \frac{1}{2}\|y - \hat{y}\|^2$（$\hat{y}$为预测值）；

2.  **交叉熵损失（Cross-Entropy Loss）**：适用于分类任务，$L = -\sum_{c=1}^C y_c \log \hat{y}_c$（$C$为类别数，$y_c$为真实标签的 one-hot 编码）。

### 配图说明



*   无新增图像，仅文本强调 “网络权重（network weights）” 是训练的核心变量，损失函数是优化的 “目标函数”，为后续梯度计算铺垫。

### 英文术语标注

损失函数（Loss Function）、网络权重（Network Weights）、真实标签（Ground Truth）、均方误差（Mean Squared Error, MSE）、交叉熵损失（Cross-Entropy Loss）、回归任务（Regression Task）、分类任务（Classification Task）

## 第 19 页：梯度下降训练流程（Training Neural Networks - Gradient Descent Flow）

### 梯度下降的三步骤流程



1.  **步骤 1：随机初始化权重（Initialize Weights Randomly）**

*   原因：避免权重对称（如全零初始化会导致所有神经元输出相同，梯度相同，无法更新）；

*   方法：从均匀分布或正态分布中采样（如$W \sim \mathcal{U}(-\frac{1}{\sqrt{d}}, \frac{1}{\sqrt{d}})$，$d$为输入维度）。

1.  **步骤 2：迭代优化（Loop until Convergence）**

    a. **计算梯度（Compute Gradient）**：通过反向传播计算损失对所有参数的梯度$\frac{\partial L}{\partial W}, \frac{\partial L}{\partial b}$；

    b. **更新权重（Update Weights）**：沿梯度负方向更新，$W = W - \alpha \cdot \frac{\partial L}{\partial W}$，$b = b - \alpha \cdot \frac{\partial L}{\partial b}$；

    c. **判断收敛（Check Convergence）**：若损失变化小于阈值（如$1e-6$）或达到最大迭代次数，停止迭代。

2.  **步骤 3：返回最优权重（Return Weights）**

*   输出收敛时的权重$W^*, b^*$，用于后续预测。

### 损失函数可视化



*   配图为 “损失值 - 权重值” 曲线，标注 “Loss Value”（纵轴）、“Value of weight”（横轴）、“Starting point”（初始点）、“Point of convergence”（收敛点），直观展示梯度下降 “从初始点沿曲线下降至最小值” 的过程。

### 英文术语标注

随机初始化（Random Initialization）、收敛（Convergence）、梯度计算（Gradient Computation）、权重更新（Weight Update）、均匀分布（Uniform Distribution）、正态分布（Normal Distribution）

## 第 20 页：神经网络的链式法则（Training Neural Networks - Chain Rule）

### 链式法则的数学基础



1.  **单变量链式法则**

*   若$x \stackrel{g}{\to} y \stackrel{f}{\to} z$（x→y 由 g 函数，y→z 由 f 函数），则$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$；

*   意义：复合函数的导数等于各环节导数的乘积，为多变量链式法则提供基础。

1.  **多变量链式法则（高维链式法则）**

*   若$x$为 d 维向量，$y = g(x)$为 m 维向量，$z = f(y)$为标量，则：

$ 
     \frac{\partial z}{\partial x_i} = \sum_{j=1}^m \frac{\partial y_j}{\partial x_i} \cdot \frac{\partial z}{\partial y_j} = \left( \frac{\partial y}{\partial x_i} \right)^T \cdot \frac{\partial z}{\partial y}
      $



*   矩阵表示：$\frac{\partial z}{\partial x} = \left( \frac{\partial y}{\partial x} \right)^T \cdot \frac{\partial z}{\partial y}$，其中$\frac{\partial y}{\partial x}$为 g 的雅可比矩阵（Jacobian Matrix），$\frac{\partial z}{\partial y}$为 f 的梯度向量。

### 神经网络中的链式法则应用



*   核心：神经网络是 “函数复合” 的产物（输入→隐藏层 1→隐藏层 2→输出），需用多变量链式法则计算损失对各层参数的梯度，是反向传播的数学核心。

### 配图说明



*   含函数复合示意图（x→g→y→f→z），标注 “Jacobian of g”（g 的雅可比矩阵）、“Jacobian of f”（f 的雅可比矩阵），直观展示链式法则的 “导数传递” 过程。

### 英文术语标注

链式法则（Chain Rule）、单变量链式法则（Single-Variable Chain Rule）、多变量链式法则（Multivariable Chain Rule）、雅可比矩阵（Jacobian Matrix）、梯度向量（Gradient Vector）、函数复合（Function Composition）

## 第 21 页：神经网络的链式法则实例（Training Neural Networks - Chain Rule Example）

### 两层神经网络的链式法则应用



*   **网络结构**：输入层→线性层 1（linear layer 1）→sigmoid 激活→线性层 2（linear layer 2）→softmax→交叉熵损失（cross-entropy loss）；

*   参数定义：


    *   线性层 1：权重$W^{(1)}$，偏置$b^{(1)}$，输出$z^{(1)} = W^{(1)}x + b^{(1)}$，激活后$a^{(1)} = \sigma(z^{(1)})$（$\sigma$为 sigmoid）；
    
    *   线性层 2：权重$W^{(2)}$，偏置$b^{(2)}$，输出$z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}$，softmax 后$\hat{y} = \text{softmax}(z^{(2)})$；
    
    *   损失：$\mathcal{L} = -\sum y_c \log \hat{y}_c$（y 为真实标签）。

### 梯度计算实例



1.  **损失对**** ****的梯度**

$ 
   \frac{\partial \mathcal{L}}{\partial W^{(2)}} = \frac{\partial z^{(2)}}{\partial W^{(2)}} \cdot \frac{\partial \mathcal{L}}{\partial z^{(2)}}
    $



*   推导：$\frac{\partial z^{(2)}}{\partial W^{(2)}} = a^{(1)T}$（因$z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}$，对$W^{(2)}$求导为$a^{(1)T}$）；

*   $\frac{\partial \mathcal{L}}{\partial z^{(2)}} = \hat{y} - y$（softmax + 交叉熵的梯度性质，直接结果）。

1.  **损失对**** ****的梯度**

$ 
   \frac{\partial \mathcal{L}}{\partial W^{(1)}} = \frac{\partial z^{(1)}}{\partial W^{(1)}} \cdot \frac{\partial a^{(1)}}{\partial z^{(1)}} \cdot \frac{\partial z^{(2)}}{\partial a^{(1)}} \cdot \frac{\partial \mathcal{L}}{\partial z^{(2)}}
    $



*   推导：


    *   $\frac{\partial z^{(1)}}{\partial W^{(1)}} = x^T$（线性层对权重的导数）；
    
    *   $\frac{\partial a^{(1)}}{\partial z^{(1)}} = \sigma(z^{(1)}) \cdot (1 - \sigma(z^{(1)}))$（sigmoid 的导数）；
    
    *   $\frac{\partial z^{(2)}}{\partial a^{(1)}} = W^{(2)T}$（线性层对输入的导数）；
    
    *   最终合并各环节导数，得到$\frac{\partial \mathcal{L}}{\partial W^{(1)}}$。

### 配图说明



*   网络结构与梯度传递示意图：标注各层参数、激活函数、损失，箭头标注梯度传递方向（如$\frac{\partial \mathcal{L}}{\partial z^{(2)}} \to \frac{\partial \mathcal{L}}{\partial a^{(1)}} \to \frac{\partial \mathcal{L}}{\partial z^{(1)}} \to \frac{\partial \mathcal{L}}{\partial W^{(1)}}$）。

### 英文术语标注

线性层（Linear Layer）、sigmoid 激活函数（Sigmoid Activation Function）、softmax 函数（Softmax Function）、交叉熵损失（Cross-Entropy Loss）、梯度传递（Gradient Propagation）

## 第 22 页：反向传播计算梯度（Training Neural Networks - Compute Gradients with BP）

### 反向传播的核心步骤



*   延续第 21 页的两层网络结构，进一步强调 “反向传播的顺序”：从输出层开始，逐层向输入层计算梯度，复用前向传播的中间结果（如$a^{(1)}, z^{(1)}, z^{(2)}$）。

### 关键梯度复用逻辑



*   计算$\frac{\partial \mathcal{L}}{\partial W^{(1)}}$时，需用到$\frac{\partial \mathcal{L}}{\partial z^{(2)}}$（已在计算$\frac{\partial \mathcal{L}}{\partial W^{(2)}}$时得到），无需重新计算；

*   优势：减少重复计算，将梯度计算复杂度从$O(n^3)$降至$O(n^2)$，使深层网络训练成为可能。

### 配图说明



*   简化的梯度传递流程图：仅标注核心梯度环节（$\frac{\partial \mathcal{L}}{\partial z^{(2)}} \to \frac{\partial \mathcal{L}}{\partial a^{(1)}} \to \frac{\partial \mathcal{L}}{\partial z^{(1)}} \to \frac{\partial \mathcal{L}}{\partial W^{(1)}}$），突出 “复用” 逻辑。

### 英文术语标注

反向传播（Backpropagation, BP）、梯度复用（Gradient Reuse）、中间结果（Intermediate Results）、梯度计算复杂度（Gradient Computation Complexity）

## 第 23 页：反向传播计算$W^{(2)}$梯度（Training Neural Networks - Compute $\frac{\partial \mathcal{L}}{\partial W^{(2)}}$）

### 详细推导过程



1.  **步骤 1：明确**** ****与**** ****的关系**

*   $z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}$，其中$W^{(2)}$为线性层 2 的权重矩阵（$C \times h$，C 为类别数，h 为隐藏层节点数），$a^{(1)}$为隐藏层 1 的输出（$h \times 1$）。

1.  **步骤 2：计算**** **

*   对$W^{(2)}$的第 c 行第 j 列元素$W^{(2)}_{c,j}$求导：$\frac{\partial z^{(2)}_c}{\partial W^{(2)}_{c,j}} = a^{(1)}_j$（因$z^{(2)}_c = \sum_j W^{(2)}_{c,j}a^{(1)}_j + b^{(2)}_c$）；

*   矩阵形式：$\frac{\partial z^{(2)}}{\partial W^{(2)}} = a^{(1)T}$（$h \times 1$的$a^{(1)}$转置为$1 \times h$，与$W^{(2)}$的维度匹配）。

1.  **步骤 3：计算**** **

*   因$\mathcal{L} = -\sum_c y_c \log \text{softmax}(z^{(2)}_c)$，利用 softmax 的梯度性质：$\frac{\partial \mathcal{L}}{\partial z^{(2)}_c} = \text{softmax}(z^{(2)}_c) - y_c = \hat{y}_c - y_c$；

*   向量形式：$\frac{\partial \mathcal{L}}{\partial z^{(2)}} = \hat{y} - y$（$C \times 1$）。

1.  **步骤 4：合并得到**** **

$ 
   \frac{\partial \mathcal{L}}{\partial W^{(2)}} = \frac{\partial z^{(2)}}{\partial W^{(2)}} \cdot \frac{\partial \mathcal{L}}{\partial z^{(2)}} = (\hat{y} - y) \cdot a^{(1)T}
    $



*   维度匹配：$(C \times 1) \cdot (1 \times h) = C \times h$，与$W^{(2)}$的维度（$C \times h$）一致。

### 配图说明



*   矩阵维度标注图：标注$W^{(2)}$（$C \times h$）、$a^{(1)}$（$h \times 1$）、$\hat{y} - y$（$C \times 1$）的维度，直观展示梯度计算的维度兼容性。

### 英文术语标注

权重矩阵维度（Weight Matrix Dimension）、softmax 梯度性质（Gradient Property of Softmax）、矩阵乘法（Matrix Multiplication）、维度匹配（Dimension Matching）

## 第 24 页：反向传播计算梯度实例（Training Neural Networks - BP Gradient Example）

### 简化实例：单输出网络



*   **网络结构**：输入 x（2 维）→线性层 1（W1：3×2，b1：3×1）→ReLU→线性层 2（W2：1×3，b2：1×1）→输出 z2→损失 L（MSE）。

### 梯度计算步骤



1.  **前向传播**

*   $z1 = W1x + b1$ → $a1 = ReLU(z1)$ → $z2 = W2a1 + b2$ → $L = \frac{1}{2}(z2 - y)^2$（y 为真实标签）。

1.  **反向传播**

    a. 计算$\frac{\partial L}{\partial z2} = z2 - y$（MSE 的导数）；

    b. 计算$\frac{\partial L}{\partial W2} = \frac{\partial z2}{\partial W2} \cdot \frac{\partial L}{\partial z2} = a1^T \cdot (z2 - y)$；

    c. 计算$\frac{\partial L}{\partial a1} = W2^T \cdot \frac{\partial L}{\partial z2}$；

    d. 计算$\frac{\partial L}{\partial z1} = \frac{\partial a1}{\partial z1} \cdot \frac{\partial L}{\partial a1}$（ReLU 导数：z1>0 时为 1，否则为 0）；

    e. 计算$\frac{\partial L}{\partial W1} = x^T \cdot \frac{\partial L}{\partial z1}$。

### 核心结论



*   反向传播的 “反向” 体现在 “从输出损失出发，逐层回溯到输入层参数”，每一步均依赖前一步的梯度结果，实现高效计算。

### 配图说明



*   简化网络的梯度传递图：标注各层参数、激活函数、损失，用数字 1-5 标注反向传播步骤，清晰展示梯度计算顺序。

### 英文术语标注

单输出网络（Single-Output Network）、ReLU 导数（ReLU Derivative）、MSE 导数（MSE Derivative）、梯度回溯（Gradient Backtracking）

## 第 25 页：反向传播的通用公式（Training Neural Networks - General BP Formula）

### 通用网络结构定义



*   设网络有 L 层，第 l 层：


    *   线性变换：$z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$；
    
    *   激活输出：$a^{(l)} = g^{(l)}(z^{(l)})$（$g^{(l)}$为第 l 层激活函数）；
    
    *   输入层：$a^{(0)} = x$；
    
    *   损失：$\mathcal{L}$（如交叉熵、MSE）。

### 反向传播通用梯度公式



1.  **损失对**** ****的梯度（误差项**** ****）**

*   输出层（l=L）：$\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial z^{(L)}}$（直接根据损失类型计算，如 softmax + 交叉熵为$\hat{y} - y$）；

*   隐藏层（l < L）：$\delta^{(l)} = \left( W^{(l+1)T} \cdot \delta^{(l+1)} \right) \odot g^{(l)\prime}(z^{(l)})$；


    *   $\odot$为哈达玛积（Element-wise Product），对应激活函数导数与梯度的逐元素相乘。

1.  **损失对**** ****的梯度**

$ 
   \frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \cdot \left( a^{(l-1)} \right)^T
    $



1.  **损失对**** ****的梯度**

$ 
   \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}
    $



*   原因：$z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$，对$b^{(l)}$求导为 1，故梯度等于误差项$\delta^{(l)}$。

### 关键复用逻辑



*   计算$\delta^{(l)}$时，需用到$\delta^{(l+1)}$（已在计算 l+1 层时得到），避免重复计算，这是反向传播高效性的核心。

### 配图说明



*   通用网络的梯度传递示意图：标注第 l 层、l+1 层的参数、激活函数、误差项$\delta^{(l)}, \delta^{(l+1)}$，箭头标注$\delta^{(l+1)} \to \delta^{(l)}$的传递方向。

### 英文术语标注

误差项（Error Term, $\delta^{(l)}$）、哈达玛积（Element-wise Product, $\odot$）、通用梯度公式（General Gradient Formula）、层索引（Layer Index, l）

## 第 26 页：反向传播的计算复杂度（Training Neural Networks - BP Computational Complexity）

### 梯度计算的复杂度分析



1.  **暴力计算的复杂度**

*   若网络每层有 n 个节点，计算损失对某层权重的梯度需矩阵乘法$n \times n$，暴力计算所有层梯度的复杂度为$O(n^3)$；

*   问题：当 n=4096（如 AlexNet 的全连接层）时，$n^3 = 6.8 \times 10^{10}$，计算量极大，无法实用。

1.  **反向传播的复杂度优化**

*   核心：利用 “损失是标量” 的特性，将梯度计算拆解为 “误差项传递 + 局部梯度计算”，复杂度降至$O(n^2)$；

*   实例：n=4096 时，$n^2 = 1.6 \times 10^7$，计算量仅为暴力计算的 0.02%，大幅降低训练门槛。

### 关键观察：损失的标量特性



*   损失$\mathcal{L}$是标量（单个数值），因此$\frac{\partial \mathcal{L}}{\partial z^{(l)}}$是向量（$n \times 1$），而非矩阵；

*   优势：误差项$\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial z^{(l)}}$为向量，与权重矩阵的乘法为向量 - 矩阵乘法，复杂度为$O(n^2)$，而非矩阵 - 矩阵乘法的$O(n^3)$。

### 引用示例：AlexNet 的全连接层



*   AlexNet 含 4096 维的全连接层（Krizhevsky et al., 2017），若用暴力计算梯度，需$O(4096^3)$操作，无法实时训练；

*   反向传播将复杂度降至$O(4096^2)$，使 AlexNet 在 2012 年 ImageNet 竞赛中夺冠，证明反向传播的实用性。

### 配图说明



*   复杂度对比图：标注 “暴力计算（Brute Force）” 与 “反向传播（Backpropagation）” 的复杂度曲线（横轴为节点数 n，纵轴为计算量），直观展示$O(n^3)$与$O(n^2)$的差异。

### 英文术语标注

计算复杂度（Computational Complexity）、暴力计算（Brute Force Computation）、标量特性（Scalar Property）、向量 - 矩阵乘法（Vector-Matrix Multiplication）、矩阵 - 矩阵乘法（Matrix-Matrix Multiplication）

## 第 27 页：反向传播的高效计算技巧（Training Neural Networks - Efficient BP Computation）

### 核心技巧：先计算误差项（$\delta$）



1.  **步骤 1：从输出层计算**** **

*   输出层误差项$\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial z^{(L)}}$（如 softmax + 交叉熵为$\hat{y} - y$），直接可得，无需复杂计算。

1.  **步骤 2：反向传递**** ****至隐藏层**

*   隐藏层 l 的误差项$\delta^{(l)} = W^{(l+1)T} \cdot \delta^{(l+1)} \odot g^{(l)\prime}(z^{(l)})$；

*   计算成本：向量乘法（$n \times n$）+ 逐元素乘法（$n$），复杂度$O(n^2)$，成本低。

1.  **步骤 3：计算权重梯度**

*   权重梯度$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \cdot (a^{(l-1)})^T$；

*   计算成本：向量 - 矩阵乘法（$n \times n$），复杂度$O(n^2)$，复用已计算的$\delta^{(l)}$与前向传播的$a^{(l-1)}$。

### 复杂度对比验证



*   假设每层节点数 n=1000：


    *   暴力计算：$O(n^3) = 10^9$操作；
    
    *   反向传播：$O(n^2) = 10^6$操作，计算量仅为暴力计算的 0.1%，效率提升显著。

### 配图说明



*   反向传播步骤拆解图：标注 “步骤 1：计算$\delta^{(L)}$”“步骤 2：传递$\delta$至隐藏层”“步骤 3：计算权重梯度”，每个步骤标注计算复杂度，直观展示高效性。

### 英文术语标注

误差项传递（Error Term Propagation）、向量乘法（Vector Multiplication）、逐元素乘法（Element-wise Multiplication）、计算成本（Computational Cost）

## 第 28 页：反向传播的权重复用（Training Neural Networks - Weight Reuse in BP）

### 权重在梯度计算中的复用逻辑



1.  **权重矩阵**** ****的复用**

*   在计算隐藏层 l 的误差项$\delta^{(l)} = W^{(l+1)T} \cdot \delta^{(l+1)} \odot g^{(l)\prime}(z^{(l)})$时，需用到$W^{(l+1)T}$（l+1 层的权重矩阵转置）；

*   优势：$W^{(l+1)}$是网络的固有参数，无需额外存储或计算，直接复用前向传播的权重，减少内存占用。

1.  **前向传播中间结果的复用**

*   激活输出$a^{(l-1)}$在计算$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \cdot (a^{(l-1)})^T$时复用，无需重新计算前向传播；

*   注意：需在正向传播时存储$a^{(l-1)}, z^{(l)}$等中间结果，内存占用略有增加，但远小于暴力计算的成本。

### 单个构建块的梯度计算



*   定义 “构建块”：含权重$W$、偏置$b$、激活函数$g$的单层结构，输入为$a_{in}$，输出为$a_{out} = g(Wa_{in} + b)$；

*   梯度计算：


    *   对输入$a_{in}$的梯度：$\frac{\partial \mathcal{L}}{\partial a_{in}} = W^T \cdot \left( \frac{\partial \mathcal{L}}{\partial a_{out}} \odot g'(Wa_{in} + b) \right)$；
    
    *   对权重$W$的梯度：$\frac{\partial \mathcal{L}}{\partial W} = \left( \frac{\partial \mathcal{L}}{\partial a_{out}} \odot g'(Wa_{in} + b) \right) \cdot a_{in}^T$；

*   核心：复用权重$W$与激活函数导数，避免重复计算。

### 配图说明



*   构建块示意图：标注 “输入$a_{in}$”“权重$W$”“偏置$b$”“激活函数$g$”“输出$a_{out}$”，箭头标注梯度传递方向，突出 “权重$W$复用”。

### 英文术语标注

权重复用（Weight Reuse）、中间结果复用（Intermediate Result Reuse）、构建块（Building Block）、内存占用（Memory Usage）

## 第 29 页：反向传播的梯度复用（Training Neural Networks - Gradient Reuse in BP）

### 梯度复用的具体场景



1.  **同一权重在不同层的梯度复用**

*   例如，计算$\frac{\partial \mathcal{L}}{\partial a^{(l)}}$（损失对 l 层激活输出的梯度）时，结果可直接用于计算$\frac{\partial \mathcal{L}}{\partial z^{(l)}} = \frac{\partial \mathcal{L}}{\partial a^{(l)}} \odot g^{(l)\prime}(z^{(l)})$，无需重新计算$\frac{\partial \mathcal{L}}{\partial a^{(l)}}$；

*   优势：减少梯度的重复计算，降低计算成本。

1.  **不同权重的梯度共享中间结果**

*   计算$\frac{\partial \mathcal{L}}{\partial W^{(l)}}$与$\frac{\partial \mathcal{L}}{\partial b^{(l)}}$时，均需用到误差项$\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial z^{(l)}}$，复用$\delta^{(l)}$避免重复计算激活函数导数与权重传递；

*   实例：$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \cdot (a^{(l-1)})^T$，$\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}$，共享$\delta^{(l)}$。

### 关键公式验证



*   若需计算$\frac{\partial \mathcal{L}}{\partial W^{(l)}}$与$\frac{\partial \mathcal{L}}{\partial W^{(l-1)}}$：


    *   计算$\frac{\partial \mathcal{L}}{\partial W^{(l)}}$得到$\delta^{(l)}$；
    
    *   计算$\frac{\partial \mathcal{L}}{\partial W^{(l-1)}}$时，复用$\delta^{(l)}$得到$\delta^{(l-1)} = W^{(l)T} \cdot \delta^{(l)} \odot g^{(l-1)\prime}(z^{(l-1)})$，无需重新计算$\delta^{(l)}$。

### 配图说明



*   梯度复用流程图：标注 “计算$\delta^{(l)}$”“复用$\delta^{(l)}$计算$\delta^{(l-1)}$”“复用$\delta^{(l)}$计算$\frac{\partial \mathcal{L}}{\partial W^{(l)}}, \frac{\partial \mathcal{L}}{\partial b^{(l)}}$”，用虚线框标注共享的$\delta^{(l)}$。

### 英文术语标注

梯度复用（Gradient Reuse）、中间结果共享（Intermediate Result Sharing）、误差项共享（Error Term Sharing）、计算成本降低（Computational Cost Reduction）

## 第 30 页：反向传播算法流程（Training Neural Networks - Backpropagation Algorithm）

### 完整算法步骤

#### 1. 前向传播（Forward Pass）



*   输入：训练样本$x$，网络参数$W^{(1)}, b^{(1)}, ..., W^{(n)}, b^{(n)}$；

*   计算各层线性变换与激活输出：

$ 
  z^{(i)} = W^{(i)}a^{(i-1)} + b^{(i)}, \quad a^{(i)} = g^{(i)}(z^{(i)}) \quad (i=1, 2, ..., n)
   $



*   初始输入：$a^{(0)} = x$；

*   输出层激活：若为分类任务，$a^{(n)} = \text{softmax}(z^{(n)})$。

#### 2. 计算输出层误差项（Initialize $\delta$）



*   输出层误差项$\delta^{(n)} = \frac{\partial \mathcal{L}}{\partial z^{(n)}}$，根据损失类型计算：


    *   交叉熵 + softmax：$\delta^{(n)} = a^{(n)} - y$（y 为真实标签）；
    
    *   MSE + 线性激活：$\delta^{(n)} = a^{(n)} - y$；
    
    *   MSE+sigmoid：$\delta^{(n)} = (a^{(n)} - y) \cdot a^{(n)} \cdot (1 - a^{(n)})$。

#### 3. 反向传播误差项（Backward Pass for $\delta$）



*   从输出层到输入层（i 从 n-1 降至 1）：

$ 
  \delta^{(i)} = W^{(i+1)T} \cdot \delta^{(i+1)} \odot g^{(i)\prime}(z^{(i)})
   $



*   激活函数导数：


    *   ReLU：$g'(z) = 1$（z>0），$g'(z) = 0$（z≤0）；
    
    *   Sigmoid：$g'(z) = g(z) \cdot (1 - g(z))$；
    
    *   Tanh：$g'(z) = 1 - g(z)^2$。

#### 4. 计算参数梯度（Compute Gradients）



*   权重梯度：$\frac{\partial \mathcal{L}}{\partial W^{(i)}} = \delta^{(i)} \cdot (a^{(i-1)})^T$（i=1, ..., n）；

*   偏置梯度：$\frac{\partial \mathcal{L}}{\partial b^{(i)}} = \delta^{(i)}$（i=1, ..., n）。

#### 5. 参数更新（Update Weights）



*   梯度下降更新：

$ 
  W^{(i)} = W^{(i)} - \alpha \cdot \frac{\partial \mathcal{L}}{\partial W^{(i)}}, \quad b^{(i)} = b^{(i)} - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b^{(i)}}
   $



*   $\alpha$为学习率，控制更新步长。

### 配图说明



*   算法流程示意图：标注 “前向传播”“初始化$\delta$”“反向传播$\delta$”“计算梯度”“更新参数” 五个步骤，用箭头连接，标注关键公式（如$z^{(i)} = W^{(i)}a^{(i-1)} + b^{(i)}$、$\delta^{(i)} = W^{(i+1)T} \cdot \delta^{(i+1)} \odot g^{(i)\prime}(z^{(i)})$）。

### 英文术语标注

反向传播算法（Backpropagation Algorithm）、前向传播（Forward Pass）、反向传播（Backward Pass）、误差项初始化（Error Term Initialization）、参数更新（Parameter Update）、学习率（Learning Rate）

## 第 31 页：反向传播的应用与历史（BP: Most Widely Used in Deep Learning）

### 反向传播的应用领域



*   **核心应用**：深度神经网络训练的主流方法，几乎所有深度学习模型（如 CNN、RNN、GAN）均依赖反向传播计算梯度；

*   具体场景：

1.  **卷积神经网络（CNN）**：用于图像分类、目标检测（如 AlexNet、ResNet）；

2.  **生成对抗网络（GAN）**：用于生成逼真样本（如 DCGAN、StyleGAN）；

3.  **注意力模型（Attention Models）**：用于自然语言处理（如 Transformer、BERT）；

4.  **强化学习（Reinforcement Learning）**：用于智能决策（如 AlphaGo、DQN）。

### 历史里程碑



| 时间          | 事件（Event）                                          | 意义（Significance）               |
| ----------- | -------------------------------------------------- | ------------------------------ |
| 1969 年      | Bryson & Ho 提出反向传播原型（Applied Optimal Control）      | 早期控制论中的反向传播思想，为后续神经网络应用奠定基础。   |
| 1986 年      | Rumelhart et al. 将反向传播引入神经网络（Nature 论文）            | 首次证明反向传播可训练多层神经网络，推动第二次 AI 浪潮。 |
| 1989 年      | 神经科学家将反向传播用于认知模型（Neuroscientists' Cognitive Model） | 探索反向传播与大脑认知机制的关联，拓展跨学科研究。      |
| 2003 年      | 反向传播用于深度神经网络（Deep Neural Networks）                 | 突破浅层网络局限，为深度学习爆发铺垫。            |
| 2014-2015 年 | 反向传播用于 GAN、CNN（如 AlexNet）                          | 成为深度学习的核心优化方法，支撑计算机视觉、生成模型的突破。 |

### 核心论文引用



*   Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors." Nature, 323 (6088), 533-536.（反向传播的里程碑论文）

### 配图说明



*   应用领域示意图：标注 “CNN”“GAN”“Attention Models”“Reinforcement Learning”，配对应模型结构图（如 CNN 的卷积层、GAN 的生成器 - 判别器），直观展示反向传播的广泛应用。

### 英文术语标注

生成对抗网络（Generative Adversarial Networks, GAN）、注意力模型（Attention Models）、强化学习（Reinforcement Learning）、卷积神经网络（Convolutional Neural Networks, CNN）、深度神经网络（Deep Neural Networks, DNN）

### 第 32 页：反向传播的替代方法（Alternatives to BP for Training DNN）

#### 替代方法分类（按 “反馈机制” 与 “生物可解释性” 划分）

反向传播虽为当前主流，但存在 “与生物神经元通信机制不一致”“依赖精确梯度计算” 等局限，学界提出多种替代方法，核心分类及对比如下：



| 方法类型（Method Type）                | 代表方法（Representative Methods） | 核心思想（Core Idea）                                        | 生物可解释性（Biological Interpretability） | 技术局限性（Technical Limitation）                           |
| -------------------------------------- | ---------------------------------- | ------------------------------------------------------------ | ------------------------------------------- | ------------------------------------------------------------ |
| **无反馈驱动（No Feedback）**          | 前馈网络（Feedforward Network）    | 无反向梯度传递，仅通过 “输入→输出” 的前向映射拟合数据，依赖手工设计特征（如早期感知机）。 | 低（仅模拟神经元基本结构，无反馈机制）      | 仅能处理线性问题或简单非线性问题，无法训练深层网络，泛化能力差。 |
| **标量反馈驱动（Scalar Feedback）**    | 赫布学习（Hebbian Learning）       | 遵循 “同时激活的神经元连接增强”（Hebb's Law：$w_{ij} \propto a_i a_j$），通过标量奖励信号（如 “正确 / 错误”）调整权重。 | 高（模拟大脑突触可塑性的基础机制）          | 缺乏精确梯度指引，权重更新方向模糊，易收敛到局部最优，无法处理复杂任务（如图像分类）。 |
| **局部反馈驱动（Local Feedback）**     | 预测编码（Predictive Coding）      | 每层神经元预测下一层输入，通过 “预测误差”（局部信号）调整权重，无需全局损失反向传播。 | 高（符合大脑 “预测 - 误差修正” 的认知机制） | 误差传播范围有限，深层网络中误差易衰减，训练稳定性差，当前仅在浅层网络（<5 层）中验证有效。 |
| **进化算法驱动（Evolutionary Drive）** | 神经进化（Neuroevolution）         | 通过 “变异 - 选择” 进化网络结构与权重（如遗传算法），无需梯度计算，直接优化任务性能。 | 低（无生物认知机制对应，属工程优化方法）    | 计算成本极高（需评估大量候选网络），收敛速度慢，难以应用于大参数模型（如百万级参数 DNN）。 |

#### 关键对比结论



*   **性能差距**：在 MNIST（手写数字分类）、CIFAR-10（图像分类）等标准数据集上，反向传播的测试误差（如 1.17%）远低于替代方法（如赫布学习的测试误差 > 5%，预测编码的测试误差 > 3%）；

*   **适用场景**：替代方法仅在 “小模型 + 简单任务”（如线性回归、二分类）中可用，反向传播仍是深层网络（>10 层）训练的唯一可行方法；

*   **未来方向**：需结合生物机制（如突触可塑性、局部反馈）与工程效率（如梯度近似计算），探索 “生物可解释 + 高性能” 的混合优化方法。

#### 配图说明



*   左侧为 “反向传播流程”：标注 “全局损失→输出层→隐藏层→输入层” 的梯度传递路径，箭头为红色（代表全局信号）；

*   右侧为 “预测编码流程”：标注 “局部预测误差→当前层权重更新”，箭头为蓝色（代表局部信号）；

*   底部标注两种方法的 “测试误差对比”（MNIST 数据集）：反向传播 1.17% vs 预测编码 3.25%，直观展示性能差距。

#### 英文术语标注

赫布学习（Hebbian Learning）、预测编码（Predictive Coding）、神经进化（Neuroevolution）、突触可塑性（Synaptic Plasticity）、局部反馈（Local Feedback）、全局损失（Global Loss）

### 第 33 页：反向传播的优缺点（BP: Pros and Cons）



*   **技术层面（Technical view）**：反向传播存在陷入局部最小值（trapped into local minima）的问题。在深度神经网络中，损失函数的地形复杂，存在多个局部极小值。当算法收敛到某个局部最小值时，梯度为零，算法停止更新参数，但这个局部最小值可能并非全局最优解，导致模型性能受限。

*   **认知层面（Cognitive view）**：反向传播与生物神经元通信的生物学机制不一致（inconformity to biological mechanisms of neural communication）。生物大脑中的神经元通信是基于复杂的电化学信号传递和突触可塑性等机制，而反向传播是基于数学的梯度计算和参数更新，这种差异使得反向传播在模拟生物智能方面存在局限性。

*   **Geoffrey Hinton 观点**：“My view is throw it all away and start again.”（我的观点是把这一切都抛开，重新开始。）以及 “The future depends on some graduate students who are deeply suspicious of everything I have said.”（未来取决于一些对我所说的一切都持怀疑态度的研究生。），这表明即使是深度学习领域的权威，也意识到当前反向传播算法存在的问题，期待新的研究突破。

### 第 34 页：替代反向传播算法的对比实验（Alternatives to BP for Training DNN）



* **对比实验结果（以 MNIST 和 CIFAR 数据集为例）**：

  \| 方法 | FC 训练误差 | FC 测试误差 | LC 训练误差 | LC 测试误差 |

  \|---|---|---|---|---|

  \| DTP, PARALLEL | 0.44 | 2.86 | 0.00 | 1.52 |

  \| DTP, ALTERNATING | 0.00 | 1.83 | 0.00 | 1.46 |

  \| SDTP, PARALLEL | 1.14 | 3.52 | 0.00 | 1.98 |

  \| SDTP, ALTERNATING | 0.00 | 2.28 | 0.00 | 1.90 |

  \| AO-SDTP, PARALLEL | 0.96 | 2.93 | 0.00 | 1.92 |

  \| AO-SDTP, ALTERNATING | 0.00 | 1.86 | 0.00 | 1.91 |

  \| FA | 0.00, 1.85 |  | 0.00 | 1.26 |

  \| DFA | 0.85, 2.75 |  | 0.23 | 2.05 |

  \| BP | 0.00 | 1.48 | 0.00 | 1.17 |

  \| BP CONVNET |  |  | 0.00 | 1.01 |

* **结论**：目前，更接近大脑机制的算法（如 DTP、FA 等）在性能上仍远不及反向传播算法（BP）。尽管这些替代算法在理论上具有一定优势，试图模仿大脑的学习机制，但在实际应用中，它们在处理复杂任务时的表现还无法与反向传播相媲美。

### 第 35 页：深度神经网络训练的挑战（Theoretically, DNN can approximate any function Practically, training DNN with BP is difficult）



*   **损失函数优化困难（Loss functions can be difficult to optimize）**：深度神经网络的损失函数通常是非凸的，存在许多局部极小值和鞍点。这使得找到全局最优解变得非常困难，反向传播算法容易陷入局部最优，导致模型性能无法达到最佳状态。以 ResNet-56 和 ResNet-110（无跳跃连接，用于 CIFAR-10 数据集）的损失函数曲面为例，其地形复杂，优化过程容易陷入局部陷阱。

*   **过拟合问题（Dealing with overfitting）**：深度神经网络具有很强的拟合能力，当训练数据不足或模型过于复杂时，容易出现过拟合现象。过拟合表现为模型在训练集上表现良好，但在测试集上性能急剧下降，无法泛化到新的数据。这是因为模型学习到了训练数据中的噪声和特定模式，而不是真正的潜在规律。

### 第 36 页：神经网络训练的高级技术（Advanced Techniques in Neural Network Training）



*   **自适应学习率（Adaptive learning rates）**：合适的学习率能够使模型收敛更加平稳，避免陷入局部最小值。学习率过大可能导致模型在训练过程中跳过最优解，变得不稳定甚至发散；学习率过小则会使模型收敛速度过慢，增加训练时间。因此，需要设计一种自适应学习率，根据梯度大小、学习进展速度、权重大小等因素动态调整学习率。

*   **高级梯度下降算法（Advanced gradient descent algorithms）**：如 Momentum、RMSProp、Adam 等。这些算法在传统梯度下降的基础上进行改进，Momentum 通过平均连续的梯度来获得更好的更新方向；RMSProp 能够自适应地调整每个参数的学习率，防止学习率衰减过快或过慢；Adam 则结合了 Momentum 和 RMSProp 的优点，同时考虑了梯度的一阶矩和二阶矩，在实际应用中表现出色。

*   **正则化（Regularization）**：包括 L1、L2 正则化、Dropout、Early Stopping 等方法。正则化的目的是防止模型在训练数据上过度拟合，提高模型的泛化能力。L1 和 L2 正则化通过对权重进行约束，使模型更加简单；Dropout 在训练过程中随机丢弃一些神经元，防止特征之间的共适应；Early Stopping 则在模型在验证集上的性能不再提升时停止训练，避免过拟合。

*   **其他方面**：如 Batch normalization（批归一化）、Initialization（初始化）、Hyperparameters（超参数调整）等。批归一化通过标准化输入、激活和输出，确保有效激活，缓解大输入、激活和输出带来的问题；合适的初始化能够使模型训练更加稳定，避免梯度消失或爆炸；超参数调整则是优化模型性能的重要环节，包括选择合适的优化器、批量大小、学习率等超参数。

### 第 37 页：自适应学习率（Adaptive Learning Rates）



*   **合适学习率的重要性**：合适的学习率能够使模型在训练过程中平稳收敛，避免陷入局部最小值。在训练过程中，学习率如果选择不当，会对模型的训练效果产生严重影响。例如，学习率过大，模型在更新参数时会跳过最优解，导致损失函数不稳定，甚至发散；学习率过小，模型收敛速度会非常缓慢，需要更多的训练时间才能达到较好的效果。

*   **学习率对模型训练的影响**：


    *   **高学习率（high learning rate）**：高学习率会使模型在参数更新时步长过大，可能导致模型跳过最优解，使损失函数在训练过程中出现剧烈波动，无法收敛到较好的结果，甚至可能使模型发散。
    
    *   **低学习率（low learning rate）**：低学习率会使模型收敛速度过慢，在训练过程中，模型需要更多的迭代次数才能接近最优解，这不仅会增加训练时间，还可能使模型陷入局部最小值，无法找到全局最优解。
    
    *   **合适学习率（good learning rate）**：合适的学习率能够平衡模型的收敛速度和稳定性，使模型在训练过程中逐渐接近最优解，损失函数能够平稳下降。

### 第 38 页：自适应学习率（Adaptive Learning Rates）



*   **设计自适应学习率**：为了克服固定学习率的缺点，需要设计一种能够 “适应” 训练过程的自适应学习率。这种学习率不是固定不变的，而是可以根据梯度大小、学习进展速度、特定权重的大小等因素进行动态调整。例如，当梯度较大时，可以适当减小学习率，以避免更新步长过大；当学习进展缓慢时，可以适当增大学习率，加快模型的收敛速度。

*   **学习率衰减策略（Learning rate decay schedules）**：学习率衰减策略对于随机梯度下降（SGD）的最佳性能至关重要。常见的学习率衰减策略包括线性衰减、指数衰减等。通过在训练过程中逐渐减小学习率，可以使模型在训练初期快速收敛，在接近最优解时更加精细地调整参数，从而提高模型的性能。

### 第 39 页：自适应学习率（Adaptive Learning Rates）



*   **AlexNet 在 ImageNet 上的训练示例**：以 AlexNet 在 ImageNet 数据集上的训练为例，展示了不同学习率对模型训练准确率的影响。在训练过程中，使用不同的学习率（如$\alpha = 0.01$、$\alpha = 0.001$、$\alpha = 0.0001$）进行训练，可以观察到不同学习率下模型的训练准确率曲线。结果表明，合适的学习率能够使模型更快地收敛到较高的准确率，而不合适的学习率则会导致模型收敛缓慢或无法达到较好的准确率。

### 第 40 页：高级梯度下降算法（Advanced Gradient Descent Algorithms）



*   **Momentum 算法**：Momentum 算法的核心思想是，最陡峭的方向并不总是最好的，将连续的梯度进行平均似乎可以得到更好的更新方向。其直观理解是，如果连续的梯度步骤指向不同的方向，那么这些不一致的方向应该相互抵消；如果连续的梯度步骤指向相似的方向，那么在这个方向上应该加快前进速度。Momentum 算法在传统梯度下降的基础上，增加了一个动量项，用于积累之前的梯度信息，从而加快模型的收敛速度。同时，Momentum 算法使用的内存比随机梯度下降更多，因为它需要存储动量项。

*   **RMSProp 算法**：RMSProp 算法通过估计每个维度的梯度大小，自适应地调整学习率，防止学习率衰减过快或过慢。它通过计算梯度的平方的移动平均来调整学习率，对于梯度波动较大的参数，减小学习率；对于梯度波动较小的参数，保持学习率。RMSProp 算法在处理非平稳目标函数时表现良好，能够使模型更加稳定地收敛。与随机梯度下降和 Momentum 算法相比，RMSProp 算法使用更多的内存，因为它需要存储梯度平方的移动平均信息。

*   **Adam 算法**：Adam 算法结合了 Momentum 和 RMSProp 的优点，同时考虑了梯度的一阶矩（动量项）和二阶矩（梯度平方的移动平均）。它通过计算偏差修正后的一阶矩和二阶矩来调整学习率，使得学习率能够自适应地变化。Adam 算法的超参数通常设置为预定义的值，不需要进行过多的调整，在许多情况下都能表现出良好的性能。Adam 算法在训练过程中进行学习率退火，通过自适应的步长调整，使模型能够更快地收敛到最优解。同时，Adam 算法在给定的批量大小下使用的内存最多，但由于其良好的性能，常常被用作机器学习中的默认优化器。

### 第 41 页：高级梯度下降算法（Advanced Gradient Descent Algorithms）



*   **Momentum 算法原理**：Momentum 算法的更新公式为$V_{dW}=\beta V_{dW}+(1 - \beta)dW$，$W = W-\alpha V_{dW}$，其中$V_{dW}$是动量项，$\beta$是动量系数（通常设置为 0.9），$\alpha$是学习率，$dW$是当前的梯度。在更新参数时，Momentum 算法不仅考虑当前的梯度，还考虑之前的梯度信息，通过动量项将之前的梯度方向 “融合” 进来，使得参数更新更加稳定。如果连续的梯度方向相似，动量项会加速参数更新；如果梯度方向不一致，动量项会抵消部分振荡，使更新更加平稳。

*   **算法效果展示**：通过图表展示了 Momentum 算法在训练过程中参数值的变化情况。可以看到，与传统的随机梯度下降相比，Momentum 算法能够更快地收敛到最优解，减少了参数更新过程中的振荡，提高了训练效率。

### 第 42 页：高级梯度下降算法（Advanced Gradient Descent Algorithms）



*   **RMSProp 算法原理**：RMSProp 算法通过计算梯度的平方的移动平均来调整学习率，公式为$S_{dW}=\beta S_{dW}+(1 - \beta)dW^{2}$，$W = W-\alpha\frac{dW}{\sqrt{S_{dW}}+\varepsilon}$，其中$S_{dW}$是梯度平方的移动平均，$\beta$是衰减系数（通常设置为 0.9），$\alpha$是学习率，$\varepsilon$是一个小的常数（通常设置为$1e-8$），用于避免分母为零。RMSProp 算法能够根据每个参数的梯度波动情况，自适应地调整学习率，对于梯度波动较大的参数，学习率会减小，从而使更新更加稳定；对于梯度波动较小的参数，学习率会保持相对较大的值，加快收敛速度。

*   **算法效果展示**：通过图表展示了 RMSProp 算法在训练过程中的优化效果。可以看到，RMSProp 算法能够有效地适应不同参数的梯度变化，使模型在训练过程中更加稳定地收敛，避免了学习率衰减过快或过慢的问题。

### 第 43 页：高级梯度下降算法（Advanced Gradient Descent Algorithms）



*   **Adam 算法原理**：Adam 算法结合了动量和 RMSProp 的思想，其更新公式包括计算一阶矩（动量项）$V_{dW}=\beta_{1}V_{dW}+(1 - \beta_{1})dW$和二阶矩（RMSProp-like）$S_{dW}=\beta_{2}S_{dW}+(1 - \beta_{2})dW^{2}$，以及偏差修正$V corr_{dW}=\frac{V_{dW}}{(1 - \beta_{1})^{t}}$，$S corr_{dW}=\frac{S_{dW}}{(1 - \beta_{2})^{t}}$，其中$\beta_{1}$通常设置为 0.9，$\beta_{2}$通常设置为 0.999，$t$是当前的迭代次数。最终的参数更新公式为$W = W-\alpha\frac{V corr_{dW}}{\sqrt{S corr_{dW}}+\varepsilon}$。Adam 算法通过同时考虑梯度的一阶矩和二阶矩，能够自适应地调整学习率，在训练初期，学习率较大，加快收敛速度；在训练后期，学习率逐渐减小，使模型更加精细地调整参数。

*   **算法效果展示**：通过图表对比了 Adam 算法与其他优化器（如 SGD、Momentum、RMSProp）在训练过程中的表现。可以看出，Adam 算法在收敛速度和稳定性方面表现出色，能够更快地找到最优解，并且在训练过程中损失函数下降更加平稳。

* ### 第 44 页：正则化（Regularization：L1/L2 与 Dropout）

  #### 一、L1 与 L2 正则化：权重约束机制

  正则化的核心是 “通过约束模型复杂度，减少过拟合”，L1 与 L2 是最常用的权重正则化方法，通过在损失函数中添加正则项实现：

  ##### 1. 损失函数定义

  设原始损失为$\mathcal{L}_{0}$（如交叉熵、MSE），正则化后的总损失为：

  $ 
  \mathcal{L} = \mathcal{L}_{0} + \lambda \cdot R(W)
   $

  其中：

  

  *   $\lambda$为正则化强度（超参数）：$\lambda$越大，约束越强，模型越简单；$\lambda=0$时无正则化；

  *   $R(W)$为正则化项，L1 与 L2 的定义不同。

  ##### 2. L1 正则化（L1 Regularization）

  

  *   **正则化项**：$R(W) = \sum_{i,j} |w_{ij}|$（所有权重的绝对值之和）；

  *   **权重更新公式**（以 SGD 为例）：

  $ 
    w_{ij} \leftarrow w_{ij} - \alpha \cdot \left( \frac{\partial \mathcal{L}_0}{\partial w_{ij}} + \lambda \cdot \text{sign}(w_{ij}) \right)
     $

  其中$\text{sign}(w_{ij})$为符号函数（$w_{ij}>0$时为 1，$w_{ij}<0$时为 - 1）；

  

  *   **核心特点**：


      *   权重稀疏化：迫使部分权重变为 0，相当于 “特征选择”，保留关键特征，降低模型复杂度；
      
      *   鲁棒性强：对异常值不敏感（绝对值损失比平方损失更抗噪），适合噪声数据（如医疗影像）。

  ##### 3. L2 正则化（L2 Regularization，又称权重衰减）

  

  *   **正则化项**：$R(W) = \sum_{i,j} w_{ij}^2$（所有权重的平方和）；

  *   **权重更新公式**（以 SGD 为例）：

  $ 
    w_{ij} \leftarrow w_{ij} - \alpha \cdot \left( \frac{\partial \mathcal{L}_0}{\partial w_{ij}} + 2\lambda \cdot w_{ij} \right)
     $

  

  *   **核心特点**：


      *   权重平滑化：使权重趋近于 0 但不轻易为 0，避免极端权重，降低过拟合风险；
      
      *   计算高效：导数计算简单（无符号函数的不连续性），训练稳定性高于 L1；
      
      *   应用广泛：CNN、Transformer 等大模型默认使用 L2 正则化，平衡性能与复杂度。

  ##### 4. L1 与 L2 的对比

  

  | 对比维度（Dimension） | L1 正则化（L1 Regularization） | L2 正则化（L2 Regularization） |
  | --------------------- | ------------------------------ | ------------------------------ |
  | 权重分布              | 稀疏分布（部分权重为 0）       | 密集分布（权重趋近于 0）       |
  | 导数连续性            | 不连续（$w=0$处不可导）        | 连续（全区间可导）             |
  | 计算复杂度            | 高（需处理稀疏矩阵）           | 低（密集矩阵运算高效）         |
  | 适用场景              | 特征选择、噪声数据             | 大模型训练、通用任务           |

  #### 二、Dropout：随机神经元丢弃机制

  当 L1/L2 正则化效果有限时（如深层网络过拟合严重），Dropout 通过 “随机丢弃神经元” 模拟 “集成学习”，大幅提升泛化能力，是计算机视觉、NLP 领域的核心正则化技术。

  ##### 1. 核心原理

  

  *   **训练阶段**：对每个隐藏层，随机选择 50% 的神经元（默认比例），将其激活值设为 0（“丢弃”），仅保留剩余 50% 神经元参与前向与反向传播；

  *   **测试阶段**：不丢弃任何神经元，将所有神经元的激活值乘以 0.5（或训练时对保留神经元乘以 2），确保输入输出的数值范围一致（避免测试时输出值偏大）；

  *   **本质**：训练时生成大量 “子模型”（每个子模型对应一种神经元丢弃组合），测试时通过 “平均子模型输出” 降低方差，减少过拟合。

  ##### 2. 数学表示（以单隐藏层为例）

  设隐藏层激活输出为$a^{(l)} \in \mathbb{R}^h$（h 为隐藏层神经元数），Dropout 操作后输出为$\tilde{a}^{(l)}$：

  

  *   训练阶段：生成掩码矩阵$m^{(l)} \sim \text{Bernoulli}(p=0.5)$（元素为 0 或 1，1 的概率为 0.5），则$\tilde{a}^{(l)} = a^{(l)} \odot m^{(l)}$；

  *   测试阶段：$\tilde{a}^{(l)} = a^{(l)} \times 0.5$（或训练时$\tilde{a}^{(l)} = (a^{(l)} \odot m^{(l)}) / 0.5$，两种方式等价）。

  ##### 3. 关键优势

  

  *   **无需额外参数**：仅需设置丢弃比例（如 0.5），不增加模型复杂度；

  *   **集成效果显著**：理论上可生成$2^h$个子模型（h 为隐藏层神经元数），集成效果远超传统集成方法（如随机森林）；

  *   **适配深层网络**：在 ResNet、BERT 等大模型中，Dropout 可将测试误差降低 1\~3 个百分点，是训练稳定的关键。

  ##### 4. 注意事项

  

  *   **丢弃比例选择**：隐藏层通常设为 0.5，输出层设为 0.1\~0.2（避免输出层信息丢失过多）；

  *   **批量大小影响**：批量较小时（如 < 32），Dropout 的随机性过强，可能导致训练不稳定，需结合批归一化使用；

  *   **不适用场景**：小模型（<1000 参数）或简单任务（如线性回归），Dropout 可能导致欠拟合。

  #### 配图说明

  

  *   上半部分：L1 与 L2 正则化的 “权重分布对比图”：


      *   L1：权重集中在 0 附近，部分权重为 0（稀疏峰）；
      
      *   L2：权重呈正态分布，无明显 0 值（密集峰）；

  *   下半部分：Dropout 的 “训练 - 测试流程对比图”：


      *   训练时：隐藏层 50% 神经元被红色叉号标注（丢弃），信号仅通过黑色神经元传递；
      
      *   测试时：所有神经元激活，输出乘以 0.5（标注 “×0.5”），箭头为蓝色（代表完整信号）。

  #### 英文术语标注

  L1 正则化（L1 Regularization）、L2 正则化（L2 Regularization）、权重衰减（Weight Decay）、Dropout、掩码矩阵（Mask Matrix）、伯努利分布（Bernoulli Distribution）、集成学习（Ensemble Learning）、欠拟合（Underfitting）



## 第 45 页：Dropout 正则化（Regularization: Dropout）

### Dropout 核心原理与训练机制



*   **训练阶段操作**：在训练过程中，随机将网络中部分神经元的激活值设置为 0（通常 “丢弃” 50% 的激活值），迫使网络不依赖任何单个神经元或特征，学习冗余且鲁棒的特征表示。例如，在判断 “是否为猫” 的任务中，即使随机关闭 “有耳朵”“有尾巴” 等特征对应的神经元，网络仍能通过其他特征（如 “毛茸茸”“有爪子”）做出正确判断，避免特征共适应（co-adaptation of features）。

*   **关键作用**：通过随机丢弃神经元，模拟 “训练多个子模型并集成” 的效果（每个二进制掩码对应一个子模型），但无需额外存储多个模型参数，大幅降低过拟合风险。以含 4096 个单元的全连接层为例，理论上存在$2^{4096}\approx10^{1233}$种可能的掩码组合，相当于训练海量子模型的集成。

### 配图与示例



*   图示展示训练时的随机掩码（random mask）：左图为原始网络结构，右图为应用掩码后部分神经元被 “关闭” 的状态，箭头标注信号传递路径的截断，直观呈现 Dropout 的 “随机丢弃” 过程。

### 英文术语标注

Dropout 正则化（Dropout Regularization）、特征共适应（Co-adaptation of Features）、二进制掩码（Binary Mask）、激活值（Activation Value）

## 第 46 页：Dropout 训练细节（Regularization: Dropout）

### 训练阶段的动态调整



*   **丢弃比例与随机性**：训练时通常对隐藏层采用 50% 的丢弃比例（“drop 50% of activations”），确保网络学习到足够冗余的特征；丢弃过程完全随机，每个神经元被选中丢弃的概率独立，避免固定模式导致的过拟合。

*   **参数更新特点**：由于神经元随机被丢弃，每次迭代训练的 “有效网络结构” 不同，但参数共享（所有子模型共用一套权重），更新后的参数需适应多种子模型，间接提升泛化能力。

### Epoch 1 训练示例



*   图示标注 “Epoch 1” 与 “random mask”：展示某一轮训练中，网络通过随机掩码关闭部分隐藏层神经元（如左侧第 2 个、右侧第 1 个神经元），输入信号仅通过未被丢弃的神经元传递至输出层，模拟 “子模型训练”。

### 英文术语标注

丢弃比例（Dropout Rate）、有效网络结构（Effective Network Structure）、参数共享（Parameter Sharing）、迭代（Iteration）

## 第 47 页：Dropout 训练细节（Regularization: Dropout）

### Epoch 2 训练示例



*   延续第 46 页逻辑，图示标注 “Epoch 2” 与新的 “random mask”：新一轮训练中，随机掩码与 Epoch 1 完全不同（如关闭左侧第 1 个、右侧第 3 个神经元），确保网络在不同 “子模型” 下均能学习，避免对特定神经元的依赖。

*   **核心差异**：两次迭代的掩码无重叠规律，迫使网络在每次训练时都需重新组合特征，最终学到的特征更通用（如 “猫” 的特征不仅依赖 “耳朵”，还包括 “眼睛形状”“毛发纹理” 等）。

### 英文术语标注

轮次（Epoch）、特征组合（Feature Combination）、通用特征（General Feature）

## 第 48 页：Dropout 测试阶段处理（Regularization: Dropout）

### 测试阶段的 “集成” 逻辑



*   **核心操作**：测试时不再随机丢弃神经元，而是 “合并所有子模型”（Combine all the models），通过 “平均化随机性”（Average out the randomness）得到最终预测结果。例如，若训练时 50% 神经元被丢弃，测试时需将所有神经元的输出乘以 0.5（或训练时对未丢弃神经元乘以 2），确保输入输出的数值范围一致（避免测试时输出值偏大）。

*   **蒙特卡洛 Dropout（Monte Carlo Dropout）**：进阶用法，测试时仍保留随机丢弃机制，多次预测后取平均值，进一步提升结果稳定性，但计算成本较高，适用于对精度要求极高的场景（如医疗诊断）。

### 配图说明



*   图示标注 “Test time” 与 “Inference time Monte Carlo dropouts”：展示测试时输入信号通过全部神经元，输出层整合所有路径的信号，箭头标注 “平均化” 过程，对比训练时的 “随机截断”，突出测试阶段的 “集成” 特性。

### 英文术语标注

测试阶段（Test Time）、推理阶段（Inference Time）、蒙特卡洛 Dropout（Monte Carlo Dropout）、平均化（Averaging）

## 第 49 页：早停正则化（Regularization: Early Stopping）

### 早停的核心逻辑



*   **原理**：在模型开始过拟合前停止训练（Stop training before overfitting），通过监控验证集损失（Validation Loss）判断训练终止时机 —— 当训练集损失持续下降但验证集损失开始上升时，说明模型已开始 “死记硬背” 训练数据，此时停止训练可保留泛化能力最佳的参数。

*   **优势**：无需修改损失函数或网络结构，仅通过训练流程控制即可缓解过拟合，计算成本低，是实践中常用的 “轻量级正则化方法”。

### 损失曲线与停止时机



*   图示为 “损失值 - 训练迭代次数” 曲线：


    *   横轴（Training Iterations）：训练迭代次数；
    
    *   纵轴（Loss）：损失值；
    
    *   曲线：蓝色为训练集损失（持续下降），红色为验证集损失（先降后升）；
    
    *   标注 “Stop training here!”：指向验证集损失开始上升的拐点，明确早停的最佳时机。

### 英文术语标注

早停（Early Stopping）、验证集损失（Validation Loss）、训练集损失（Training Loss）、过拟合（Overfitting）、迭代次数（Training Iterations）

## 第 50 页：神经网络训练的其他高级技术（Advanced Techniques in Neural Network Training）

### 技术框架回顾



*   重申训练深度神经网络的四大核心技术方向，与第 36 页呼应，强化知识体系：

1.  **自适应学习率（Adaptive learning rates）**：动态调整学习率，平衡收敛速度与稳定性；

2.  **高级梯度下降算法（Advanced gradient descent algorithms）**：Momentum、RMSProp、Adam 等，优化参数更新效率；

3.  **正则化（Regularization）**：L1/L2、Dropout、早停，缓解过拟合；

4.  **其他关键技术**：批归一化（Batch normalization）、参数初始化（Initialization）、超参数调优（Hyperparameters），解决训练稳定性与效率问题。

### 配图说明



*   无新增图像，仅文本梳理技术框架，为后续分页讲解 “批归一化”“初始化” 等内容铺垫，确保逻辑连贯。

### 英文术语标注

批归一化（Batch Normalization）、参数初始化（Parameter Initialization）、超参数调优（Hyperparameter Tuning）、训练稳定性（Training Stability）

## 第 51 页：批归一化（Batch Normalization）

### 批归一化的核心动机



*   **解决 “内部协变量偏移（Internal Covariate Shift）”**：训练过程中，每层输入的分布随前层参数更新而变化（如第一层权重更新导致第二层输入均值从 0 变为 2），后层需不断适应新分布，导致收敛缓慢。批归一化通过标准化每层输入，固定其分布（均值 0、方差 1），稳定训练过程。

*   **缓解极端值影响**：避免输入 / 激活 / 输出值过大或过小（如 ReLU 激活后部分神经元输出为 0，部分为 100+），确保激活函数处于 “有效梯度区域”（如 ReLU 的 x>0 区域、Sigmoid 的中间区域），减少梯度消失。

### 标准化操作与 ReLU 激活的配合



*   **标准化公式**：

1.  去均值：$\bar{x}_i = x_i - E[x]$，其中$E[x] \approx \frac{1}{N}\sum_{i=1}^N x_i$（N 为批量大小）；

2.  去方差：$\bar{x}_i = \frac{x_i - E[x]}{\sqrt{E[(x_i - E[x])^2]}}$，确保方差为 1；

*   **与 ReLU 的协同**：标准化后，输入值集中在 \[-1,1]，ReLU 激活后仅保留 x>0 部分，避免大量神经元 “死亡”（x≤0 时 ReLU 输出为 0，梯度为 0），提升网络表达能力。

### 配图说明



*   图示对比 “无归一化（Without Normalization）” 与 “有归一化（With Normalization）” 的 ReLU 激活分布：左图无归一化时，激活值分布分散（部分为 - 10\~0，部分为 5\~10），大量神经元 “死亡”；右图有归一化时，激活值集中在 - 1\~1，ReLU 有效激活比例显著提升。

### 英文术语标注

内部协变量偏移（Internal Covariate Shift）、标准化（Standardization）、批量大小（Batch Size）、有效梯度区域（Effective Gradient Region）、神经元死亡（Neuron Death）

## 第 52 页：批归一化的层内流程（Batch Normalization）

### 批归一化在网络层中的具体步骤

以 “输入→线性层→ReLU→批归一化→线性层→输出” 的简化网络为例，详细流程如下：



1.  **线性层 1 计算**：$z^{(1)} = W^{(1)}x + b^{(1)}$，$a^{(1)} = \text{ReLU}(z^{(1)})$；

2.  **批归一化计算**：

*   计算批量均值：$\mu^{(1)} = \frac{1}{N}\sum_{i=1}^N a^{(1)}_i$；

*   计算批量方差：$\sigma^{(1)} = \sqrt{\frac{1}{N}\sum_{i=1}^N (a^{(1)}_i - \mu^{(1)})^2}$；

*   标准化：$\bar{a}^{(1)}_i = \frac{a^{(1)}_i - \mu^{(1)}}{\sigma^{(1)}}$；

1.  **线性层 2 计算**：$z^{(2)} = W^{(2)}\bar{a}^{(1)} + b^{(2)}$，最终输出。

### 计算成本的权衡



*   **问题**：若对全量数据计算均值 / 方差，每轮梯度更新需遍历所有样本，计算成本极高（“don't want to evaluate all points in the dataset every gradient step”）；

*   **解决方案**：用当前批量（mini-batch）的均值 / 方差近似全量数据分布，公式调整为：

$ 
  \mu^{(1)} \approx \frac{1}{B}\sum_{j=1}^B a^{(1)}_{i_j}, \quad \sigma^{(1)} \approx \sqrt{\frac{1}{B}\sum_{j=1}^B (a^{(1)}_{i_j} - \mu^{(1)})^2}
   $

其中 B 为批量大小（如 32、64），在计算成本与精度间取得平衡。

### 配图说明



*   图示标注 “ReLU→BN→softmax→cross-ent loss” 的流程，用箭头连接各层，标注批量均值$\mu^{(1)}$、方差$\sigma^{(1)}$的计算位置，直观展示批归一化在网络中的作用节点。

### 英文术语标注

迷你批量（Mini-Batch）、全量数据（Full Dataset）、计算成本（Computational Cost）、分布近似（Distribution Approximation）

## 第 53 页：批归一化的训练与测试差异（Batch Normalization）

### 训练阶段：批量统计更新



*   **核心操作**：训练时，用当前迷你批量的均值 / 方差更新 “移动平均统计量”（Running Mean/Variance），公式为：

$ 
  \text{RunningMean} = \gamma \times \text{RunningMean} + (1-\gamma) \times \mu_{\text{batch}}
   $

$ 
  \text{RunningVar} = \gamma \times \text{RunningVar} + (1-\gamma) \times \sigma_{\text{batch}}^2
   $

其中$\gamma$为衰减系数（通常取 0.99），确保统计量平滑更新，接近全量数据分布。



*   **可学习参数**：引入缩放因子$\gamma$与偏移因子$\beta$（“learnable scale and bias”），标准化后执行$\bar{a}_i = \gamma \cdot \bar{a}_i + \beta$，恢复网络表达能力（避免标准化后特征分布过于固定）。

### 测试阶段：固定统计量



*   **核心操作**：测试时不再使用当前批量的均值 / 方差，而是直接调用训练阶段积累的 “移动平均统计量”（“use fixed stats to normalize”），确保测试结果稳定（同一输入始终得到相同输出）。

*   **示例流程**：测试样本输入后，直接用$\text{RunningMean}$和$\text{RunningVar}$标准化，无需重新计算批量统计，提升推理效率。

### 配图说明



*   图示对比训练与测试流程：训练时标注 “random minibatches”，测试时标注 “fixed stats”，箭头分别指向 “更新移动平均” 与 “直接使用移动平均”，清晰区分两阶段差异。

### 英文术语标注

移动平均统计量（Running Mean/Variance）、衰减系数（Decay Coefficient）、缩放因子（Scale Factor）、偏移因子（Bias Factor）、推理效率（Inference Efficiency）

## 第 54 页：批归一化的优势（Batch Normalization）

### 批归一化的四大核心优势



1.  **支持更大学习率**：标准化后输入分布稳定，学习率可从$1e-4$提升至$1e-2$，收敛速度加快 2\~3 倍；

2.  **训练速度提升**：无需等待前层适应输入分布，模型快速收敛，如 ResNet 训练时间从 “数周” 缩短至 “数天”；

3.  **减少正则化依赖**：标准化本身具有轻微正则化效果（批量统计的随机性），可降低 Dropout、L2 正则化的强度（“requires less regularization”）；

4.  **鲁棒性更强**：对初始化、学习率的敏感度降低，即使参数初始化略差，仍能稳定训练（“good idea in many cases”）。

### 公式回顾与强调



*   重申批归一化核心公式，强化记忆：

$ 
  \mu^{(1)} \approx \frac{1}{B}\sum_{j=1}^B a^{(1)}_{i_j}, \quad \sigma^{(1)} \approx \sqrt{\frac{1}{B}\sum_{j=1}^B (a^{(1)}_{i_j} - \mu^{(1)})^2}
   $

$ 
  \bar{a}_i^{(1)} = \frac{a_i^{(1)} - \mu^{(1)}}{\sigma^{(1)}} \gamma + \beta
   $

### 配图说明



*   图示标注 “ReLU→BN→softmax→cross-ent loss” 的完整流程，突出批归一化在 “稳定激活值分布” 中的关键作用，与前页内容形成闭环。

### 英文术语标注

收敛速度（Convergence Speed）、正则化依赖（Regularization Dependency）、鲁棒性（Robustness）、参数敏感度（Parameter Sensitivity）

## 第 55 页：参数初始化的重要性（Initialization）

### 初始化不当的问题



*   **权重过大（Too large initialization）**：导致线性变换后$z = Wx + b$的绝对值过大，激活函数（如 Sigmoid、Tanh）进入饱和区（导数接近 0），反向传播时梯度消失（exploding gradients），参数无法更新；

*   **权重过小（Too small initialization）**：导致$z$的绝对值过小，激活值集中在 0 附近，反向传播时梯度同样接近 0（vanishing gradients），网络学习缓慢；

*   **零初始化（Zero Initialization）**：所有权重初始化为 0，导致所有神经元输出相同，梯度相同，参数更新后仍保持对称，网络无法学习特征。

### 初始化流程与观察指标



*   **初始化流程**：

1.  选择初始化方法（如 Xavier、He 初始化）；

2.  训练网络；

3.  观察损失函数（cost function）与决策边界（decision boundary），判断初始化是否合适；

*   **合适初始化（Appropriate initialization）**：损失函数快速下降，决策边界逐步逼近最优解，无梯度消失 / 爆炸问题。

### 配图说明



*   图示为 “损失值 - 轮次（Epoch）” 曲线：


    *   红色曲线（Too large/Too small）：损失值下降缓慢或停滞；
    
    *   蓝色曲线（Appropriate）：损失值快速下降并收敛；
    
    *   标注 “100”“0” 等坐标刻度，直观对比不同初始化的效果。

### 英文术语标注

梯度消失（Vanishing Gradients）、梯度爆炸（Exploding Gradients）、饱和区（Saturation Region）、决策边界（Decision Boundary）、零初始化（Zero Initialization）

## 第 56 页：权重过小的问题（Initialization）

### 权重过小导致的梯度消失



*   **具体表现**：权重过小时，$z = Wx + b$的输出值集中在 0 附近，激活函数（如 ReLU）的输出也接近 0，反向传播时梯度$\frac{\partial L}{\partial W} = \delta \cdot x^T$同样接近 0，参数更新步长极小（$W = W - \alpha \cdot \text{small gradient}$），网络学习停滞（“leads to vanishing gradients”）。

### 初始化效果对比



*   图示为 “决策边界 - 轮次” 曲线：


    *   权重过小（Too small）：决策边界几乎不变，无法拟合数据分布；
    
    *   合适初始化（Appropriate）：决策边界快速调整，逐步正确分类样本；

*   参考链接：[https://www.deeplearning.ai/ai-notes/initialization/](https://www.deeplearning.ai/ai-notes/initialization/)（提供初始化方法的权威资料）。

### 英文术语标注

学习停滞（Learning Stagnation）、更新步长（Update Step Size）、数据分布（Data Distribution）、样本分类（Sample Classification）

## 第 57 页：合适初始化的效果（Initialization）

### 合适初始化的核心目标



*   **目标**：并非直接找到最优参数（“not to start at a good solution”），而是确保梯度与激活值 “行为良好”（well-behaved）—— 激活值分布均匀，梯度大小适中，避免消失 / 爆炸，为后续训练奠定基础。

### 效果对比与参考链接



*   **图示对比**：


    *   零初始化 / 权重过小 / 过大：损失值高且下降缓慢，决策边界错误；
    
    *   合适初始化：损失值快速下降至低水平，决策边界准确分类样本；

*   再次引用参考链接：[https://www.deeplearning.ai/ai-notes/initialization/](https://www.deeplearning.ai/ai-notes/initialization/)，引导深入学习。

### 英文术语标注

行为良好的梯度（Well-behaved Gradients）、激活值分布（Activation Distribution）、最优参数（Optimal Parameters）

## 第 58 页：主流初始化方法（Initialization）

### 初始化方法的研究进展



*   **核心结论**：“零初始化” 或 “相同随机值初始化” 无法有效训练网络（“cannot work well”），需根据激活函数选择专用初始化方法，关键研究成果包括：

1.  **Xavier 初始化（2010）**：适用于 Sigmoid/Tanh，通过控制输入输出方差一致，避免梯度消失；

2.  **He 初始化（2015）**：适用于 ReLU，考虑 ReLU “关闭一半神经元” 的特性，输入方差扩大 2 倍；

3.  **Fixup 初始化（2019）**：无需批归一化，通过层索引调整偏置，稳定深层网络训练；

4.  **彩票假说（2019）**：从随机初始化的网络中 “剪枝” 出稀疏子网络（“lottery ticket”），性能接近原网络。

### 激活函数与初始化的匹配



| 激活函数（Activation Function） | 适用初始化方法 | 核心原因                                  |
| ------------------------------- | -------------- | ----------------------------------------- |
| Sigmoid/Tanh                    | Xavier         | 输出均值接近 0，方差需与输入一致          |
| ReLU/ELU                        | He             | ReLU 仅保留 x>0 部分，输入方差需扩大 2 倍 |

### 配图说明



*   图示展示 Sigmoid、Tanh、ReLU 的函数曲线：


    *   Sigmoid：输出 \[0,1]，x=0 时导数最大（0.25）；
    
    *   Tanh：输出 \[-1,1]，x=0 时导数最大（1）；
    
    *   ReLU：x>0 时输出 x，x≤0 时输出 0；

*   标注函数公式，强化 “激活函数特性决定初始化方法” 的逻辑。

### 英文术语标注

Xavier 初始化（Xavier Initialization）、He 初始化（He Initialization）、Fixup 初始化（Fixup Initialization）、彩票假说（Lottery Ticket Hypothesis）、稀疏子网络（Sparse Subnetwork）

## 第 59 页：超参数概述（Hyperparameters）

### 超参数的分类



*   **优化相关超参数（Optimization related）**：直接影响训练过程，需优先调优：


    *   批量大小（batch size）：影响训练速度与稳定性（如 32、64）；
    
    *   学习率（learning rate）：控制更新步长（如 1e-3、1e-4）；
    
    *   动量系数（momentum）：Momentum/Adam 中的$\beta_1$（通常 0.9）；
    
    *   初始化方法（initialization）：Xavier/He 等；
    
    *   批归一化（batch normalization）：是否启用；
    
    *   Dropout 比例（Dropout rate）：如 0.5；

*   **结构与任务相关超参数（Both related）**：影响网络表达能力：


    *   网络架构（Architecture）：CNN/RNN/Transformer；
    
    *   层数与层大小（# and size of layers）：如 5 层、每层 128 个单元；
    
    *   激活函数（activation）：ReLU/Sigmoid 等。

### 优化器对比框架



*   列出主流优化器（SGD、SGD with Momentum、AdaGrad、RMSprop、Adam），为后续 “超参数调优” 铺垫，强调优化器选择是超参数调优的核心环节。

### 配图说明



*   无新增图像，仅文本分类超参数类型，构建 “优化 - 结构” 的超参数体系，为第 60-61 页的调优方法铺垫。

### 英文术语标注

超参数（Hyperparameters）、批量大小（Batch Size）、动量系数（Momentum Coefficient）、网络架构（Network Architecture）、优化器（Optimizer）

## 第 60 页：超参数对训练的影响（Hyperparameters）

### 学习率的关键影响



*   **学习率过高（very high learning rate）**：参数更新步长过大，损失函数震荡上升（“overfit and diverge”），无法收敛；

*   **学习率过低（low learning rate）**：损失函数下降缓慢，需大量轮次才能接近最优解，训练效率低；

*   **学习率合适（good learning rate）**：损失函数平稳下降，快速收敛到低损失值。

### 泛化相关超参数



*   **泛化相关超参数（Generalization related）**：包括正则化强度（L1/L2 的$\lambda$）、Dropout 比例等，需平衡 “训练精度” 与 “泛化能力”—— 正则化过强会导致欠拟合，过弱会导致过拟合。

### 配图说明



*   图示为 “损失值 - 轮次” 曲线：


    *   红色曲线（very high learning rate）：震荡上升；
    
    *   蓝色曲线（good learning rate）：平稳下降；
    
    *   绿色曲线（low learning rate）：缓慢下降；

*   右侧标注 “INPUT LAYER→HIDDEN LAYERS→OUTPUT LAYER” 的网络结构，强化 “超参数影响网络训练全流程” 的认知。

### 英文术语标注

学习率影响（Learning Rate Impact）、泛化能力（Generalization Ability）、欠拟合（Underfitting）、过拟合（Overfitting）、正则化强度（Regularization Strength）

## 第 61 页：超参数调优方法（Hyperparameters）

### 超参数调优的核心原则



1.  **区分超参数类型**：

*   优化相关超参数（学习率、动量）：问题早显现（训练初期损失不下降）；

*   结构相关超参数（层数、激活函数）：效果晚显现（训练完成后精度低）；

1.  **粗到精调优（Coarse to fine）**：

*   粗调：用宽范围随机搜索（如学习率 1e-5\~1e-1）快速筛选有效区间；

*   精调：在有效区间内用网格搜索（Grid Search）或贝叶斯优化（Bayesian Optimization）优化；

1.  **系统化搜索**：

*   网格搜索（Grid Search）：遍历所有超参数组合，适合小范围；

*   随机搜索（Random Search）：随机采样超参数组合，效率高于网格搜索（多数超参数对性能影响小）。

### 配图说明



*   图示对比 “Grid Layout” 与 “Random Layout”：


    *   网格搜索：超参数组合呈网格状分布，覆盖全面但效率低；
    
    *   随机搜索：超参数组合随机分布，更快找到最优组合；

*   标注 “Important Parameter”，突出关键超参数（如学习率）的调优优先级。

### 英文术语标注

粗到精调优（Coarse-to-Fine Tuning）、网格搜索（Grid Search）、随机搜索（Random Search）、贝叶斯优化（Bayesian Optimization）、超参数组合（Hyperparameter Combination）

## 第 62 页：超参数调优步骤（Hyperparameters）

### 七步超参数调优流程



1.  **检查初始损失（Check initial loss）**：确认初始参数设置合理（如初始损失接近随机猜测水平）；

2.  **小样本过拟合（Overfit a small sample）**：用 100\~1000 个样本训练，若无法过拟合，说明网络结构或初始化存在问题；

3.  **确定有效学习率（Find LR that makes loss go down）**：用学习率扫描（LR Sweep）找到使损失下降的学习率区间；

4.  **粗网格搜索（Coarse grid, 1-5 epochs）**：用少量轮次快速筛选超参数组合；

5.  **精网格搜索（Refine grid, train longer）**：在有效区间内延长训练轮次，优化超参数；

6.  **分析曲线（Look at loss and accuracy curves）**：判断是否过拟合 / 欠拟合，调整正则化等参数；

7.  **迭代优化（GOTO step 5）**：重复精调步骤，直至性能满足需求。

### 配图说明



*   图示为 “训练损失 / 精度 - 迭代次数” 曲线：


    *   左图（Training Loss）：合适超参数下损失快速下降；
    
    *   右图（Train/Val Accuracy）：训练精度上升，验证精度稳定，无过拟合；

*   标注 “Grid Search”“Random Search”，对比两种搜索方法的效率。

### 英文术语标注

学习率扫描（LR Sweep）、小样本过拟合（Small-Sample Overfitting）、迭代优化（Iterative Optimization）、训练精度（Training Accuracy）、验证精度（Validation Accuracy）

## 第 63 页：当前最优深度神经网络的规模（How ‘deep’ are SOTA Deep Neural Networks?）

### 计算机视觉领域的 SOTA 模型



| 模型（Model）            | 发布时间 | 核心特点（Core Feature）                         | 参数规模（Parameter Scale）     | 关键成果（Key Achievement）                             |
| ------------------------ | -------- | ------------------------------------------------ | ------------------------------- | ------------------------------------------------------- |
| ResNet                   | 2016     | 残差连接（Residual Connection），解决梯度消失    | 数百万（如 ResNet-50：2560 万） | 支持 1001 层训练，ImageNet 分类准确率超 90%             |
| DenseNet-201/ResNeXt-101 | 2017     | 密集连接 / 分组卷积，增强特征复用                | 数百万～千万级                  | 在目标检测、分割任务中性能领先                          |
| DALL-E                   | 2021     | Transformer 架构，文本 - 图像生成                | 120 亿参数                      | 实现零样本文本 - 图像生成（如 “会喷火的柯基”）          |
| Swin Transformer V2      | 2022     | 窗口注意力（Window Attention），支持高分辨率图像 | 30 亿参数                       | ImageNet 分类准确率超 91%，适合 4K 图像处理             |
| Stable Diffusion 3       | 2024     | 扩散模型（Diffusion Model），多模态生成          | 80 亿参数                       | 生成 4K 高分辨率图像，支持文本 - 图像 / 图像 - 图像生成 |

### 核心论文引用



*   He et al. (2016): "Deep residual learning for image recognition."（ResNet 的里程碑论文）；

*   Ramesh et al. (2021): "Zero-shot text-to-image generation."（DALL-E 的核心论文）；

*   Liu et al. (2022): "Swin transformer v2: Scaling up capacity and resolution."（Swin Transformer V2 论文）。

### 配图说明



*   图示为 Stable Diffusion 3 的文本 - 图像生成流程：标注 “text encoder→prior→decoder”，输入文本 “a corgi playing a trumpet throwing flame”，输出对应图像，直观展示大模型的生成能力。

### 英文术语标注

当前最优（State of the Art, SOTA）、残差连接（Residual Connection）、密集连接（Dense Connection）、窗口注意力（Window Attention）、扩散模型（Diffusion Model）、零样本生成（Zero-Shot Generation）

## 第 64 页：自然语言处理领域的 SOTA 模型（How ‘deep’ are SOTA Deep Neural Networks?）

### NLP 领域的 SOTA 模型



| 模型（Model） | 发布时间 | 核心特点（Core Feature）                 | 参数规模（Parameter Scale） | 关键成果（Key Achievement）                                  |
| ------------- | -------- | ---------------------------------------- | --------------------------- | ------------------------------------------------------------ |
| GPT-4         | 2023.03  | 多模态（文本 + 图像），120 个注意力层    | 1.76 万亿参数               | 在语言推理、代码生成、视觉问答任务中性能领先，支持多语言处理 |
| PaLM          | 2022.04  | 路径并行（Pathways）架构，118 个注意力层 | 5400 亿参数                 | 少样本学习能力强，在 MMLU 等基准测试中准确率超 80%           |
| LLaMA 2       | 2023     | 开源模型，优化对话与文本摘要             | 70 亿～700 亿参数           | 在对话任务中自然度高，适合低成本部署                         |
| BloombergGPT  | 2023     | 金融领域专用，训练数据含大量金融文本     | 500 亿参数                  | 金融文本分类、情感分析准确率超通用模型                       |

### 小样本学习性能



*   图示为 “参数规模 - 小样本算术任务准确率” 曲线：参数从 0.4B 增加到 175B 时，准确率从 30% 提升至 60%，证明 “参数规模与小样本能力正相关”。

### 核心论文引用



*   Brown et al. (2020): "Language models are few-shot learners."（GPT-3 论文，奠定大语言模型基础）；

*   Chowdhery et al. (2022): "Palm: Scaling language modeling with pathways."（PaLM 论文）。

### 英文术语标注

多模态（Multi-Modal）、路径并行（Pathways）、开源模型（Open-Source Model）、少样本学习（Few-Shot Learning）、金融领域专用模型（Finance-Specific Model）

## 第 65 页：大模型的资源消耗（How does DNN scale with resources?）

### 训练资源需求



*   **计算硬件与时间**：


    *   GPT-4：25000 块 GPU，训练 100 天，计算量超$10^{23}$ FLOPs；
    
    *   PaLM：6144 块 TPU，训练 50 天，计算量约$10^{22}$ FLOPs；
    
    *   GPT-3：1024 块 GPU，训练 34 天，计算量约$10^{21}$ FLOPs；

*   **成本与能耗**：


    *   Gemini-Ultra（2024）：训练成本约 10\~20 亿美元；
    
    *   碳排放：训练一个大模型（如 GPT-3）的碳排放相当于 5 辆美国汽车的终身排放量（MIT Technology Review 数据），环保问题凸显。

### 模型规模与性能的关系



*   **边际效益递减**：参数从 175B（GPT-3）增加到 1.76T（GPT-4），性能提升幅度逐渐减小，需结合高效架构（如 MoE）平衡规模与效率。

### 配图说明



*   图示为 “训练计算量 - MMLU 基准分数” 曲线：


    *   横轴：训练计算量（占 GPT-4 的百分比）；
    
    *   纵轴：MMLU 基准分数（衡量语言理解能力）；
    
    *   标注 GPT-4、Gemini Ultra、PaLM2 等模型的位置，直观展示 “计算量与性能正相关，但边际效益递减”。

### 英文术语标注

计算量（Computation Amount, FLOPs）、碳排放（Carbon Footprint）、边际效益递减（Diminishing Marginal Returns）、高效架构（Efficient Architecture）、MoE（Mixture of Experts）

## 第 66 页：大模型的碳排放与可持续性（How does DNN scale with resources?）

### 碳排放对比



*   **关键数据（Strubell et al. 研究）**：


    *   训练含 2.13 亿参数的 Transformer（带神经架构搜索）：碳排放 626,155 磅 CO₂当量；
    
    *   美国汽车终身排放量：约 126,000 磅 CO₂当量；
    
    *   纽约 - 旧金山往返航班（1 名乘客）：约 1,984 磅 CO₂当量；
    
    *   结论：训练大模型的碳排放是美国汽车终身排放量的 5 倍，环境影响显著。

### 可持续 AI 的发展方向



*   **优化方向**：

1.  模型压缩（剪枝、量化）：减少参数规模，降低计算量；

2.  绿色计算（Green Computing）：使用可再生能源供电的 GPU 集群；

3.  高效算法：如稀疏注意力、动态路由，减少冗余计算。

### 配图说明



*   图示为 “碳排放 - 活动类型” 柱状图：


    *   标注 “Transformer 训练”“美国汽车终身”“航班往返” 等活动的碳排放数值；
    
    *   红色柱状图（Transformer 训练）显著高于其他活动，突出环保问题的紧迫性。

### 英文术语标注

可持续 AI（Sustainable AI）、模型压缩（Model Compression）、绿色计算（Green Computing）、稀疏注意力（Sparse Attention）、动态路由（Dynamic Routing）、可再生能源（Renewable Energy）

## 第 67 页：第五讲总结（Lecture 5: Deep Neural Networks - Summary）

### 核心知识体系回顾



1.  **神经网络基础**：

*   人工神经元结构（输入 - 权重 - 求和 - 激活 - 输出），激活函数（ReLU/Sigmoid/Tanh）的作用；

*   深度神经网络（DNN）的层级结构，CNN/RNN/Transformer 的适用场景。

1.  **训练核心技术**：

*   反向传播（BP）：通过链式法则高效计算梯度，复杂度从$O(n^3)$降至$O(n^2)$；

*   优化算法：Momentum（动量）、RMSProp（自适应学习率）、Adam（动量 + 自适应）；

*   正则化：Dropout（随机丢弃）、L1/L2（权重约束）、早停（防止过拟合）；

*   批归一化：解决内部协变量偏移，加速收敛；

*   初始化：Xavier（Sigmoid/Tanh）、He（ReLU），避免梯度消失 / 爆炸。

1.  **大模型挑战与趋势**：

*   规模与资源：参数达万亿级，训练成本高、碳排放大；

*   未来方向：模型压缩、绿色计算、高效架构（MoE），平衡性能与可持续性。

### 配图说明



*   图示为 DNN 训练流程闭环：标注 “Neural Networks→Backpropagation→Training in Practice”，箭头连接各环节，形成 “模型 - 优化 - 训练” 的完整知识框架。

### 英文术语标注

知识体系（Knowledge System）、链式法则（Chain Rule）、梯度计算（Gradient Computation）、模型可持续性（Model Sustainability）、高效架构（Efficient Architecture）

## 第 68 页：后续课程预告（Next Lectures）

### 第六讲：卷积神经网络（CNN）- I



*   **核心内容**：


    *   CNN 的基本结构：卷积层（局部连接、参数共享）、池化层（降维）、全连接层；
    
    *   经典 CNN 架构：AlexNet、VGG、ResNet，及其在图像分类中的应用；
    
    *   CNN 的数学原理：卷积运算、感受野（Receptive Field）的计算。

### 其他后续主题



*   **第七讲：CNN - II**：目标检测（Faster R-CNN、YOLO）、语义分割（FCN、U-Net）；

*   **第九讲：循环神经网络（RNN）与图神经网络（GNN）**：LSTM/GRU 处理序列数据，GCN/GAT 处理图数据；

*   **实践环节**：深度学习编程教程（基于 PyTorch），含 GAN、生成模型的代码实现。

### 配图说明



*   图示标注 “Convolutional Neural Network (CNN)”“Generative Adversarial Networks (GAN)”，配 CNN 卷积操作示意图与 GAN 的生成器 - 判别器结构，激发后续学习兴趣。

### 英文术语标注

卷积层（Convolutional Layer）、池化层（Pooling Layer）、感受野（Receptive Field）、目标检测（Object Detection）、语义分割（Semantic Segmentation）、图神经网络（Graph Neural Network, GNN）、PyTorch（深度学习框架）