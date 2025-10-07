# 媒体与认知 Lecture 4：特征工程 - II（4.pdf 讲义翻译）

## 第 1 页



*   **标题**：媒体与认知（Media and Cognition）

*   **副标题**：第 4 讲：特征工程 - II（Lecture 4: Feature Engineering-II）

*   **所属院系**：清华大学电子工程系（Dept. of EE, Tsinghua University）

*   **主讲人**：方璐（Lu FANG）

## 第 2 页



*   **页面主题**：回顾特征工程 I（Recall: Feature Engineering I (Lecture 2)）

*   **核心知识点 1**：特征需具备鲁棒性（robust），能应对数据变化（data variance），且泛化能力（generalize）良好，即面对新数据时仍能有效发挥作用。

*   **核心知识点 2**：特征之间应避免冗余（redundant），冗余特征会增加计算成本，且可能干扰模型学习，需保证各特征提供独特信息。

*   **核心知识点 3**：特征在不同类别（classes）间需具有区分性（discriminative），以便模型通过特征差异区分不同类别样本。

*   **示例**：花卉分类（Flower classification）任务中，可选择 “颜色（color）”“花瓣数量（number of petals）”“叶子形状（shape of leaf）” 作为特征。

*   **后续衔接**：基于选定的特征进行预测（Make prediction based on selected features）是第 4 讲的重点，将实现从特征（feature）到具体任务（task）的落地。

## 第 3 页



*   **页面主题**：美国爱荷华州埃姆斯市房价影响特征分析（What features may affect house price in City of Ames, Iowa, USA?）

*   **特征 1 与房价关系**：“浴室数量（Number of bathrooms）” 特征与房屋销售价格（sale price）呈正相关（positively correlated），即浴室数量越多，房价通常越高。

*   **特征 2 与房价关系**：“房屋年龄（Age of the house）” 特征与房屋销售价格呈负相关（negatively correlated），即房屋建成时间越久，房价往往越低。

*   **特征 3 与房价关系**：“房屋类型（New house or second-hand house，新房或二手房）” 特征与房屋销售价格相关（correlated），新房通常比同条件二手房价格更高。

## 第 4 页



*   **页面主题**：选定特征的应用场景（What can we do with the selected features?）

*   **应用场景 1**：分类（Classification），根据特征将样本划分到预设类别中，如邮件分类（垃圾邮件 / 正常邮件）。

*   **应用场景 2**：检测（Detection），从复杂环境中识别出特定目标并确定其位置，如人脸检测、目标检测。

*   **应用场景 3**：推荐（Recommendation），基于用户或物品特征为用户推荐符合需求的内容，如商品推荐、影视推荐。

*   **应用场景 4**：识别（Recognition），对样本进行身份或类别确认，如人脸识别、语音识别。

*   **示例**：基于面部图像（facial image）识别人物（identify specific persons）任务中，特征需对眼镜（glasses）、光线（lighting）、颜色（color）等干扰因素具备鲁棒性（robust），确保不同条件下仍能准确识别。

## 第 5 页



*   **页面主题**：特征工程 - II 内容框架（Feature Engineering - II）

*   **模块 1：线性回归（Linear Regression）**


    *   公式化（Formulation）：明确线性回归的数学表达形式，建立输入与输出的线性关系。
    
    *   最小二乘估计（LSE，Least Squared Error）：通过最小化平方误差构建目标函数，求解模型参数。
    
    *   梯度下降（Gradient Descent）：一种迭代优化算法，通过沿梯度方向更新参数，找到目标函数最小值。

*   **模块 2：二元分类（Binary Classification）**


    *   动机（Motivation）：说明为何需要二元分类，以及其在实际问题中的应用价值，如疾病诊断（患病 / 未患病）。
    
    *   支持向量机（SVM，Support Vector Machine）：一种经典分类算法，通过寻找最大间隔超平面实现分类。
    
    *   逻辑回归（Logistic Regression）：将线性回归输出映射到 \[0,1] 区间，用于预测二分类问题的概率。

*   **模块 3：多分类（Multi-class Classification）**


    *   动机（Motivation）：阐述现实问题中类别数量往往多于 2 个，需扩展分类能力，如手写数字识别（0-9 共 10 类）。
    
    *   软 max 回归（Softmax Regression，监督学习（supervised））：将逻辑回归推广到多分类场景，输出各类别概率。
    
    *   K 均值聚类（K-means Clustering，无监督学习（unsupervised））：无需标签，通过数据相似度将样本划分为多个聚类。

## 第 6 页



*   **页面主题**：线性回归基础（Linear Regression）

*   **变量定义**：设输入样本 $x^{(i)} \in \mathbb{R}^{d}$（$d$ 为特征维度），其第 $j$ 维特征表示为 $x_{j}^{(i)}$（$j=1,2,...,d$）。

*   **线性关系假设**：线性回归假设目标变量 $y$ 是输入 $x$ 的线性函数（linear function），数学表达式为：$y^{(i)}=h_{\theta}\left(x^{(i)}\right)=\sum_{j=1}^{d} \theta_{j} x_{j}^{(i)}=\theta^{T} x^{(i)}$，其中 $h_{\theta}(x^{(i)})$ 为假设函数（hypothesis function），$\theta \in \mathbb{R}^{d}$ 为模型参数（parameter）向量。

*   **映射关系**：线性回归实现从输入 $x^{(i)}$ 到输出 $y^{(i)}$ 的映射（Projecting from $x^{(i)}$ to $y^{(i)}$），通过参数 $\theta$ 控制映射规则。

*   **学习过程**：从训练数据（training data）中学习假设函数 $h(x)=\theta^{T} x^{(i)}$ 的参数 $\theta$，即找到最优 $\theta$ 使模型预测值尽可能接近真实值。

*   **预测过程**：利用训练得到的最优参数 $\theta^{*}$，对测试样本 $x_{test}$ 进行预测，得到预测结果 $y_{test }=\theta^{* T} x_{test }$。

## 第 7 页



*   **页面主题**：线性回归的目标函数（Linear Regression - Mean Squared Error）

*   **均方误差定义**：线性回归的目标是最小化均方误差（Mean Squared Error），损失函数（Loss Function）表达式为：$L(\theta)=\frac{1}{2} \sum_{i=1}^{n}\left[h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right]^{2}$，其中 $n$ 为训练样本数量，$h_{\theta}(x^{(i)})$ 为模型对第 $i$ 个样本的预测值，$y^{(i)}$ 为第 $i$ 个样本的真实值。

*   **损失函数意义**：该函数衡量模型预测值与真实值之间的平均平方偏差，值越小说明模型预测越准确。

*   **优化问题本质**：线性回归的参数求解本质是最小二乘估计（LSE，Least Squared Error）问题，即寻找参数 $\theta$ 使上述损失函数最小。

## 第 8-9 页



*   **页面主题**：最小二乘估计的闭式解（Closed-form Solution for LSE）

*   **矩阵表示**：将训练样本的输入和输出以矩阵形式表示，设 $X$ 为 $n \times d$ 的输入矩阵（每行对应一个样本的所有特征），$X=\left[\begin{array}{c}\left(x^{(1)}\right)^{T} \\ \left(x^{(2)}\right)^{T} \\ \vdots \\ \left(x^{(n)}\right)^{T}\end{array}\right]$；$Y$ 为 $n$ 维输出向量（每个元素对应一个样本的真实值），$Y=\left[\begin{array}{c}y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(n)}\end{array}\right]$。此时损失函数可表示为矩阵形式：$L(\theta)=\frac{1}{2}(X \theta-Y)^{T}(X \theta-Y)$。

*   **求导过程**：对损失函数 $L(\theta)$ 关于参数 $\theta$ 求梯度（Taking derivative of $L(\theta)$ w.r.t $\theta$）：

1.  展开损失函数：$\frac{1}{2}\left[(X\theta)^T X\theta - (X\theta)^T Y - Y^T (X\theta) + Y^T Y\right]$；

2.  利用矩阵求导公式（$\frac{\partial x^{T} a}{\partial x}=\frac{\partial a^{T} x}{\partial x}=a$，$\frac{\partial x^{T} A x}{\partial x}=A x+A^{T} x$），对展开式求导；

3.  化简后得到梯度：$\nabla _{\theta }L(\theta )=X^T X\theta - X^T Y$。

*   **闭式解求解**：令梯度等于 0（$\nabla _{\theta }L(\theta )=0$），求解方程 $X^T X\theta - X^T Y=0$，可得参数 $\theta$ 的闭式解（Closed-form Solution）：$\theta^{*}=(X^{T} X)^{-1} X^{T} Y$，其中 $\theta^{*}$ 是损失函数 $L(\theta)$ 的最小值点（minimizer）。

## 第 10-11 页



*   **页面主题**：从解析解到数值解（from Analytical Solution to Numerical）

*   **数值解的必要性**：当样本数量 $n$ 或特征维度 $d$ 极大时，计算矩阵 $X^T X$ 的逆矩阵（$(X^T X)^{-1}$）可能面临计算量过大、内存不足或矩阵不可逆等问题，此时解析解（Analytical Solution）不再适用，需采用数值解（Numerical Solution）。

*   **数值解的类比理解**：可将数值优化过程类比为 “在雾中下山（descending the mountain in the fog）”：


    *   无法获取全局地形信息（May not know the global info），即难以直接找到全局最优解；
    
    *   但可通过感知当前位置的坡度（梯度），向地势更低（损失更小）的方向移动，逐步找到附近的最优位置（find the optimal nearby is doable）。

*   **数值解核心思想**：通过逐步优化（Optimize step-by-step），不断调整参数以降低损失函数值，直到模型收敛（convergence，即参数变化量小于预设阈值，或损失函数值不再明显下降）。

## 第 12 页



*   **页面主题**：梯度下降算法（Gradient Descent）

*   **算法初始化**：随机选择一个初始参数 $\theta^0$（Start with some initial $\theta$），作为优化的起点。

*   **迭代更新规则**：在每一轮迭代中，按照以下公式更新参数：$\theta^{t+1}=\theta^{t}-\alpha \nabla_{\theta} L(\theta)$，其中 $t$ 为迭代次数，$\theta^t$ 为第 $t$ 轮迭代后的参数，$\theta^{t+1}$ 为第 $t+1$ 轮迭代后的参数，$\alpha$ 为学习率（learning rate），$\nabla_{\theta} L(\theta)$ 为损失函数在当前参数 $\theta^t$ 处的梯度。

*   **学习率的意义**：学习率 $\alpha$ 控制每一步迭代的参数更新幅度，它的取值对优化效果至关重要（Tuning the learning rate $\alpha$ can be difficult）：


*   若 $\alpha$ 过大（too big），参数更新幅度过大，可能导致损失函数震荡甚至发散（diverge），无法收敛到最优解；
    
    *   若 $\alpha$ 过小（too small），参数更新缓慢，迭代次数大幅增加，优化效率极低（optimization is too slow）。

*   **算法动机**：梯度（$\nabla_{\theta} L(\theta)$）指向损失函数增长最快的方向，因此沿梯度的反方向（减去梯度）更新参数，能使损失函数值以最快速度下降（steepest decrease of loss function $L(\theta)$），从而快速逼近最优解。

## 第 13 页



*   **页面主题**：梯度下降的学习率影响（Gradient Descent - Learning Rate Impact）

*   **算法流程补充**：梯度下降的完整流程为：初始化参数 $\theta^{t=0}$（Initilize $\theta^{t=0}$）；重复执行参数更新 $\theta ^{t+1}=\theta ^{t} - \alpha \nabla _{\theta }L(\theta )$（While not converged do）；当模型收敛后，返回最优参数 $\theta$（Return $\theta$）。

*   **不同学习率的效果**：

1.  学习率过大（$\alpha$ is too big）：参数更新幅度过大，会导致损失函数值在最优解附近剧烈震荡，甚至偏离最优解方向，最终优化无法收敛（optimization cannot diverge）；

2.  学习率过小（$\alpha$ is too small）：参数每次更新的变化量极小，需要大量迭代才能使损失函数接近最优值，优化速度极慢（optimization is too slow）；

3.  学习率适当（$\alpha$ is chosen properly）：参数更新幅度适中，损失函数值能平稳下降，最终收敛到最优解（Optimization can converge）。

## 第 14 页



*   **页面主题**：动态学习率（Dynamic Learning Rate）

*   **学习率调度策略**：为解决固定学习率难以适应整个优化过程的问题，引入动态学习率，即随着迭代次数调整学习率大小。常用的学习率调度（learning schedule）方式为：在每次迭代 $t$ 时，将学习率调整为 $\frac{a}{b+t}$，其中 $a$ 为初始学习率（initial learning rate），$b$ 为开始调整学习率的迭代次数（when the annealing begins）。

*   **常见学习率衰减策略**：

1.  线性衰减（Linear decay）：学习率随迭代轮次（epoch）线性降低，公式为：$Learning\ rate (epoch) = Initial\ learning\ rate - (Epoch \times Decrement\ per\ epoch)$，其中 “Decrement per epoch” 为每轮迭代的学习率衰减量；

2.  指数衰减（Exponential decay）：学习率在前期迭代中快速下降（decrease rapidly in the first few epochs），后期保持在较低值（spend more epochs with a lower value），且永远不会降至 0（never reach exactly zero），能在后期精细调整参数以逼近最优解。

## 第 15 页



*   **页面主题**：从梯度下降到随机梯度下降（from GD to Stochastic GD (SGD)）

*   **梯度下降（GD）的局限性**：传统梯度下降（GD）在每次迭代时，需使用全部训练数据集（full dataset）计算梯度（calculates gradient on full dataset）。当数据集规模庞大时，存在两个关键问题：

1.  计算成本高（loss computation over entire training set can be very slow）：遍历全部数据计算梯度耗时久，迭代效率低；

2.  内存占用大（using full training is intractable in terms of memory）：大规模数据集难以全部加载到内存中，导致计算无法进行。

*   **随机梯度下降（SGD）的改进**：随机梯度下降（Stochastic GD）每次迭代仅使用随机选择的一小部分数据（a random portion of (a batch of) data）计算梯度（calculate gradient on a random portion），有效解决上述问题。其参数更新公式与梯度下降一致：$\theta^{t+1}=\theta^{t}-\alpha \nabla_{\theta} L(\theta)$，但损失函数定义为批次内样本的平均损失：$L(\theta)=\frac{1}{N} \sum_{i=1}^{N} l_{i}$，其中 $l_{i}$ 为单个训练样本的损失（sample wise loss），$N$ 为随机选择的小批次数据的样本数量（batchsize of the randomly chosen mini-batch）。

## 第 16 页



*   **页面主题**：随机梯度下降与梯度下降的收敛对比（SGD vs GD - Convergence Comparison）

*   **可视化对比**：通过图表展示两种算法的收敛过程，其中蓝色曲线（Blue: full）代表标准梯度下降（standard Gradient decent method），每次迭代使用全部训练数据（uses entire training data as a batch）；红色曲线（Red: batch）代表随机梯度下降（SGD method），每次迭代仅使用随机打乱（randomly shuffled）后的小批次数据（mini-batches (batchsize N) data）。

*   **收敛效率结论**：在该示例中，随机梯度下降达到收敛状态所需的迭代步骤（steps）更少（SGD requires fewer steps to converge），即收敛速度比使用全部训练数据的标准梯度下降更快（faster convergence than standard gradient decent with full training data）。这是因为 SGD 每次迭代计算量小，能快速更新参数，即使单个批次的梯度存在噪声，长期来看仍能向最优解方向移动。

## 第 17 页



*   **页面主题**：不同批次大小的训练损失对比（Training loss vs iteration - Batch Size Impact）

*   **图表解读**：横坐标为迭代次数（iteration），纵坐标为训练损失（training loss）。图中包含四条曲线，分别对应不同的批次大小（batchsize）：

1.  “full”：批次大小等于整个训练数据集（batchsize is the whole training dataset），即标准梯度下降；

2.  “Stochastic”：批次大小为 1（batchsize is 1），即最基础的随机梯度下降；

3.  “Mini-batch,b=10”：批次大小为 10 的小批量梯度下降；

4.  “Mini-batch,b=100”：批次大小为 100 的小批量梯度下降。

*   **批次大小影响**：随着批次大小增大，损失曲线更接近 “full” 曲线（larger batchsize better approximates the full training dataset），即损失下降过程更平稳，更能反映整个训练集的损失变化趋势。但批次过大也会导致每次迭代计算量增加，迭代速度变慢，需在收敛稳定性和迭代效率间权衡。

## 第 18 页



*   **页面主题**：梯度下降与随机梯度下降的全局最优性（Could GD/SGD always find the global optima?）

*   **线性回归中的最优性**：在线性回归问题中，损失函数（LSE 问题）是凸函数（convex）。凸函数的特点是局部最小值（local minimal）与全局最小值（global minimal）完全相同（local minimal = global minimal），即整个函数空间中只有一个最低位置，因此梯度下降和随机梯度下降只要迭代足够多次，最终都能找到全局最优解。

*   **非线性场景的挑战**：但在实际应用中，很多任务的回归或分类模型并非线性的（regression is not always in linear space），其损失函数往往是非线性（nonlinear）且复杂的（complex loss function）。对于这类损失函数，函数空间中可能存在多个局部最小值，梯度下降和随机梯度下降可能陷入局部最小值（local minimum），无法找到全局最优解。

## 第 19 页



*   **页面主题**：跳出局部最小值的方法（How to jump out of local minimum?）

*   **方法 1：动量（Momentum）**：为参数更新过程引入 “动量”，模拟物理中的惯性概念，为损失函数在特定方向提供基础冲量（basic impulse）。当模型接近局部最小值时，动量能帮助参数 “冲过” 狭窄或较浅的局部最小值区域（avoid narrow or small local minima），继续向全局最优解方向移动。

*   **方法 2：梯度的噪声估计（Noisy estimate of the gradient）**：不使用精确的梯度值（not the exact gradient value），而是在梯度计算中引入一定噪声。这种噪声使模型有机会探索（explore the searching space）损失函数空间中的其他区域，可能发现更优的参数位置，从而跳出局部最小值。

*   **方法 3：批次数据集的随机性（Batched dataset also works）**：随机梯度下降中，每个批次的数据（each batch of data）仅能提供真实全局梯度的一个估计值（estimation for the true global gradient）。这种估计的不准确性（inaccuracy）恰好起到了探索参数空间的作用（works as the space exploration），帮助模型避免一直停留在局部最小值附近。

## 第 20 页



*   **页面主题**：实用优化器及其性能（Practical optimizers & their performance）

*   **常见优化器列举**：

1.  梯度下降（GD - Gradient Descent）：最基础的优化器，作为性能对比的基准（baseline）；

2.  带动量的梯度下降（Momentum - Gradient Descent with momentum）：引入动量机制，加速收敛并减少震荡；

3.  Nesterov 加速梯度（Nesterov - Nesterov's Accelerated Gradient）：在动量基础上改进，先根据历史动量更新参数，再计算梯度，进一步提升收敛速度；

4.  同时扰动随机近似（Spsa - Simultaneous Perturbation Stochastic Approximation）：通过随机扰动参数计算梯度，适用于高维或难以直接计算梯度的场景；

5.  自适应动量估计（Adam - Adaptive Moment Estimation）：结合动量和自适应学习率，能为不同参数调整合适的学习率，在多种任务中表现优异。

*   **性能可视化方式**：将优化目标函数以热力图（heat map）形式呈现，颜色越浅（或温度越低）代表损失函数值越小。各类优化器的性能通过其在热力图中寻找 “最冷点”（最小损失值）的过程和效率来体现，优秀的优化器能更快找到并接近最冷点。

## 第 21 页



*   **页面主题**：二元分类任务示例（Binary Classification - Examples）

*   **示例 1：花卉分类**：区分茉莉花类（Jasmine Class）和荷花类（Lotus Class），需找到合适特征（如花瓣形状、花茎长度等）构建模型，实现两类花卉的准确分类。

*   **示例 2：邮件分类**：区分个人邮件（Personal Class）和垃圾邮件（Spam Class），可基于邮件关键词、发送地址、内容长度等特征，判断邮件类别，过滤垃圾邮件。

*   **二元分类核心问题**：如何设计或选择特征，以及构建分类模型，实现对两个不同类别样本的有效区分（How to distinguish these two types of flowers? / How to classify personal emails and spam emails?）。

## 第 22 页



*   **页面主题**：二元分类的样本表示与最优边界选择（Binary Classification - Sample Representation & Optimal Boundary）

*   **样本表示**：设训练样本集为 $\{(x^{(i)}, y^{(i)})\} (i=1, ..., n)$，其中 $n$ 为样本总数（total number of training samples），$x^{(i)}$ 为第 $i$ 个样本的特征向量，$y^{(i)} \in \{-1,1\}$ 为样本的类别标签（class labels），分别代表两个不同类别。

*   **分类边界选择问题**：给定两个类别的样本，存在多个可能的线性边界（linear separator）将其分开，但并非所有边界都是最优的。最优边界应具备良好的泛化能力，即对新样本的分类准确率最高。从几何角度看，最优边界通常是 “最大间隔边界”（maximum margin boundary），即边界到两个类别中最近样本的距离（margin）最大，这种边界对噪声样本更鲁棒，泛化能力更强。

## 第 23 页



*   **页面主题**：支持向量机基础（Support Vector Machine (SVM) - Basics）

*   **SVM 核心定义**：支持向量机（SVM）是一种经典的二元分类算法，其核心是寻找具有最大间隔（largest margin）的线性分离器（linear separator），以实现最优分类效果。

*   **决策边界数学表达**：设决策边界为超平面（在二维空间中为直线），其数学表达式为 $w^{T} x^{(i)} + b = 0$，其中：


*   $w^{T}$ 是与决策边界正交的向量（a vector orthogonal to an arbitrary decision boundary），控制边界的方向；
    
    *   $b$ 是偏移项（a scaler “offset” term），控制边界在空间中的位置；
    
    *   $x^{(i)}$ 是落在决策边界上的样本点，满足该方程。

*   **支持向量（Support Vectors）**：在两个类别中，距离决策边界最近的样本点称为支持向量，这些点决定了决策边界的位置和最大间隔的大小。如页面图示中的 3 个样本点（these 3 points are support vectors），移除非支持向量样本不会改变决策边界，支持向量是 SVM 的核心样本。

## 第 24 页



*   **页面主题**：支持向量机的几何间隔（Support Vector Machine (SVM) - Geometric Margin）

*   **几何间隔定义**：几何间隔（Geometric margins $\gamma^{(i)}$）指样本点 $x^{(i)}$ 到决策边界的垂直距离（a point $x^{(i)}$'s distance to decision boundary），如页面中线段 AB 的长度（the line segment AB）。

*   **间隔最大化目标**：SVM 的目标是最大化几何间隔，且重点关注所有样本中最小的几何间隔（$\gamma=min \gamma^{(i)}, \forall i$），即确保决策边界到两个类别中最近样本的距离最大（maximum $\gamma^{(i)}$ for nearest point to the boundary）。这种 “最大化最小间隔” 的策略能提升模型的鲁棒性（safer），减少噪声样本对分类结果的影响。此时，模型的总间隔（Margin）为 $2\gamma$（最小几何间隔的两倍）。

*   **几何间隔与决策边界的关系**：设样本点 $A$ 为 $x^{(i)}$，点 $B$ 为样本点 $A$ 在决策边界上的投影，则点 $B$ 的坐标可表示为 $x^{(i)}-\gamma^{(i)} \cdot \frac{w}{\|w\|}$（其中 $\|w\|$ 为向量 $w$ 的模）。由于点 $B$ 在决策边界上，满足 $w^{T} x + b = 0$，代入点 $B$ 坐标可得 $w^{T}\left(x^{(i)}-\gamma^{(i)} \cdot \frac{w}{\| w\| }\right)+b=0$，为后续推导几何间隔公式奠定基础。

## 第 25 页



*   **页面主题**：支持向量机的几何间隔公式与约束条件（Support Vector Machine (SVM) - Geometric Margin Formula & Constraints）

*   **几何间隔公式推导**：基于上一页中样本点投影与决策边界的关系，展开方程 $w^{T}\left(x^{(i)}-\gamma^{(i)} \cdot \frac{w}{\| w\| }\right)+b=0$，化简后可得几何间隔的计算公式：$\gamma^{(i)}=\frac{w^{T} x^{(i)}+b}{\| w\| }$。该公式也可表示为 $\left(\frac{w}{\|w\| }\right)^{T} x^{(i)}+\frac{b}{\|w\| }$，即归一化参数后的线性函数输出。

*   **类别约束条件**：根据样本类别标签 $y^{(i)}$，几何间隔需满足以下约束：

1.  当 $y^{(i)}=1$（正类）时，样本点应在决策边界的正侧，几何间隔大于等于最小间隔 $\gamma$，即 $\gamma^{(i)}=\frac{w^{T} x^{(i)}+b}{\| w\| } \geq \gamma$；

2.  当 $y^{(i)}=-1$（负类）时，样本点应在决策边界的负侧，几何间隔小于等于 $-\gamma$，即 $\gamma^{(i)}=\frac{w^{T} x^{(i)}+b}{\| w\| } \leq -\gamma$。

*   **统一约束形式**：结合类别标签，可将上述约束统一表示为 $\gamma^{(i)}=y^{(i)}\left[\left(\frac{w}{\| w\| }\right)^{T} x^{(i)}+\frac{b}{\| w\| }\right] \geq \gamma$。进一步令 $\frac{w}{\|w\| \gamma}$ 和 $\frac{b}{\|w\| \gamma}$ 为新的参数（由于参数缩放不影响决策边界的位置和方向），可将约束简化为 $y^{(i)}\left(w^{T} x^{(i)}+b\right) \geq 1$，为后续优化问题构建奠定基础。

## 第 26 页



*   **页面主题**：支持向量机的优化目标转化（Support Vector Machine (SVM) - Optimization Objective Transformation）

*   **间隔与参数的关系**：由前文可知，模型总间隔（Margin）为 $2\gamma$，且 $\gamma=\frac{1}{\|w\|}$（基于约束 $y^{(i)}\left(w^{T} x^{(i)}+b\right) \geq 1$ 推导）。因此，最大化总间隔 $2\gamma$ 等价于最大化 $\frac{2}{\|w\|}$。

*   **优化目标转化**：由于最大化 $\frac{2}{\|w\|}$ 与最小化 $\frac{1}{2}\|w\|_2^2$（$\|w\|_2$ 为向量 $w$ 的 2 - 范数）在数学上等价（且最小化二次函数更易求解），因此 SVM 的优化目标可转化为：在约束条件 $y^{(i)}\left(w^{T} x^{(i)}+b\right) \geq 1$（$i=1, ..., n$）下，最小化 $\frac{1}{2}\|w\|_2^2$。

*   **求解方法提示**：该优化问题属于带约束的凸二次规划问题，可通过拉格朗日对偶（Lagrangian duality）方法求解，将原始问题（Primal Problem）转化为对偶问题（Dual Problem），降低求解复杂度。

## 第 27 页



*   **页面主题**：支持向量机的原始问题与对偶问题（Support Vector Machine (SVM) - Primal & Dual Problems）

*   **原始问题（Primal Problem）**：


*   目标函数：$min _{w, b} \frac{1}{2}\| w\| _{2}^{2}$（最小化参数向量 $w$ 的 2 - 范数平方的一半）；
    
*   约束条件：$y^{(i)}\left(w^{T} x^{(i)}+b\right) \geq 1$（$i=1, ..., n$），确保所有样本都在决策边界的正确一侧，且满足最小间隔要求。

*   **对偶问题（Dual Problem）**：通过引入拉格朗日乘子 $\alpha_i \geq 0$（$i=1, ..., n$），构建拉格朗日函数并转化为对偶问题：


*   目标函数：$max _{\alpha } L(\alpha )=\sum _{i=1}^{n} \alpha _{i}-\frac {1}{2}\sum _{i,j=1}^{n}y^{(i)} y^{(j)} \alpha _{i} \alpha _{j}(x^{(i)})^{T} x^{(j)}$（最大化拉格朗日函数关于 $\alpha$ 的值）；
    
    *   约束条件：$\sum_{i=1}^{n} \alpha_{i} y^{(i)}=0$（$i=1, ..., n$）和 $\alpha_{i} \geq 0$（$i=1, ..., n$），确保对偶问题与原始问题等价。

*   **对偶问题的意义**：对偶问题将原始问题中对参数 $w$ 和 $b$ 的优化，转化为对拉格朗日乘子 $\alpha$ 的优化，且目标函数中仅包含样本间的内积（$(x^{(i)})^{T} x^{(j)}$），为后续引入核技巧（Kernel Trick）奠定基础。

## 第 28 页



*   **页面主题**：支持向量机的对偶问题求解思路（Support Vector Machine (SVM) - Dual Problem Solving Idea）

*   **二次规划求解的局限性**：对偶问题虽为凸二次规划问题，但当样本数量 $n$ 较大时，直接使用传统二次规划（QP，Quadratic Programming）方法求解 $n$ 个拉格朗日乘子 $\alpha_i$，会面临计算量过大、效率低下的问题（Why not directly solving it via quadratic programming (QP)?）。

*   **SMO 算法引入**：为解决大规模样本下的求解问题，引入序列最小优化（SMO，Sequential Minimal Optimization）算法，该算法通过迭代方式逐步求解 $\alpha_i^*$（最优拉格朗日乘子）。

*   **SMO 核心思想**：每次仅选择一对拉格朗日乘子（$\alpha_i$ 和 $\alpha_j$）进行更新，同时固定其他所有 $\alpha_k$（$k \neq i,j$）。通过构建这对乘子之间的关系，将问题转化为单变量优化问题，可快速求解，重复该过程直到收敛。这种 “分而治之” 的策略大幅降低了计算复杂度，使 SVM 能处理大规模数据集。

## 第 29 页



*   **页面主题**：SMO 算法概念详解（1/3）（A brief introduction to SMO concept (1/3)）

*   **算法核心挑战**：SMO 算法需求解 $n$ 个拉格朗日乘子 $\alpha_i$，但由于存在约束条件 $\sum_{i=1}^{n} \alpha_{i} y^{(i)}=0$，单个乘子的取值依赖于其他乘子，无法独立求解（construct relations between all $\alpha$ find the exact value for one $\alpha \to$ all $n$ $\alpha$ values determined）。

*   **算法核心思想**：通过迭代构建成对乘子（$\alpha_i$ 和 $\alpha_j$）之间的关系，每次仅优化一对乘子，利用约束条件消去一个变量，将问题转化为单变量二次规划问题求解，逐步逼近所有乘子的最优值（construct the relations between pairs of $\alpha$ iteratively, solve the last $\alpha$ using quadratic programming）。

*   **迭代步骤**：

1.  选择待更新的乘子对（$\alpha_i$ 和 $\alpha_j$）：采用启发式策略（heuristic），优先选择能使目标函数产生最大变化的乘子对，以加速收敛（pick the two that will allow us to make the biggest progress towards the global maximum）；

2.  固定其他乘子优化目标函数：在保持所有其他 $\alpha_k$（$k \neq i,j$）固定的情况下，仅针对 $\alpha_i$ 和 $\alpha_j$ 重新优化拉格朗日函数 $L(\alpha)$；

3.  重复迭代：不断执行步骤 1 和步骤 2，直到目标函数收敛或满足预设精度要求。

*   **选择乘子对的原因**：若仅修改单个乘子 $\alpha_i$，无法满足约束条件 $\sum_{i=1}^{n} \alpha_{i} y^{(i)}=0$（Cannot satisfy the $\alpha$ constraint only modifying one $\alpha$ component），因此必须成对更新乘子。

## 第 30 页



*   **页面主题**：SMO 算法概念详解（2/3）（A brief introduction to SMO concept (2/3)）

*   **约束条件转化（以 **** **** 和 **** **** 为例）**：选取 $\alpha_1$ 和 $\alpha_2$ 作为待更新的乘子对，根据约束条件 $\sum_{i=1}^{n} \alpha_{i} y^{(i)}=0$，可将其转化为 $\alpha_1 y^{(1)}+\alpha_2 y^{(2)}=-\sum_{i=3}^{n} \alpha_{i} y^{(i)}$。由于在本次迭代中仅优化 $\alpha_1$ 和 $\alpha_2$，其他乘子 $\alpha_3, ..., \alpha_n$ 固定，因此等式右侧可视为常数 $\zeta$（$\alpha_1 y^{(1)}+\alpha_2 y^{(2)}=\zeta$）。

*   **变量消去**：利用类别标签 $y^{(i)} \in \{-1,1\}$ 的性质（$(y^{(1)})^2=1$），可将 $\alpha_1$ 表示为 $\alpha_2$ 的函数：$\alpha_1=\left(\zeta-\alpha_2 y^{(2)}\right) y^{(1)}$。

*   **目标函数转化**：将 $\alpha_1=\left(\zeta-\alpha_2 y^{(2)}\right) y^{(1)}$ 代入拉格朗日函数 $L(\alpha)$，此时目标函数 $L(\alpha_{1},\alpha_{2},... ,\alpha_{i},... \alpha_{n})$ 转化为仅包含 $\alpha_2$ 和固定乘子（$\alpha_3, ..., \alpha_n$）的函数 $L(\alpha_{2},... ,\alpha_{i},... \alpha_{n})$，实现了从双变量到单变量的简化。

## 第 31 页



*   **页面主题**：SMO 算法概念详解（3/3）（A brief introduction to SMO concept (3/3)）

*   **目标函数进一步简化**：通过持续迭代选择不同乘子对，重复 “变量消去 - 目标函数简化” 的过程，最终可将拉格朗日函数 $L(\alpha_{1},\alpha_{2},... ,\alpha_{i},... \alpha_{n})$ 简化为仅包含最后一个乘子 $\alpha_n$ 的函数 $L(\alpha_n)$（$L(\alpha _{1},\alpha _{2},... ,\alpha _{i},... \alpha _{n})= ... =L(\alpha _{n})$）。

*   **单变量二次规划求解**：此时，目标函数成为关于 $\alpha_n$ 的二次函数（形式为 $A \alpha_n^2 + B \alpha_n + C$，其中 $A、B、C$ 为已知常数），可通过简单的二次规划（QP）方法精确求解 $\alpha_n$ 的最优值（use the simple quadratic programming (QP) to solve the exact value of $\alpha_n$）。

*   **乘子值恢复**：求解出 $\alpha_n$ 后，利用之前迭代过程中建立的乘子对关系（如 $\alpha_1$ 与 $\alpha_2$、$\alpha_2$ 与 $\alpha_3$ 等的关系），反向推导并恢复所有拉格朗日乘子 $\alpha_i$ 的最优值。

*   **乘子约束修正**：若通过二次规划求解得到的乘子 $\alpha_i$ 小于 0（违反 $\alpha_i \geq 0$ 的约束），则需对其进行裁剪（clip），将其设置为 0（$\alpha_{i}^{new }=0$）；若 $\alpha_i \geq 0$，则保持求解结果不变（$\alpha_{i}^{new }=\alpha_{i}^{new,uncliped }$），确保所有乘子满足约束条件。

*   **扩展学习提示**：如需了解更详细的 SMO 算法推导，可参考斯坦福大学相关课程资料（[https://see.stanford.edu/materials/aimlcs229/cs229-notes3.pdf](https://see.stanford.edu/materials/aimlcs229/cs229-notes3.pdf)）。

## 第 32 页



*   **页面主题**：支持向量的判定与性质（Support Vector Machine (SVM) - Support Vector Identification & Properties）

*   **支持向量的判定标准**：在 SVM 的最优解中，支持向量（Support Vectors）是指拉格朗日乘子 $\alpha_i^* \neq 0$ 的样本点 $x^{(i)}$（Support vectors are those points $x^{(i)}$ for which $\alpha_i^* \neq 0$）。

*   **支持向量的稀疏性**：在实际求解结果中，仅有少数样本点的 $\alpha_i^*$ 非零（only a few $\alpha_i^*$ can be nonzero），大部分样本点的 $\alpha_i^* = 0$。例如页面示例中，仅 $\alpha_1^*$、$\alpha_7^*$、$\alpha_8^*$ 非零（$\alpha_7^* = 0.6$、$\alpha_8^* = 1.4$、$\alpha_2^* = 0.8$），其余 $\alpha_i^*$ 均为 0，体现了支持向量的稀疏性。

*   **支持向量与约束条件的关系**：对于支持向量（$\alpha_i^* \neq 0$），根据 KKT 条件（Karush-Kuhn-Tucker Conditions），其必然满足 $y^{(i)}\left(w^{T} x^{(i)}+b\right) = 1$，即这些样本点恰好落在最大间隔边界上（如页面图示中满足 $w^T x + b = 1$ 或 $w^T x + b = -1$ 的样本点）；对于非支持向量（$\alpha_i^* = 0$），则满足 $y^{(i)}\left(w^{T} x^{(i)}+b\right) > 1$，即样本点在最大间隔边界内侧，对决策边界的位置无影响。

## 第 33 页



* **页面主题**：支持向量机的参数求解与预测（Support Vector Machine (SVM) - Parameter Calculation & Prediction）

* **最优参数 **** **** 的求解**：根据 SVM 对偶问题的最优解 $\alpha_i^*$，可通过以下公式计算原始问题中参数 $w$ 的最优值 $w^*$：$w^{*}=\sum_{i=1}^{n} \alpha_{i}^{*} y^{(i)} x^{(i)}$。该公式表明，$w^*$ 是支持向量（仅 $\alpha_i^* \neq 0$ 的样本）的线性组合，非支持向量由于 $\alpha_i^* = 0$，对 $w^*$ 无贡献，进一步体现了支持向量的核心作用。

* **KKT 条件与最优参数 **** **** 的求解**：利用原始问题与对偶问题等价性的 KKT 条件（Karush-Kuhn-Tucker Conditions），该条件包含三个关键约束：

  对于支持向量（$\alpha_i^* \neq 0$），根据互补松弛条件，必有 $y^{(i)}\left(w^{T} x^{(i)}+b\right)-1 = 0$，因此可通过支持向量样本求解 $b^*$：$b^{*}=y^{(j)}-\sum _{i=1}^{n}\alpha _{i}^{*}y^{(i)}(x^{(i)})^{T}x^{(j)}$，其中 $j$ 为任意支持向量的索引。

1.  $\alpha_{i} \geq 0$（拉格朗日乘子非负）；

2.  $y^{(i)}\left(w^{T} x^{(i)}+b\right)-1 \geq 0$（样本满足间隔约束）；

3.  $\alpha_{i}\left[y^{(i)}\left(w^{T} x^{(i)}+b\right)-1\right]=0$（互补松弛条件）。

*   **测试样本预测**：对测试样本 $x_{test}$ 进行分类预测时，首先计算预测函数 $h(x_{test})={w^{*}}^{T}x_{test}+b^{*}$。由于 $w^*$ 是支持向量的线性组合，可将预测函数进一步展开为 $h(x_{test})=\sum _{i\in SV}\alpha _{i}^{*} y^{(i)}(x^{(i)})^{T}x_{test}+b^{*}$，其中 $i \in SV$ 表示仅考虑 $\alpha_i^* \neq 0$ 的支持向量样本。最终分类规则为：若 $h(x_{test})>0$，则将测试样本分类为正类；否则分类为负类。

## 第 34 页



*   **页面主题**：支持向量机的核心解读（Interpretation of SVM）

*   **解读 1：最优参数的稀疏性**：最优参数 $w^*$ 是少量支持向量样本的线性组合（The optimal $w^*$ is a linear combination of a small number of data points），非支持向量样本对 $w^*$ 无贡献。这种稀疏性不仅降低了模型的存储成本（只需保存支持向量），还减少了预测时的计算量（仅需计算测试样本与支持向量的内积）。

*   **解读 2：核技巧的适配性**：无论是求解拉格朗日乘子 $\alpha_i^*$，还是进行预测，SVM 的核心计算均围绕样本间的内积（$(x^{(i)})^{T} x^{(j)}$）展开。这一特性使 SVM 能够自然地引入核技巧（kernel），通过将内积替换为核函数 $K(x^{(i)}, x^{(j)})$，实现对非线性问题的处理，无需显式将样本映射到高维空间。

*   **解读 3：预测的支持向量依赖性**：对新测试样本 $x_{test}$ 的分类决策，仅依赖于该样本与支持向量的相似度（通过内积或核函数衡量），而非所有训练样本。预测公式可简化为 $\hat{y}_{test }=sign\left(\sum_{i \in SV} \alpha_{i}^{*} y^{(i)}\left(x^{(i)}\right)^{T} x_{test }+b\right)$，其中 $sign(\cdot)$ 为符号函数，进一步体现了支持向量在 SVM 中的核心地位。

## 第 35 页



*   **页面主题**：核技巧：从线性到非线性（Kernel Trick: Linear -> Non-linear）

*   **核技巧的引入背景**：当训练样本在原始特征空间中无法通过线性边界分离时（即非线性问题），传统线性 SVM 无法有效分类。核技巧通过将样本映射到高维特征空间（high dimensional space），使原本非线性可分的样本在高维空间中线性可分，从而扩展 SVM 的适用范围。

*   **核函数的定义**：回顾 SVM 对偶问题的目标函数 $L(w, b, \alpha)=\sum_{i=1}^{n} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{n} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}\left(x^{(i)}\right)^{T} x^{(j)}$，其中 $\left(x^{(i)}\right)^{T} x^{(j)}$ 是原始特征空间中样本 $x^{(i)}$ 与 $x^{(j)}$ 的内积。核函数 $K(x^{(i)}, x^{(j)})$ 定义为高维特征空间中样本映射后的内积，即 $K\left(x^{(i)}, x^{(j)}\right)=\phi\left(x^{(i)}\right)^{T} \phi\left(x^{(j)}\right)$，其中 $\phi(\cdot)$ 是从原始空间到高维空间的映射函数。

*   **核技巧的优势**：无需显式计算映射函数 $\phi(\cdot)$ 和高维特征向量（计算成本极高），只需直接计算核函数 $K(x^{(i)}, x^{(j)})$，即可间接实现高维空间中的内积运算，大幅降低计算复杂度。

*   **示例代码提示**：在实际应用中，可通过调用机器学习库（如 Scikit-learn）实现带核函数的 SVM，例如使用径向基函数（RBF）核的代码为：`SVC(kernel='rbf', gamma=0.01), fit(X,y)`，其中 `gamma` 为核函数的超参数，控制核函数的影响范围。

## 第 36 页



*   **页面主题**：核技巧的几何意义（Kernel Trick: Linear -> Non-linear - Geometric Meaning）

*   **核心思想可视化**：当原始特征空间（original x-space）中的样本无法通过直线（线性边界）分离时（When a line is not enough...），核技巧通过将样本映射到高维特征空间（feature d-space），使样本在高维空间中可通过平面（线性边界）分离；而该高维空间中的线性边界，对应原始空间中的曲线（非线性边界），二者本质上是同一分类规则的不同空间表达（the curve \[in original space] is the same as the plane \[in feature space]）。

*   **示例图示解读**：页面图示展示了二维原始空间中非线性可分的样本（如点 (0,3)、(1,2)、(2,1)、(3,0)），通过核函数映射到高维空间后，可找到线性边界将其分离；该线性边界在原始空间中表现为曲线，实现了对非线性样本的分类。

*   **核技巧的关键价值**：核技巧无需改变 SVM 的核心算法逻辑，仅通过替换内积运算，即可让线性 SVM 处理非线性问题，兼顾了模型的简洁性和处理复杂数据的能力。

## 第 37 页



*   **页面主题**：核函数选择 quiz 与类型（Kernel Trick Quiz & Kernel Types）

*   **核函数选择 quiz**：页面通过表格展示了原始空间中 4 个样本点（(0,3)、(1,2)、(2,1)、(3,0)）在不同函数下的输出值，需判断哪个函数可作为核函数实现样本分离：


    *   “x+y” 函数：所有样本输出均为 3，无法区分样本，不能作为核函数；
    
    *   “xy” 函数：样本输出为 0、2、2、0，可初步区分样本，但区分能力有限；
    
    *   “x²” 函数：样本输出为 0、1、4、9，能清晰区分不同样本，更适合作为核函数。

*   **常见核函数类型**：

1.  线性核（Linear Kernel）：$K(s, t) = s^T t$，本质是原始空间的内积，对应线性 SVM，适用于线性可分问题；

2.  高斯核（Gaussian Kernel，又称径向基函数 RBF 核）：$K(s, t)=exp \left(-\frac{\| s-t\| ^{2}}{2 \sigma^{2}}\right)$，其中 $\sigma$ 为带宽参数，控制核函数的局部性，适用于非线性问题，灵活性高；

3.  多项式核（Polynomial Kernel）：$K(s, t)=\left(s^{T} t + c\right)^d$（页面简化为 $K(s, t)=\left(s^{T} t\right)^d$），其中 $d$ 为多项式次数，$c$ 为常数项，适用于样本具有多项式分布规律的问题；

4.  Sigmoid 核（Sigmoid Kernel）：$K(s, t)=tanh \left(\eta s^{T} t+v\right)$，其中 $\eta$ 和 $v$ 为超参数，可模拟神经网络的激活函数，适用于部分非线性问题。

## 第 38 页



* **页面主题**：核函数的数学推导（Kernel Trick: Linear -> Non-linear - Mathematical Derivation）

* **多项式核的推导示例**：以二次多项式核（$K(s, t) = (s^T t)^2$）为例，展示核函数与高维映射的关系。设样本 $s = (s_1, s_2, s_3)$、$t = (t_1, t_2, t_3)$（三维原始空间），则：

  $K(s, t) = (s^T t)^2 = \left(\sum_{i=1}^{3} s_i t_i\right)\left(\sum_{j=1}^{3} s_j t_j\right) = \sum_{i=1}^{3} \sum_{j=1}^{3} s_i s_j t_i t_j$。

  若定义高维映射函数 $\phi(x)$ 为原始特征的所有二次组合（如 $x_1x_1, x_1x_2, x_1x_3, x_2x_1, x_2x_2, x_2x_3, x_3x_1, x_3x_2, x_3x_3$），则 $\phi(s)^T \phi(t) = \sum_{i=1}^{3} \sum_{j=1}^{3} s_i s_j t_i t_j = K(s, t)$，即多项式核函数等价于高维空间中映射向量的内积。

* **核函数的核心要求**：核函数需满足 Mercer 条件（即核矩阵半正定），以确保存在对应的高维映射空间。其本质是无需显式计算高维映射向量（计算成本高），通过核函数直接获取高维空间的内积结果，实现 “隐式高维映射”。

* **核函数的设计目标**：核函数需能反映样本间的相似度（how similar $s$ and $t$ are），使相似样本在高维空间中距离较近，不相似样本距离较远，从而帮助 SVM 找到最优分类边界。

## 第 39 页



*   **页面主题**：多分类问题的引入（Multi-class Classification - Introduction）

*   **多分类的必要性**：现实世界中，分类任务往往不止两个类别（We are not living in a world of ‘Binary opposition’），二分类（2-class）无法满足实际需求（binary classification is far from enough）。例如，图像识别中需区分 “猫”“狗”“汽车” 等多个类别，手写数字识别需区分 0-9 共 10 个类别。

*   **多分类的核心问题**：如何将二分类模型的思想扩展到多分类场景（How to migrate from 2-class to multi-class?），实现对 $K$ 个类别（$K \geq 2$）的准确分类。

*   **ImageNet 数据集示例**：ImageNet 数据集包含 14,197,122 张图像，涵盖 21,841 个类别（synsets indexed），且常用子集包含超过 1000 个类别（more than 1000 categories），是典型的多分类任务数据集，对模型的多分类能力提出了极高要求。

*   **多分类的应用场景**：除图像分类外，多分类还广泛应用于目标检测（object detection，目标级分类）、语义分割（semantic segmentation，像素级分类）、动作识别（action recognition，时空联合分类）、人脸识别（facial recognition，大规模数据库中的分类与匹配）等复杂任务。

## 第 40 页



*   **页面主题**：多分类任务的核心工具（Multi-class Tasks - Core Tool）

*   **Softmax 的必要性**：前文提到的目标检测（30.1 FPS，每秒帧率）、语义分割（pixelwise classification）、动作识别（temporal and spatial classification）、人脸识别（classification & matching in large database）等复杂多分类任务，均需要 Softmax 函数作为核心工具（need softmax!）。

*   **Softmax 与逻辑回归的关系**：Softmax 回归是二分类逻辑回归（binary logistic regression）的推广形式（generalization form）。逻辑回归通过 Sigmoid 函数将线性输出映射到 \[0,1] 区间，输出二分类概率；Softmax 回归则通过 Softmax 函数将多维度线性输出映射到 \[0,1] 区间，且所有类别概率之和为 1，输出多分类概率。

*   **Softmax 函数公式**：对于 $K$ 分类任务，设模型对第 $j$ 个类别的线性输出为 $z_j$，则 Softmax 函数输出该类别的概率为：$\sigma(z)_{j}=\frac{e^{z_{j}}}{\sum_{k=1}^{K} e^{z_{k}}}$（$j=1, ..., K$）。其中，分子 $e^{z_j}$ 确保概率非负，分母 $\sum_{k=1}^{K} e^{z_k}$ 实现概率归一化，使所有类别概率之和为 1。

## 第 41 页



*   **页面主题**：二元逻辑回归的框架（Binary Logistic Regression - Framework）

*   **样本与标签定义**：设 $y^{(i)}$ 表示第 $i$ 个样本 $x^{(i)}$ 的类别标签（$y^{(i)}$ describes the class of sample $x^{(i)}$），在二分类任务中，$y^{(i)} \in \{0, 1\}$，其中 $y^{(i)}=1$ 表示样本属于正类（如茉莉花类 Jasmine Class），$y^{(i)}=0$ 表示样本属于负类（如荷花类 Lotus Class）。

*   **训练 / 学习阶段（Training/Learning）**：


*   已知信息：输入特征 $x^{(i)}$（Input, known）、样本标签 $y^{(i)}$（Target/Label, known）；
    
*   未知信息：模型参数 $\theta$（unknown, to learn!）；
    
    *   核心任务：通过训练数据学习参数 $\theta$，构建假设函数 $h_{\theta}(x)$（Hypothesis function），使假设函数能准确映射输入到类别标签。

*   **测试 / 预测阶段（Test/Prediction）**：


*   已知信息：测试样本输入 $x^{(i)}$、训练得到的参数 $\theta$（estimated during training）；
    
    *   核心任务：将测试样本输入代入假设函数 $h_{\theta}(x)$，得到预测结果 $\hat{y}$（Prediction），判断样本所属类别（?? Class）。

*   **假设函数的差异**：线性回归的假设函数为 $h_{\theta}(x^{(i)})=\sum_{j=1}^{d} \theta_{j} x_{j}=\theta^{T} x^{(i)}$，输出为连续值；而二分类任务需输出离散类别，因此需重新设计假设函数 $h_{\theta}(x)$，引入 Sigmoid 函数实现从连续输出到概率的映射。

## 第 42 页



*   **页面主题**：二元逻辑回归的假设函数（Binary Logistic Regression - Hypothesis Function）

*   **假设函数定义**：二元逻辑回归的假设函数为 $h_{\theta}(x)=g(\theta^{T} x)=\frac{1}{1+e^{-\theta^{T} x}}$，其中 $g(z)=\frac{1}{1+e^{-z}}$ 为 Sigmoid 函数（logistic function），$\theta^{T} x$ 为线性组合项。

*   **决策函数的可微性问题**：若直接使用符号函数（$h(x)=sign\left(\theta^{T} x\right)$）作为决策函数，当 $\theta^{T} x > 0$ 时输出 1，否则输出 0，该函数在 $\theta^{T} x = 0$ 处不可微（not differentiable），无法通过梯度下降等优化算法学习参数。

*   **替代方案：概率型假设函数**：采用可微的 Sigmoid 函数，将假设函数解释为样本属于正类的概率：$p_{\theta}(y=1 | x)=\frac{1}{1+exp \left(-\theta^{T} x\right)}$，属于负类的概率则为 $p_{\theta}(y=0 | x)=1 - p_{\theta}(y=1 | x)$。

*   **Sigmoid 函数的性质**：


*   当 $z \to \infty$（$\theta^{T} x \to \infty$）时，$g(z) \to 1$，样本属于正类的概率趋近于 1；
    
    *   当 $z \to -\infty$（$\theta^{T} x \to -\infty$）时，$g(z) \to 0$，样本属于正类的概率趋近于 0；
    
    *   函数值始终在（0,1）区间内（$h_{\theta}(x)$ is always bounded between 0 and 1），满足概率的取值要求。

*   **参数学习目标**：从训练数据 $\{(x^{(i)}, y^{(i)})\} (i=1, ..., n)$ 中学习参数 $\theta$，使假设函数输出的概率与样本真实标签尽可能一致。

## 第 43 页



* **页面主题**：二元逻辑回归的似然函数（Binary Logistic Regression - Likelihood Function）

* **概率模型与似然函数**：假设函数 $h_{\theta}(x)$ 对应样本属于正类的概率，因此可将样本的概率模型表示为：

  上述两个公式可合并为统一的似然表达式（Bernoulli 分布）：$P\left( y | x; \theta \right) =[h_{\theta }(x)]^{y}\cdot [1-h_{\theta }(x)]^{\, 1-y}$，其中 $y \in \{0,1\}$。


*   $P(y=1 | x ; \theta)=h_{\theta}(x)=\frac{1}{1+e^{-\theta^{T} x}}$（正类概率）；

*   $P(y=0 | x ; \theta)=1-h_{\theta}(x)=1-\frac{1}{1+e^{-\theta^{T} x}}$（负类概率）。

*   **训练数据的似然函数**：设训练样本 $\{(x^{(i)}, y^{(i)})\} (i=1, ..., n)$ 独立同分布（i.i.d.），则所有训练样本的联合似然函数为：$L(\theta )=P(Y|X; \theta )=\prod _{i=1}^{n}h_{\theta }(x^{(i)})^{y^{(i)}}\left[ 1-h_{\theta }(x^{(i)})\right] ^{1-y^{(i)}}$，其中 $X$ 为 $n \times d$ 的输入特征矩阵，$Y$ 为 $n$ 维标签向量。

*   **参数学习目标**：学习参数 $\theta$ 的目标是最大化似然函数 $L(\theta)$，即使得训练数据在当前参数下出现的概率最大，从而保证模型对训练数据的拟合效果。

## 第 44 页



* **页面主题**：二元逻辑回归的对数似然与优化（Binary Logistic Regression - Log-Likelihood & Optimization）

* **对数似然函数**：由于似然函数是乘积形式，直接最大化计算复杂且易出现数值下溢，因此通常最大化其对数形式（log-likelihood function），将乘积转化为求和，简化计算：

  $log L(\theta ) = \sum _{i=1}^{n}y^{(i)} log h_{\theta }(x^{(i)})+(1-y^{(i)})log [1-h_{\theta }(x^{(i)})]$。

* **梯度下降优化**：通过对对数似然函数关于参数 $\theta_j$ 求梯度，得到参数更新方向，使用梯度上升（最大化对数似然）或梯度下降（最小化负对数似然）进行参数优化。参数更新公式为：

  $\theta_{j}^{t+1}:=\theta_{j}^{t}+\alpha * \frac{1}{n} \sum_{i=1}^{n}\left[\left(y^{(i)}-h_{\theta}\left(x^{(i)}\right)\right) x_{j}^{(i)}\right]$，其中 $\alpha$ 为学习率，$t$ 为迭代次数，$\frac{1}{n} \sum_{i=1}^{n}\left[\left(y^{(i)}-h_{\theta}\left(x^{(i)}\right)\right) x_{j}^{(i)}\right]$ 为对数似然函数关于 $\theta_j$ 的梯度。

* **与线性回归的异同**：


*   相同点：参数更新公式形式相似，均基于 “误差（$y^{(i)}-h_{\theta}(x^{(i)})$）× 特征 $x_j^{(i)}$” 的累积；
    
    *   不同点：假设函数 $h_{\theta}(x^{(i)})$ 不同，逻辑回归使用 Sigmoid 函数（输出概率），线性回归使用线性函数（输出连续值）。

*   **闭式解的缺失**：与线性回归不同，逻辑回归的对数似然函数是非线性的，不存在解析形式的闭式解（closed form solution），必须通过梯度下降等迭代优化算法求解参数。

## 第 45 页



*   **页面主题**：从二分类到多分类的扩展思路（How to migrate from binary- to multi-class?）

*   **二分类的局限性**：二分类任务中，样本标签 $y^{(i)} \in \{0,1\}$（仅两个取值），模型只需区分 “是” 或 “否”；而多分类任务中，标签需扩展到多个取值（$y^{(i)} \in \{0,1,...,K-1\}$ 或 $\{1,2,...,K\}$），模型需从 $K$ 个类别中选择正确类别。

*   **标签扩展**：打破二分类标签的限制（Get rid of the restriction），允许标签 $y^{(i)}$ 从多个选项中取值（each $y^{(i)}$ can take values from multiple choices），设总类别数为 $K$，则 $y^{(i)} \in \{1,2,...,K\}$。

*   **模型修改方向**：

1.  假设函数：需从输出 “二分类概率” 扩展为输出 “$K$ 个类别概率”，且所有概率之和为 1；

2.  目标函数：需将二分类的似然函数扩展为多分类的似然函数，适应多标签场景；

3.  优化算法：需确保参数更新能有效最大化多分类似然函数，收敛到最优解。

*   **核心工具**：Softmax 回归是实现二分类到多分类扩展的核心模型，它通过 Softmax 函数实现多类别概率输出，是逻辑回归的自然推广。

## 第 46 页



*   **页面主题**：多分类中的 Softmax 回归（Using Softmax Regression in Multi-class case）

*   **Softmax 函数的引入**：


*   二分类中，Sigmoid 函数将线性输出映射到（0,1）区间，实现单类别概率输出；
    
    *   多分类中，需同时输出 $K$ 个类别概率，且满足 $\sum_{j=1}^{K} P(y=j | x; \theta) = 1$，因此引入 Softmax 函数进行概率归一化（Softmax function for normalization among possibilities）。

*   **二分类与多分类的函数对比**：


*   逻辑函数（二分类）：$P(y=1 | x ; \theta)=\frac{1}{1+e^{-\theta^{T} x}}$，输出正类概率，负类概率可通过 $1 - P(y=1 | x ; \theta)$ 得到；
    
    *   Softmax 函数（多分类）：$P(y=k | x ; \theta)=\frac{exp\left[(\theta^{(k)})^{T} x\right]}{\sum_{j=1}^{K} exp\left[(\theta^{(j)})^{T} x\right]}$（$k=1,...,K$），其中 $\theta^{(k)}$ 为第 $k$ 个类别的专属参数向量，输出每个类别的概率，且所有概率之和为 1。

*   **Softmax 回归的命名由来**：由于模型使用 Softmax 函数实现多类别概率输出，因此被称为 Softmax 回归（Softmax Regression），它是专门为多分类任务设计的监督学习模型。

## 第 47 页



* **页面主题**：Softmax 回归的似然函数（Softmax Regression - Likelihood Function）

* **类别概率假设**：Softmax 回归假设第 $i$ 个样本 $x^{(i)}$ 属于第 $k$ 个类别的概率为：$P\left(y^{(i)}=k | x^{(i)} ; \theta\right)=\frac{exp \left[\left(\theta^{(k)}\right)^{T} x^{(i)}\right]}{\sum_{j=1}^{K} exp \left[\left(\theta^{(j)}\right)^{T} x^{(i)}\right]}$，其中 $\theta = [\theta^{(1)}, \theta^{(2)}, ..., \theta^{(K)}]^T$ 为模型的所有参数（$K$ 个类别，每个类别对应一个 $d$ 维参数向量）。

* **指示函数的使用**：引入指示函数 $1\{y^{(i)}=k\}$（当 $y^{(i)}=k$ 时，函数值为 1；否则为 0），用于表示样本 $x^{(i)}$ 是否属于第 $k$ 个类别。

* **完整似然函数**：设训练样本 $\{(x^{(i)}, y^{(i)})\} (i=1, ..., n)$ 独立同分布，则所有训练样本的联合似然函数为：

  $L(\theta)=P(Y | X ; \theta)=\prod_{i=1}^{n} \prod_{k=1}^{K}\left[\frac{exp \left[\left(\theta^{(k)}\right)^{T} x^{(i)}\right]}{\sum_{j=1}^{K} exp \left[\left(\theta^{(j)}\right)^{T} x^{(i)}\right]}\right]^{\left\{y^{(i)}=k\right\}}$。

  该公式的含义是：对于每个样本 $i$ 和每个类别 $k$，若样本属于类别 $k$（$1\{y^{(i)}=k\}=1$），则似然函数乘以该样本属于类别 $k$ 的概率；否则乘以 1（无贡献），最终得到所有样本的联合概率。

* **对数似然函数（成本函数）**：对似然函数取对数，将乘积转化为求和，得到对数似然函数：

  $log L(\theta)=\sum_{i=1}^{n} \sum_{k=1}^{K} 1\left\{y^{(i)}=k\right\} log \frac{exp \left[\left(\theta^{(k)}\right)^{T} x^{(i)}\right]}{\sum_{j=1}^{K} exp \left[\left(\theta^{(j)}\right)^{T} x^{(i)}\right]}$。

  模型学习的目标是最大化该对数似然函数，或等价地最小化负对数似然函数（作为成本函数）。

## 第 48 页



*   **页面主题**：Softmax 回归与逻辑回归的关系（Relationship to Binary Logistic Regression）

*   **核心结论**：当多分类任务的类别数 $K=2$ 时，Softmax 回归退化为二元逻辑回归（softmax regression reduces to logistic regression），即 Softmax 回归是逻辑回归的推广形式（generalization of logistic regression）。

*   **数学推导（**** **** 时）**：

1. 当 $K=2$ 时，Softmax 回归的假设函数为：

   $h_{\theta}(x^{(i)}) = \frac{1}{exp \left[\left(\theta^{(1)}\right)^{T} x^{(i)}\right]+exp \left[\left(\theta^{(2)}\right)^{T} x^{(i)}\right]}\left[\begin{array}{c} exp \left[\left(\theta^{(1)}\right)^{T} x^{(i)}\right] \\ exp \left[\left(\theta^{(2)}\right)^{T} x^{(i)}\right] \end{array}\right]$；

2. 对分子分母同时除以 $exp \left[\left(\theta^{(1)}\right)^{T} x^{(i)}\right]$，化简得：

   $h_{\theta}(x^{(i)}) = \frac{1}{1 + exp \left[\left(\theta^{(2)} - \theta^{(1)}\right)^{T} x^{(i)}\right]}\left[\begin{array}{c} 1 \\ exp \left[\left(\theta^{(2)} - \theta^{(1)}\right)^{T} x^{(i)}\right] \end{array}\right]$；

3. 令 $\theta' = \theta^{(2)} - \theta^{(1)}$（新的参数向量），则假设函数进一步简化为：

   $h_{\theta}(x^{(i)}) = \left[\begin{array}{c} \frac{1}{1 + exp \left[(\theta')^{T} x^{(i)}\right]} \\ 1 - \frac{1}{1 + exp \left[(\theta')^{T} x^{(i)}\right]} \end{array}\right]$；

4. 此时，假设函数的第一个元素为样本属于类别 1 的概率（与逻辑回归的假设函数完全一致），第二个元素为属于类别 2 的概率，即完全恢复了二元逻辑回归的形式。

*   **推导意义**：该推导验证了 Softmax 回归与逻辑回归的一致性，表明多分类模型与二分类模型并非完全独立，而是存在自然的扩展关系，有助于理解多分类模型的本质。

## 第 49 页



*   **页面主题**：无监督聚类：K 均值算法引入（K-means Clustering - Introduction）

*   **监督与无监督的区别**：


    *   Softmax 回归属于监督学习（supervised），需依赖带有真实标签（groundtruth labels）的训练数据，学习从输入到类别的映射；
    
    *   当无标签数据（no groundtruth labels）时，需通过无监督学习（unsupervised）方法，仅基于数据的内在特征（如相似度、距离）将样本划分为不同类别，即聚类（Clustering）任务。

*   **聚类任务示例**：


    *   单高斯分布（Single Gaussian）的数据无法直接区分类别，而混合高斯分布（Mixture of two Gaussians）的数据可通过聚类划分为两个簇；
    
    *   手写数字聚类：给定一系列无标签的手写数字图像（如数字 5、6、2），需判断哪些图像属于同一数字类别（which digits come from which cluster），且训练数据中无任何标签信息。

*   **K 均值聚类的定位**：K 均值（K-means）是最经典、最常用的无监督聚类算法之一，它通过预先指定聚类数量 $K$，将数据划分为 $K$ 个互不重叠的簇（cluster），使同一簇内样本相似度高，不同簇间样本相似度低。

## 第 50 页



*   **页面主题**：K 均值聚类的核心步骤（K-means Clustering - Core Steps）

*   **算法前提假设**：假设已知聚类中心（cluster centroids）$\mu_1, \mu_2, ..., \mu_K \in \mathbb{R}^d$（$d$ 为特征维度），每个聚类中心代表一个簇的 “中心位置”。

*   **步骤 1：样本分配（Assignment Step）**：为每个样本 $x^{(i)}$ 分配到与其距离最近的聚类中心对应的簇。使用欧氏距离（Euclidean distance）衡量样本与聚类中心的相似度，分配规则为：$z^{(i)}=argmin_{j}\left\| x^{(i)}-\mu_{j}\right\| ^{2}$，其中 $z^{(i)} \in \{1, ..., K\}$ 表示样本 $x^{(i)}$ 所属的簇索引，$\left\| x^{(i)}-\mu_{j}\right\| ^{2}$ 为样本 $x^{(i)}$ 与第 $j$ 个聚类中心 $\mu_j$ 的平方欧氏距离。

*   **步骤 2：聚类中心更新（Update Step）**：根据步骤 1 的样本分配结果，重新计算每个簇的聚类中心，使新中心为该簇内所有样本的均值（mean）。更新规则为：$\mu_{j}=\frac{\sum_{i=1}^{n} 1\left\{z^{(i)}=j\right\} x^{(i)}}{\sum_{i=1}^{n} 1\left\{z^{(i)}=j\right\}}$，其中 $1\left\{z^{(i)}=j\right\}$ 为指示函数（样本 $i$ 属于簇 $j$ 时为 1，否则为 0），分子为簇 $j$ 内所有样本的特征之和，分母为簇 $j$ 内的样本数量。

*   **算法本质**：通过 “分配 - 更新” 的迭代过程，不断调整样本所属簇和聚类中心位置，使簇内样本的平方误差和（SSE，Sum of Squared Errors）最小化，最终达到聚类稳定状态。

## 第 51 页



*   **页面主题**：K 均值聚类的完整算法流程（K-means Clustering - Complete Algorithm）

*   **步骤 1：初始化聚类中心**：随机初始化 $K$ 个聚类中心 $\mu_1, \mu_2, ..., \mu_K \in \mathbb{R}^d$。初始化方法通常有两种：


    *   随机选择 $K$ 个训练样本作为初始聚类中心（确保初始中心来自真实数据分布）；
    
    *   在样本特征的取值范围内随机生成 $K$ 个点作为初始中心（需注意范围合理性，避免与数据分布偏差过大）。

*   **步骤 2：迭代执行 “分配 - 更新”**：重复执行以下两个步骤，直到算法收敛：

1.  样本分配（Assignment）：对每个样本 $i \in \{1, ..., n\}$，根据平方欧氏距离最小原则，确定其所属簇：$z^{(i)}=argmin_{j}\left\| x^{(i)}-\mu_{j}\right\| ^{2}$；

2.  中心更新（Update）：对每个簇 $j \in \{1, ..., K\}$，计算簇内所有样本的均值，作为新的聚类中心：$\mu_{j}=\frac{\sum_{i=1}^{n} 1\left\{z^{(i)}=j\right\} x^{(i)}}{\sum_{i=1}^{n} 1\left\{z^{(i)}=j\right\}}$。

*   **收敛判定条件**：算法收敛的标志通常为以下两种情况之一：

1.  聚类中心的变化量小于预设阈值（如 $\left\| \mu_j^{t+1} - \mu_j^t \right\| < \epsilon$，$\epsilon$ 为极小正数），说明中心位置已稳定；

2.  所有样本的簇分配结果不再变化（$z^{(i)}^{t+1} = z^{(i)}^t$ 对所有 $i$ 成立），说明聚类结构已稳定。

*   **算法特点**：K 均值聚类算法简单易懂、计算效率高，适用于大规模数据集，但对初始聚类中心敏感，且需预先指定聚类数量 $K$。

## 第 52 页



*   **页面主题**：聚类的其他解决方案（Alternative Solutions of Clustering）

*   **高斯混合模型（GMM）**：高斯混合模型（Gaussian Mixture Model, GMM）是一种基于概率的聚类方法（probabilistic way of clustering），与 K 均值的硬分配（样本明确属于某一簇）不同，GMM 采用软分配（soft assignment）策略：


*   输出每个样本属于各个簇的概率（置信度）：$w_{j}^{(i)}=p\left(z^{(i)}=j | x^{(i)} ; \phi, \mu, \Sigma\right)$，其中 $w_{j}^{(i)}$ 为样本 $i$ 属于簇 $j$ 的后验概率，$\phi$ 为簇先验概率，$\mu$ 为簇均值，$\Sigma$ 为簇协方差矩阵；
    
    *   估计每个簇的概率分布：GMM 假设每个簇的数据服从多元高斯分布，通过学习得到每个簇的分布参数（均值 $\mu_j$ 和协方差 $\Sigma_j$），可进一步分析簇的形状、大小和方向。

*   **GMM 与 K 均值的对比**：

*   K 均值：非概率模型，硬分配，仅考虑簇均值，对球形分布数据效果好；
    
*   GMM：概率模型，软分配，考虑簇的均值、协方差和先验概率，对非球形、不同大小的簇适应性更强，但计算复杂度更高。

*   **GMM 的不稳定性**：GMM 的训练结果受初始参数影响较大，多次随机初始化（如 8 次尝试，8 attempts with random initializations）可能得到不同的聚类结果和损失值（costs），部分尝试可能无法良好收敛（Not every attempt converge well），因此实际应用中通常多次运行并选择最优结果。

*   **扩展学习资源**：如需深入学习 GMM，可参考相关教程和库文档，如 [https://lukapopijac.github.io/gaussian-mixture-model/](https://lukapopijac.github.io/gaussian-mixture-model/)、[https://scikit-learn.org/stable/modules/mixture.html](https://scikit-learn.org/stable/modules/mixture.html) 等。

## 第 53 页



*   **页面主题**：特征工程 - II 总结与后续展望（Conclusion & Future Outlook）

*   **特征工程 - II 核心内容回顾**：

1.  特征选择标准：特征需具备鲁棒性（抗数据变化）、非冗余性（无重复信息）和区分性（类间差异大），并通过花卉分类、房价预测等示例验证；

2.  模型应用：学习了如何利用特征构建线性回归（解决回归问题）、二元分类（SVM、逻辑回归）和多分类（Softmax 回归、K 均值聚类）模型，覆盖监督学习和无监督学习场景；

3.  优化方法：掌握了梯度下降、随机梯度下降、SMO 等优化算法，以及动态学习率、核技巧等提升模型性能的关键技术。

*   **特征工程的局限性**：当前的特征工程依赖人工设计（human engineered features），需 “祈祷” 设计的特征恰好适合当前任务（“pray” that these human engineered features are “just right”）。但实际应用中，特征往往难以满足线性可分条件，即使通过核技巧扩展到高维空间，也可能因特征表达能力不足导致模型性能受限。

*   **后续学习方向**：为解决人工特征的局限性，需引入深度学习（Deep Learning）技术（Lecture 5）。深度学习通过神经网络的多层结构，可自动从数据中学习多层次、复杂的特征表示（extract more flexible features），无需人工设计，大幅提升了模型对复杂任务的处理能力，如图像识别、自然语言处理等。