# Deep_Learning Final_prj

任务：MNIST手写数字识别

**研究背景：**

深度学习模型在各种**计算机视觉任务**中取得了卓越的性能。然而，大多数成功的方法依赖于对大型标记数据集的完**全监督学习**，获取这样的标签通常是一项成本和劳动密集型任务。**弱监督学习**旨在通过利用更便宜、更丰富的监督形式来减少对大量标记数据的依赖。

在这个项目中，我们探索了**一种特定形式的弱监督：从图像对之间的关系比较中学习**。我们不提供绝对的类标签，而是**只指定图像对之间的相对顺序**（例如，“图像 A 中的数字大于图像 B 中的数字”）。这种类型的监督在**难以获得精确标签**但**相对比较更容易生成或自然可用**的领域中尤其有价值。

## 共同实验规则

为了保证不同方案可以公平比较，任务二至任务五统一遵守以下规则：

- 默认使用`42、43、44、45、46`五个随机种子，不只展示最好的一次结果。
- 训练参数优先在`configs/`对应的JSON文件中修改，不要直接改训练代码。
- 关系任务报告`relation test accuracy`，数字分类任务报告`digit test accuracy`。
- 最终结果使用`training_results/`中的`runs.csv`和`summary.json`，报告均值和标准差。
- 对比1-shot和3-shot时，每个类别分别只能使用1张或3张带标签图片。
- PPT中的结构图需要标明输入、共享的`BaseModel`、任务头、损失函数和输出。
- 绘制embedding图前安装可视化依赖：`python -m pip install -r requirements-visualization.txt`。
- 四种模型统一使用`seed_42`和每类200张测试图片进行embedding分析，不根据测试集准确率挑选seed。
- `embedding_analysis.py`会输出`t-SNE`、PCA小提琴图、Spearman ρ、`Discovered order`和对应JSON/CSV数据。

## 任务一：构建基础模型`BaseModel`

### 作用

给后续四种方法提供完全相同的视觉特征提取器。统一骨干网络后，实验差异主要来自监督信息和训练方式，而不是模型规模不同。

### 核心实现

对应代码：`src/models/base_model.py`

输入是一批形状为`[B, 1, 28, 28]`的MNIST图片，输出是`[B, 128]`的特征向量。网络的数据形状变化为：

```text
[B, 1, 28, 28]
→ Conv(1→32) + ReLU + MaxPool
→ [B, 32, 14, 14]
→ Conv(32→64) + ReLU + MaxPool
→ [B, 64, 7, 7]
→ Conv(64→64) + ReLU
→ Flatten(3136)
→ Linear(3136→64) + ReLU + Dropout
→ Linear(64→128)
→ 128维embedding
```

前两组卷积和池化逐步提取局部形状并降低空间尺寸；第三个卷积层继续组合高级特征；全连接层将二维特征压缩成固定长度表示。`BaseModel`本身不输出数字类别，也没有独立训练入口，它必须连接具体任务头并通过对应损失函数训练。

### 推荐操作

1. 阅读`__init__()`和`forward()`，整理每层输入、输出尺寸和参数作用。
2. 画一张从MNIST图片到128维embedding的结构图。
3. 用下面的命令检查前向传播输出：

```powershell
python -c "import torch; from src.models.base_model import BaseModel; m=BaseModel(); print(m(torch.randn(4,1,28,28)).shape)"
```

预期输出是`torch.Size([4, 128])`。

4. 简单谈谈所有方案共享相同的骨干网络的好处

### 建议交付内容

- 一张网络结构图。
- 一张各层输出尺寸和功能表。
- 1～2页PPT说明共享骨干网络对公平比较的意义。

## 任务二：构建全监督基线`SupervisedModel`

### 作用

使用全部数字标签建立性能上界参考。在相同骨干网络和训练设置下，弱监督方案应尽量接近全监督基线。

### 核心实现

对应文件：

- `src/models/supervised_model.py`
- `src/training/train_supervised.py`
- `configs/supervised.json`

模型在128维embedding后增加`Linear(128→10)`分类头。训练时输入单张图片和真实数字标签，使用交叉熵损失同时更新`BaseModel`和分类头。每个epoch在验证集上评测，并根据验证集数字分类准确率保存最佳模型。

### 推荐操作

1. 画出“单张图片→BaseModel→10类logits→交叉熵损失”的流程图。
2. 检查`configs/supervised.json`中的epoch、seed、batch size和学习率。
3. 执行完整训练（50k,500k）：

```powershell
python -m src.training.train_supervised
```

4. 从`summary.json`提取五次实验的测试准确率均值和标准差。
5. 使用`seed_42`中按验证集选择的最佳权重生成embedding分析结果：

```powershell
python -m src.visualization.embedding_analysis `
  --model supervised `
  --checkpoint "training_results/supervised/<实验目录>/seed_42/best_supervised_model.pth" `
  --output-dir "figures/supervised_seed42"
```

6. 观察t-SNE中的类别簇是否清晰，并解释为什么分类效果很好但PCA投影的数字顺序可能较弱。
7. 使用同一`seed_42`实验生成数字分类混淆矩阵：

```powershell
python -m src.visualization.confusion_matrix_analysis `
  --input "training_results/supervised/<实验目录>/seed_42/digit_confusion_matrix.csv" `
  --output-dir "figures/supervised_seed42/confusion_matrix" `
  --title "Supervised - Seed 42"
```

脚本会生成原始计数图、按真实类别归一化的比例图和准确率JSON。全监督应该不会混淆

### 建议交付内容

- 一张全监督训练流程图。
- 五个seed的准确率与`mean ± std`。
- 混淆矩阵图。
- t-SNE图、PCA小提琴图、Spearman ρ和`Discovered order`。
- 说明该结果为什么可以作为后续方案的比较基准。
- 1～2页PPT说明你得到的实验结果
- 配合任务5整理数据，做统一对比

## 任务三：构建基础弱监督模型`WeaklySupervisedModel`

### 作用

实现论文中的结构化关系监督，作为后续改进方案的基础。训练阶段只使用“图片A中的数字是否大于图片B”这一关系标签，不直接提供数字类别。

### 核心实现

对应文件：

- `src/models/weakly_supervised_model.py`
- `src/training/train_weakly_supervised.py`
- `configs/weakly_supervised.json`

一对图片分别经过共享的`BaseModel`和同一个10分类头，得到两个类别概率分布`p(A=i)`和`p(B=j)`。模型显式计算：

```text
P(A > B) = Σ p(A=i) · p(B=j)，其中i > j
```

关系概率与0/1关系标签使用二元交叉熵训练。测试时拿掉成对比较过程，直接使用单张图片的10类概率进行数字识别。该方法明确规定了类别之间存在`0<1<...<9`的顺序，因此结构约束较强。

### 推荐操作

1. 结合论文Figure 3(a)画出模型结构图、共享权重、类别分布和关系概率的计算过程。
2. 先运行50k关系对实验：

```powershell
python -m src.training.train_weakly_supervised
```

3. 如需测试500k，先生成数据，再把配置中的`train_size`和`experiment_name`改成500k：

```powershell
python -m data.mnist.download_MNIST --train-sizes 500k
```

4. 同时比较关系测试准确率和数字测试准确率。不能只根据关系准确率判断模型是否学会了数字分类。
5. 使用`seed_42`的最佳关系模型权重生成embedding分析结果：

```powershell
python -m src.visualization.embedding_analysis `
  --model weakly_supervised `
  --checkpoint "training_results/weakly_supervised/<实验目录>/seed_42/best_weakly_supervised_model.pth" `
  --output-dir "figures/weakly_supervised_seed42"
```

6. 画出五个seed的数字准确率分布，并选择最好和最差seed的混淆矩阵进行对比。分别替换下面命令中的`<seed>`后运行两次：

```powershell
python -m src.visualization.confusion_matrix_analysis `
  --input "training_results/weakly_supervised/<实验目录>/seed_<seed>/digit_confusion_matrix.csv" `
  --output-dir "figures/weakly_supervised_seed_<seed>/confusion_matrix" `
  --title "Weakly Supervised - Seed <seed>"
```

脚本会生成原始计数图、按真实类别归一化的比例图和准确率JSON。对比最好和最差seed是否出现类别整体错位，例如真实数字3被集中预测成4。
7. 结合t-SNE、Spearman ρ和类别投影分布，分析“关系准确率稳定，但数字准确率可能波动较大”的原因：关系标签只约束相对顺序，不能充分确定每个特征簇与绝对数字类别的对应边界。

### 建议交付内容

- 一张结构化关系概率计算图。
- 关系准确率和数字准确率的五seed统计。
- 最好/最差seed混淆矩阵。
- t-SNE图、PCA小提琴图、Spearman ρ和`Discovered order`。
- 对高关系准确率不一定带来高分类准确率的解释。
- 1～2页PPT说明你得到的实验结果
- 配合任务5整理数据，做统一对比

## 任务四：构建`SiameseModel`

### 作用

研究更弱结构约束的关系学习。当前MNIST实验仍使用`A>B`关系标签，但模型不再显式规定10个数字类别及其大小计算公式，而是让关系头从两个embedding中自行学习判断规则。

### 核心实现

对应文件：

- `src/models/siamese_model.py`
- `src/training/train_siamese.py`
- `configs/siamese.json`

训练分成两个阶段：

```text
阶段一：A、B → 共享BaseModel → 拼接两个embedding → MLP关系头 → P(A>B)
阶段二：加载最佳关系模型的BaseModel → 冻结骨干网络 → 用1-shot或3-shot训练数字分类头
```

与任务三相比，Siamese关系头不要求关系必须由10类概率和固定顺序计算，因此可以进一步扩展到“是否同类”“是否相似”等其他二元关系。

### 推荐操作

1. 结合论文Figure 3(b)画出双分支共享权重结构，并标明两条分支不是两个独立CNN。
2. 解释embedding拼接和MLP关系头如何代替任务三中的固定大小关系公式。
3. 执行默认1-shot实验：

```powershell
python -m src.training.train_siamese
```

4. 将`configs/siamese.json`中的`shots_per_class`改为3并修改实验名称，再运行3-shot实验。
5. 比较1-shot与3-shot数字准确率，也要报告关系测试准确率。
6. 使用关系预训练阶段的最佳权重生成embedding分析结果：

```powershell
python -m src.visualization.embedding_analysis `
  --model siamese `
  --checkpoint "training_results/siamese/<实验目录>/seed_42/best_siamese_relation_model.pth" `
  --output-dir "figures/siamese_seed42"
```

7. 与任务三使用相同关系数据量（50k,500k）和五个seed，结合ρ和`Discovered order`分析结构灵活性带来的收益与代价。
8. 使用`seed_42`少样本分类结果生成混淆矩阵：

```powershell
python -m src.visualization.confusion_matrix_analysis `
  --input "training_results/siamese/<实验目录>/seed_42/digit_confusion_matrix.csv" `
  --output-dir "figures/siamese_seed42/confusion_matrix" `
  --title "Siamese 1-shot - Seed 42"
```

脚本会生成原始计数图、按真实类别归一化的比例图和准确率JSON。比较1-shot和3-shot时应分别绘图，观察增加绝对标签后哪些类别的错分减少。
9. 配合任务5整理数据，做统一对比

### 建议交付内容

- 一张Siamese两阶段训练图。
- 1-shot和3-shot的`mean ± std`。
- t-SNE图、PCA小提琴图、Spearman ρ和`Discovered order`。
- 说明该结构为何更容易迁移到没有自然类别顺序的数据集。
- 1～2页PPT说明你得到的实验结果

## 任务五：构建联合监督模型`JointRelationalClassifier`

### 作用

验证“关系监督 + 极少量绝对类别标签”能否明显缩小与全监督基线的差距。少量标签为关系学习得到的特征簇提供类别锚点，使类别边界和数字编号更加明确。

### 核心实现

对应文件：

- `src/models/joint_relational_classifier.py`
- `src/training/train_joint_relational.py`
- `configs/joint_relational.json`

关系图片对和少量带标签图片共享同一个`BaseModel`与分类头，并在同一次训练中优化：

```text
L_total = L_relation + λ · L_classification
```

其中`L_relation`学习`A>B`关系，`L_classification`使用每类1张或3张带标签图片学习绝对类别，`λ`由配置中的`classification_weight`控制。它和Siamese的关键区别是：联合监督从训练开始就同时更新骨干网络，而Siamese先做关系预训练，再冻结骨干训练分类头。

### 推荐操作

1. 画出包含关系数据流、少量标签数据流以及两个损失合并的模型整体架构。
2. 执行默认的1-shot、`λ=0.5`实验：

```powershell
python -m src.training.train_joint_relational
```

3. 将`shots_per_class`改为3，重复3-shot实验。
4. 保持seed和关系数据量（50k,500k）与Siamese一致，比较两种方法的数字准确率和方差。
5. 时间允许时，可测试`classification_weight=0.1、0.5、1.0`，观察关系监督和类别监督之间的权衡。
6. 使用联合训练的最佳权重生成embedding分析结果：

```powershell
python -m src.visualization.embedding_analysis `
  --model joint_relational `
  --checkpoint "training_results/joint_relational/<实验目录>/seed_42/best_joint_relational_model.pth" `
  --output-dir "figures/joint_relational_seed42"
```

7. 将联合监督的t-SNE、ρ和`Discovered order`与任务三、任务四对比，分析少量绝对标签为什么能减少类别错位与跨seed波动。
8. 使用`seed_42`数字分类结果生成混淆矩阵：

```powershell
python -m src.visualization.confusion_matrix_analysis `
  --input "training_results/joint_relational/<实验目录>/seed_42/digit_confusion_matrix.csv" `
  --output-dir "figures/joint_relational_seed42/confusion_matrix" `
  --title "Joint Relational 1-shot - Seed 42"
```

脚本会生成原始计数图、按真实类别归一化的比例图和准确率JSON。对不同`classification_weight`或shot数分别绘图，检查少量绝对监督是否减少类别错位。

### 建议交付内容

- 一张联合监督模型与损失函数结构图。
- 1-shot和3-shot结果。
- t-SNE图、PCA小提琴图、Spearman ρ和`Discovered order`。
- 与全监督、基础弱监督、Siamese的统一对比表。
- 对`classification_weight`作用及联合训练优势的解释。
- 1～2页PPT说明你得到的实验结果

## 任务六(Leader)：使用Siamese模型尝试CIFAR-10任务 / 汇总PPT+演讲

推荐任务：这里我再想想怎么做 :-(

![image-20261109173010042](E:\AAASiSi\2026秋课程\CS280_DeepLearning\Final_prj\Deep_Learning Final_prj.assets\image-20261109173010042.png)

## 项目运行环境与GPU配置

### 核心依赖

项目只保留四个直接依赖：

- `torch==2.7.0`
- `torchvision==0.22.0`
- `numpy==2.2.6`
- `tqdm==4.67.1`

`requirements.txt`不包含Jupyter、绘图工具、间接依赖或macOS专用包，因此Windows、Linux和macOS可以共用。Python标准库（如`csv`、`json`和`pathlib`）不需要额外安装。

### 已验证的本机(shiyu)环境

- 操作系统：Windows
- Python：3.12.13
- GPU：NVIDIA GeForce RTX 5050 Laptop GPU
- 显存：8151 MiB（约8 GB）
- NVIDIA驱动：621.01
- PyTorch：2.7.0+cu128
- Torchvision：0.22.0+cu128
- PyTorch CUDA Runtime：12.8
- cuDNN：9.7.1
- CUDA可用：是

### 创建虚拟环境

在`cs231n_final_project`目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 本机NVIDIA GPU安装方式

先安装CUDA 12.8版本的PyTorch：

```powershell
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

PyTorch CUDA wheel自带所需CUDA运行库，本机仍需正确安装NVIDIA驱动。其他设备可以直接执行`python -m pip install -r requirements.txt`，具体可用后端由PyTorch安装版本决定。

### 训练设备选择

训练程序按以下顺序自动选择设备：

1. NVIDIA CUDA
2. Apple MPS
3. CPU

本机训练配置建议保持：

```json
{
  "batch_size": 128,
  "num_workers": 0,
  "deterministic": true
}
```

Siamese和联合监督配置使用各自的`relation_batch_size`。`num_workers=0`适合Windows，并可避免500k关系数据在多个worker进程间产生额外内存开销。

检查CUDA是否生效：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

预期输出包含：

```text
True
NVIDIA GeForce RTX 5050 Laptop GPU
```
