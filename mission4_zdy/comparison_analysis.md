# 任务四实验对比：Siamese 1-shot 与 3-shot

## 设置

两组实验均使用 50k 关系对、关系阶段 5 epochs、分类头 100 epochs、seed 42–46。每类带标签图片数分别为 1 和 3。关系模型按验证集关系准确率选 checkpoint；少样本阶段冻结 backbone，分类头按固定最后一个 epoch 评估。五 seed 汇总中的标准差为样本标准差（ddof=1）。两组均在 CPU 上运行。

## 汇总结果

| 指标 | 1-shot | 3-shot |
|---|---:|---:|
| Relation test accuracy | 98.943% ± 0.121% | 98.943% ± 0.121% |
| Digit test accuracy | 96.132% ± 0.784% | 96.920% ± 0.131% |
| Digit accuracy range | 95.12%–97.00% | 96.76%–97.09% |
| Mean training time per seed | 118.71 ± 8.54 s | 114.97 ± 9.07 s |

3-shot 相比 1-shot 的数字准确率均值提高 **0.788 个百分点**，标准差从 **0.784** 降至 **0.131 个百分点**。五个 seed 中，3-shot 有四个 seed 高于对应的 1-shot，seed 45 低 0.04 个百分点。关系准确率完全相同（每个 seed 的关系验证/测试指标也一致），因为标签数只在关系预训练结束后改变。

结论应限定在本次 MNIST 和当前训练设置：增加每类两张标签图后，少样本分类平均准确率略有提高，seed 间波动明显变小；这组结果不能推出其他数据集也有相同幅度的提升。

`comparison_1shot_3shot.svg` 和 PNG 版本展示五 seed 均值及样本标准差误差线。为显示小幅差异，图中纵轴范围经过收窄；定量比较请以图中标值和上表为准。

## Seed 42 的 embedding 与混淆矩阵

每类抽取 200 张测试图片，共 2,000 个 embedding。两种 shot 设置的关系 backbone 产生相同的 embedding 分析结果：PCA 第一主成分解释 65.46% 方差，Spearman ρ = **0.881**，`Discovered order` 为 **0 → 1 → 2 → 3 → 4 → 5 → 8 → 9 → 6 → 7**。因此两组 t-SNE/PCA 图相同是符合两阶段训练流程的：分类标签数量不会回头更新冻结的 backbone。

seed 42 的数字准确率从 1-shot 的 **95.12%（9512/10000）** 提高到 3-shot 的 **96.82%（9682/10000）**。混淆矩阵里，数字 6 被错分为 5 的数量从 190 降到 60，数字 9 被错分为 8 的数量从 76 降到 32。它们是 seed 42 的具体例子；判断总体效果应以五 seed 均值和标准差为准。

Spearman ρ 描述 embedding 的 PCA 主轴与数字标签之间的整体单调关系；混淆矩阵反映分类头的类别预测。ρ 较高不意味着每个类别边界都清晰，也不能单独说明某类错误的成因。

## 对没有自然类别顺序的数据集的适用性

Siamese 关系头通过拼接两张图片的 embedding 并学习二元关系，不需要把类别编码成 `0<1<...<9`。迁移到没有天然大小顺序的数据集时，可将关系标签改为“同类/不同类”或“相似/不相似”，因此模型结构更通用。当前结果来自 MNIST 的数字大小比较，只说明了这种结构上的适用可能，不能据此宣称其他数据集上已有性能验证。

## 任务五可直接取用

- `task5_comparison.csv`：1-shot、3-shot 的并列表格字段和统计数值。
- `1shot/training_results/runs.csv` 与 `3shot/training_results/runs.csv`：五 seed 逐次结果。
- `1shot/embedding_seed42/embedding_metrics.json` 与 `3shot/embedding_seed42/embedding_metrics.json`：Spearman、投影顺序、类别统计等数据。
