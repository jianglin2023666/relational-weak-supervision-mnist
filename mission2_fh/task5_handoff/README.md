# 提供给任务5的全监督基线数据

本目录只包含任务2的100轮、batch64实验。

| 文件 | 内容 |
|---|---|
| `supervised_summary.csv` | 一行模型级汇总，适合统一对比表 |
| `supervised_per_seed.csv` | 五seed结果、最佳轮次、训练预算及训练历史摘要 |
| `per_class_metrics.csv` | 五seed各十类的precision、recall、F1和样本数 |
| `embedding_sample_indices_seed42.csv` | 2000张embedding分析图片的官方MNIST测试集索引及标签 |
| `embedding_metrics_seed42.json` | 固定seed42的PCA、rho及Discovered order |

## 指标口径

- 所有名称以 `_percent` 结尾的准确率、precision、recall、F1均在0至100之间。
- 标准差字段以 `_percentage_points` 结尾，使用五seed样本标准差，`ddof=1`。0.077表示0.077个百分点。
- `pc1_explained_variance_ratio_seed42` 在0至1之间，0.230887表示23.0887%。
- Spearman rho在-1至1之间。当前项目按raw rho符号决定是否翻转PC1，最终rho非负；JSON同时保留raw rho和翻转标记。
- `relation_train_pairs`、`shots_per_class`、`relation_test_accuracy_percent` 的空值表示本全监督任务不适用，不能填为0参与平均。
- `absolute_training_label_count=50000` 指带绝对数字标签的训练图片数，不含用来选择模型的验证标签，也不含测试标签。评估监督成本时需同时披露验证标签的使用。

## 对比约定

训练图片、关系对、绝对标签是不同单位。50k或500k关系对方案都可以引用本条全监督参考，不需要把全监督条目伪装成两次不同的数据规模实验。

本次训练每seed为100轮、batch64、78200次优化器更新。其他方法同时报告训练步数、关系对数、绝对标签数、分类头训练轮次和冻结策略。相同seed编号不意味着不同架构具有完全相同的随机过程。

数字识别比较digit test accuracy，关系预测另报relation test accuracy。两者不能放在同一列直接排名。

绘图时请在各自seed42、按既定验证指标选择的checkpoint上，使用本目录保存的相同2000个测试图片索引。每类200张，保持t-SNE参数一致；具体参数见上级 `figures/embedding_seed42/embedding_metrics.json` 和 `source_snapshot/src/visualization/embedding_analysis.py`。

本目录rho等embedding指标只来自seed42，不是五seed均值或标准差。不要把它们复制成五次独立观测后计算误差棒。

原始五seed数据见上级 `training_results/`；t-SNE坐标、PCA投影和128维embedding见上级 `figures/embedding_seed42/`；完整测试预测见 [seed42_test_predictions.csv](../predictions/seed42_test_predictions.csv)。
